"""
Part 3: Multi-Agent Chatbot — all agents and Head_Agent controller.
Use with: from part3_agents import Head_Agent, Obnoxious_Agent, ...
"""
from openai import OpenAI

# ----- Politeness_Agent: must pass first; no topic-check until the message is polite -----
DEFAULT_POLITENESS_PROMPT = """You judge whether the user's message is polite and respectful.

Consider IMPOLITE: insults, rudeness, disrespect, profanity, threats, or offensive language. A message that asks a valid question but includes insults or rude phrasing is impolite.
Consider POLITE: respectful questions, greetings, or requests even if they mention any topic.

Output ONLY one word: "Yes" if the message is polite and respectful, or "No" if it is impolite.

User message:
{query}"""


class Politeness_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.model = "gpt-4.1-nano"
        self.prompt_template = DEFAULT_POLITENESS_PROMPT

    def set_prompt(self, prompt):
        self.prompt_template = prompt

    def is_polite(self, query: str) -> bool:
        prompt = self.prompt_template.format(query=query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            return text.startswith("YES")
        except Exception:
            return True  # on error, allow through (obnoxious will catch if needed)


# ----- Obnoxious_Agent (no LangChain) -----
DEFAULT_OBNOXIOUS_PROMPT = """You are a content moderation agent. Your task is to determine if the user's message is obnoxious.

Consider as obnoxious: insults, profanity, hate speech, harassment, threats, or clearly inappropriate/offensive language directed at the assistant or others.

Output ONLY a single word: "Yes" if the message is obnoxious, or "No" if it is not.

User message:
{query}"""


class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.model = "gpt-4.1-nano"
        self.prompt_template = DEFAULT_OBNOXIOUS_PROMPT

    def set_prompt(self, prompt):
        self.prompt_template = prompt

    def extract_action(self, response) -> bool:
        if response is None:
            return False
        text = response.strip().upper()
        return text.startswith("YES")

    def check_query(self, query):
        prompt = self.prompt_template.format(query=query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            return self.extract_action(resp.choices[0].message.content)
        except Exception:
            return False


# ----- Relevant_Documents_Agent (no LangChain) -----
DEFAULT_RELEVANCE_PROMPT = """You are a relevance judge. Given a user query and retrieved document snippets, decide if these documents can help answer the query.

User query: {query}

Retrieved documents:
{docs}

Answer ONLY "Yes" or "No".
- Say "Yes" if the documents discuss the concepts or topics in the query (even if the answer is spread across snippets, or if they cover "A" and "B" separately when the query asks for "differences between A and B").
- Say "No" only if the documents are clearly off-topic or useless (e.g. about something unrelated).

Your answer:"""


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.prompt_template = DEFAULT_RELEVANCE_PROMPT

    def set_prompt(self, prompt):
        self.prompt_template = prompt

    def get_relevance(self, conversation) -> str:
        query, raw = None, []
        if isinstance(conversation, dict):
            query = conversation.get("query", "")
            raw = conversation.get("docs", [])
        elif isinstance(conversation, (list, tuple)) and len(conversation) >= 2:
            query, raw = conversation[0], conversation[1]
        else:
            return "No"
        docs = []
        for d in raw:
            if isinstance(d, str):
                docs.append(d)
            elif hasattr(d, "page_content"):
                docs.append(d.page_content)
            else:
                docs.append(str(d))
        docs_text = "\n---\n".join(docs) if docs else "(no documents)"
        prompt = self.prompt_template.format(query=query or "", docs=docs_text)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip().upper()
            return "Yes" if content.startswith("YES") else "No"
        except Exception:
            return "No"


# ----- Context_Rewriter_Agent -----
REPHRASE_PROMPT = """You rephrase the user's latest message into one clear, standalone question using the conversation history.

Rules:
- If the latest message is vague (e.g. "Tell me more", "Can you give an example?", "What about that?", "Explain further"), you MUST use the previous user question and the assistant's answer to form a concrete question. Example: if the user previously asked "What is logistic regression?" and the assistant explained it, then "Tell me more" should become "Tell me more about logistic regression" or "What are more details about logistic regression?"
- If the latest message is already clear and specific, return it unchanged.
- Output ONLY the rephrased question, nothing else. No explanation.

Conversation history:
{history}

User's latest message: {latest}

Rephrased question:"""


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.prompt_template = REPHRASE_PROMPT

    def rephrase(self, user_history, latest_query):
        if not latest_query or not user_history:
            return latest_query or ""
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in user_history[-6:]])
        prompt = self.prompt_template.format(history=history_text, latest=latest_query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )
            out = (resp.choices[0].message.content or "").strip()
            return out if out else latest_query
        except Exception:
            return latest_query


# ----- Query_Agent -----
TOPIC_PROMPT = """Does the user's query seem to be a real question about a textbook or course (e.g. machine learning, statistics)? Include questions about theorems, proofs, chapters, definitions, or course material. Do NOT answer "Yes" merely because the text mentions "machine learning"—the query must be a genuine question about the material. Answer only "Yes" or "No".

Query: {query}

Answer:"""


class DocResult:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


def _embed_text(client, text, model="text-embedding-3-small"):
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


COMPARISON_EXTRACT_PROMPT = """The user asked a question that may ask for differences or comparison between two concepts (e.g. "What are the differences between A and B?"). If so, output exactly two short search phrases that we can use to search a textbook, one per line, no numbering. Example: for "differences between supervised and unsupervised learning" output:
supervised learning
unsupervised learning
If the question is NOT about comparing two distinct concepts, output exactly one line: NONE

User question: {query}

Your two phrases (one per line) or NONE:"""

EXTRACT_RELEVANT_PART_PROMPT = """The user message may mix a question about the course material (e.g. machine learning) with something off-topic. Extract ONLY the part that is about the course/material (machine learning, ML, textbook). Ignore any part about cooking, sports, geography, weather, etc.
If the whole message is about the material, return it unchanged. If only one part is about the material, return just that part. Output only the extracted text, no explanation.

User message: {query}

Extracted part (about course material only):"""


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings=None) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.embed_model = "text-embedding-3-small"
        self._embeddings = embeddings
        self.namespace = "ns2500"
        self.topic_prompt = TOPIC_PROMPT

    def set_prompt(self, prompt):
        self.topic_prompt = prompt

    def get_relevant_part(self, query):
        """For hybrid prompts, extract only the part about course material (ML). Returns stripped string or original if unchanged."""
        prompt = EXTRACT_RELEVANT_PART_PROMPT.format(query=query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception:
            pass
        return query

    def get_comparison_phrases(self, query):
        """If the query asks to compare two things, return (phrase1, phrase2) for dual retrieval; else None."""
        prompt = COMPARISON_EXTRACT_PROMPT.format(query=query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw or raw.upper() == "NONE":
                return None
            lines = [s.strip() for s in raw.split("\n") if s.strip()]
            if len(lines) >= 2:
                return (lines[0], lines[1])
        except Exception:
            pass
        return None

    def _embed(self, text):
        if callable(self._embeddings):
            return self._embeddings(text)
        if self._embeddings and hasattr(self._embeddings, "embed_query"):
            return self._embeddings.embed_query(text)
        return _embed_text(self.client, text, self.embed_model)

    def extract_action(self, response, query=None):
        return response and response.strip().upper().startswith("YES")

    def is_query_on_topic(self, query):
        prompt = self.topic_prompt.format(query=query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            return self.extract_action(resp.choices[0].message.content)
        except Exception:
            return False

    def query_vector_store(self, query, k=5):
        try:
            vector = self._embed(query)
            res = self.index.query(
                vector=vector,
                top_k=k,
                namespace=self.namespace,
                include_metadata=True,
            )
            docs = []
            for m in (res.matches or []):
                meta = (m.metadata or {})
                text = meta.get("text", meta.get("content", str(m)))
                docs.append(DocResult(text, meta))
            return docs
        except Exception:
            return []


# ----- Answering_Agent -----
ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the following context. Be concise.

If the user asks for "differences", "comparison", or "between A and B", you may combine information from different parts of the context (different snippets may discuss A and B separately). Do not say "the context does not contain" if the concepts are present somewhere in the context.
If the context truly does not mention the asked concepts, say so briefly.

Context:
{context}

Conversation so far:
{history}

User question: {query}

Answer:"""


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.prompt_template = ANSWER_PROMPT

    def generate_response(self, query, docs, conv_history, k=5):
        doc_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in (docs or [])[:k]]
        context = "\n\n".join(doc_texts) if doc_texts else "(No relevant documents)"
        history = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in (conv_history or [])[-4:]])
        prompt = self.prompt_template.format(context=context, history=history or "(none)", query=query)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return "Sorry, I could not generate a response."


# ----- Head_Agent -----
REFUSE_IMPOLITE = "Please ask in a polite and respectful way."
REFUSE_OBNOXIOUS = "I won't respond to that. Please ask in a respectful way."
REFUSE_IRRELEVANT = "I can only answer questions about the course material (e.g. machine learning). Your question seems out of scope."
REFUSE_NO_RELEVANT_DOCS = "I couldn't find relevant material to answer that. Try rephrasing or asking about machine learning topics."

# Small talk: respond politely without retrieval (LLM decides, no hardcoded list)
SMALL_TALK_RESPONSE = "Hello! I'm here to help with questions about the machine learning material. What would you like to know?"

SMALL_TALK_PROMPT = """Decide if the user's message is ONLY greeting, small talk, or chitchat—with no substantive question about course/material (e.g. machine learning).

Examples of small talk: "Hi!", "Good morning", "How are you?", "What's your favorite way to spend the weekend?", "Do you have any plans?", "Nice weather today."
NOT small talk: "What is overfitting?", "Explain logistic regression", "Hi, can you explain neural networks?"

Answer ONLY one word: "Yes" if it is only small talk/greeting, or "No" if it contains or is a real question about the material.

User message: {query}

Answer:"""


class Small_Talk_Agent:
    """Uses LLM to detect greeting/small talk so we don't refuse with 'out of scope'."""
    def __init__(self, client) -> None:
        self.client = client
        self.model = "gpt-4.1-nano"

    def is_small_talk(self, query: str) -> bool:
        if not (query or "").strip():
            return False
        prompt = SMALL_TALK_PROMPT.format(query=query.strip())
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            return text.startswith("YES")
        except Exception:
            return False


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name, namespace="ns2500") -> None:
        from pinecone import Pinecone
        self.client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(pinecone_index_name)
        self.namespace = namespace
        self.politeness_agent = None
        self.obnoxious_agent = None
        self.small_talk_agent = None
        self.context_rewriter = None
        self.query_agent = None
        self.relevant_agent = None
        self.answering_agent = None

    def setup_sub_agents(self):
        self.politeness_agent = Politeness_Agent(self.client)
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.small_talk_agent = Small_Talk_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.client, None)
        self.query_agent.namespace = self.namespace
        self.relevant_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def respond(self, user_message, conv_history=None):
        """Returns (response_text, agent_path, source_passages).
        source_passages is a list of strings (PDF chunks used for the answer), or None if no RAG was used (e.g. refused).
        """
        conv_history = conv_history or []
        agent_path = []
        if self.obnoxious_agent is None:
            self.setup_sub_agents()

        # 1) First gate: must be polite. Do not go to topic-check just because "machine learning" appears.
        if not self.politeness_agent.is_polite(user_message):
            agent_path.append("Politeness_Agent")
            return REFUSE_IMPOLITE, " -> ".join(agent_path), None

        if self.obnoxious_agent.check_query(user_message):
            agent_path.append("Obnoxious_Agent")
            return REFUSE_OBNOXIOUS, " -> ".join(agent_path), None

        if self.small_talk_agent.is_small_talk(user_message):
            agent_path.append("Small_Talk")
            return SMALL_TALK_RESPONSE, " -> ".join(agent_path), None

        # Fallback for vague follow-ups: use last user question so retrieval finds relevant docs
        _vague = user_message.strip().lower()
        _vague_phrases = ("tell me more", "more", "can you give an example", "give an example", "example?", "what about that", "explain further", "go on", "and?", "继续说", "再讲一下", "举个例子")
        if conv_history and any(_vague == p or _vague.startswith(p) or p in _vague for p in _vague_phrases):
            effective_query = conv_history[-1][0]  # last user question
        else:
            effective_query = self.context_rewriter.rephrase(conv_history, user_message)
        if not effective_query.strip():
            return "Could you rephrase that?", "Context_Rewriter", None

        # Hybrid: extract only the part about course material so we answer that and ignore the rest
        effective_query = self.query_agent.get_relevant_part(effective_query).strip() or effective_query
        if not effective_query:
            return "Could you rephrase that?", "Context_Rewriter", None

        # Topic check only after polite + not obnoxious; LLM-based, not keyword "machine learning"
        if not self.query_agent.is_query_on_topic(effective_query):
            agent_path.append("Query_Agent(topic_check)")
            return REFUSE_IRRELEVANT, " -> ".join(agent_path), None

        # For "differences between A and B" style questions, retrieve with both concepts so PDF content about each is found
        phrases = self.query_agent.get_comparison_phrases(effective_query)
        if phrases:
            docs1 = self.query_agent.query_vector_store(phrases[0], k=4)
            docs2 = self.query_agent.query_vector_store(phrases[1], k=4)
            seen = set()
            docs = []
            for d in docs1 + docs2:
                t = d.page_content
                if t not in seen:
                    seen.add(t)
                    docs.append(d)
                if len(docs) >= 8:
                    break
            docs = docs[:8]
        else:
            docs = self.query_agent.query_vector_store(effective_query, k=8)
        agent_path.append("Query_Agent(retrieve)")
        if not docs:
            return REFUSE_NO_RELEVANT_DOCS, " -> ".join(agent_path), None

        rel = self.relevant_agent.get_relevance({"query": effective_query, "docs": docs})
        if rel != "Yes":
            agent_path.append("Relevant_Documents_Agent")
            return REFUSE_NO_RELEVANT_DOCS, " -> ".join(agent_path), None

        agent_path.append("Relevant_Documents_Agent")
        agent_path.append("Answering_Agent")
        answer = self.answering_agent.generate_response(effective_query, docs, conv_history, k=8)
        source_passages = [d.page_content for d in docs]
        return answer, " -> ".join(agent_path), source_passages

    def main_loop(self):
        self.setup_sub_agents()
        history = []
        print("Multi-Agent Chatbot (ML topic). Type 'quit' to exit.")
        while True:
            u = input("You: ").strip()
            if u.lower() in ("quit", "exit", "q"):
                break
            if not u:
                continue
            resp, path = self.respond(u, history)
            print(f"Bot: {resp}")
            print(f"[Path: {path}]")
            history.append((u, resp))
