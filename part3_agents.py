"""
Part 3: Multi-Agent Chatbot — all agents and Head_Agent controller.
Use with: from part3_agents import Head_Agent, Obnoxious_Agent, ...
"""
from openai import OpenAI

# ----- Obnoxious_Agent (no LangChain): single gate for polite + not obnoxious -----
DEFAULT_OBNOXIOUS_PROMPT = """You are a content moderation agent. Decide if the user's message is polite and respectful, or should be refused.

Refuse (answer "Yes") if the message is: impolite, obnoxious, insulting, rude, profanity, hate speech, harassment, threats, or offensive in any way. A question that includes insults (e.g. "Explain ML, idiot") should be refused.
Allow (answer "No") if the message is polite and respectful, even if it is off-topic or just a greeting.

Output ONLY one word: "Yes" if we should refuse the message, or "No" if it is acceptable.

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
DEFAULT_RELEVANCE_PROMPT = """You are a relevance judge. Given a user query and retrieved document snippets from a textbook, decide if these documents can help answer the query.

User query: {query}

Retrieved documents:
{docs}

Answer ONLY "Yes" or "No".
- Say "Yes" if the documents discuss the concepts, algorithms, or topics in the query (even if the answer is spread across snippets, or only partly addressed, or they cover "A" and "B" separately when the query asks for "differences between A and B"). Partial relevance is enough.
- Say "No" only if the documents are clearly off-topic or have nothing to do with the query (e.g. completely unrelated subject).

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
TOPIC_PROMPT = """Is the user asking a question about the course material (e.g. machine learning, ML, statistics, the textbook)?

Answer "Yes" if they are asking for an explanation, definition, comparison, or information about the subject: e.g. "What is machine learning?", "What is ML?", "Explain overfitting", "What is logistic regression?", "Difference between supervised and unsupervised learning", "What does chapter 3 say about...". These are all on-topic.

Answer "No" only if the query is clearly unrelated to the course: e.g. weather, sports, cooking, or pure greeting/small talk with no question about the material.

Query: {query}

Answer (Yes or No):"""


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

EXTRACT_RELEVANT_PART_PROMPT = """You extract only the machine learning (or course material) part of the user's message. The user may mix an ML question with off-topic questions (weather, buildings, food, sports, etc.). Your output must contain ONLY the part that asks about the course/material (ML, textbook). Ignore and drop any clause about non-course topics, no matter whether the ML part appears at the beginning, middle, or end of the message.

Examples:
- "Can you clarify Bias-Variance tradeoff and what's the weather today?" → Can you clarify Bias-Variance tradeoff?
- "What is the tallest building in the world and how do Decision Trees operate?" → How do Decision Trees operate?
- "Describe how Decision Trees operate, and what is the tallest building?" → Describe how Decision Trees operate.

If the whole message is about the material, return it unchanged. Output only the extracted question (the ML part only), nothing else.

User message: {query}

Extracted part (machine learning / course only):"""

EXTRACT_SEARCH_PHRASE_PROMPT = """From this question about course material, extract a short phrase (2–8 words) that would work as a search query for a textbook index. Use the main concept or algorithm name (e.g. "k-nearest neighbors", "logistic regression", "bias variance tradeoff"). Output ONLY the phrase, nothing else.

Question: {query}

Search phrase:"""


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

    def get_search_phrase(self, query):
        """Extract a short search phrase from the query for retry retrieval. Returns stripped string or None."""
        if not (query or "").strip():
            return None
        prompt = EXTRACT_SEARCH_PHRASE_PROMPT.format(query=query.strip())
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40,
                temperature=0,
            )
            out = (resp.choices[0].message.content or "").strip()
            return out if out and len(out) < 100 else None
        except Exception:
            return None

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
        except Exception as e:
            import sys
            print(f"[Pinecone retrieval error] {e}", file=sys.stderr)
            return []


# ----- Answering_Agent -----
ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the following context. Be concise. Answer only the question about the course material; do not mention or answer any off-topic part (e.g. weather, sports).

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

    def generate_response_stream(self, query, docs, conv_history, k=5):
        """Yields text chunks for streaming display. Only yields non-empty strings."""
        doc_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in (docs or [])[:k]]
        context = "\n\n".join(doc_texts) if doc_texts else "(No relevant documents)"
        history = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in (conv_history or [])[-4:]])
        prompt = self.prompt_template.format(context=context, history=history or "(none)", query=query)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and getattr(chunk.choices[0].delta, "content", None):
                    part = chunk.choices[0].delta.content
                    if part:
                        yield part
        except Exception:
            yield "Sorry, I could not generate a response."


# ----- Head_Agent -----
REFUSE_OBNOXIOUS = "I won't respond to that. Please ask in a polite and respectful way."
REFUSE_IRRELEVANT = "I can only answer questions about the course material (e.g. machine learning). Your question seems out of scope."
REFUSE_NO_RELEVANT_DOCS = "I couldn't find relevant material to answer that. Try rephrasing or asking about machine learning topics."

# Small talk: respond politely without retrieval (LLM decides, no hardcoded list)
SMALL_TALK_RESPONSE = "Hello! I'm here to help with questions about the machine learning material. What would you like to know?"

SMALL_TALK_PROMPT = """You decide whether the user's message is ONLY greeting/small talk with NO substantive question about the course (machine learning, textbook).

Answer "No" (NOT small talk) if the message contains ANY question about the course material—e.g. "clarify the concept of X", "explain X", "what is X", "define X", or any ML concept (bias-variance, overfitting, logistic regression, etc.). Answer "No" even if the message also mentions weather, food, or other off-topic things. We will answer only the ML part later.

Answer "Yes" (small talk) ONLY when the message is purely greeting or chitchat with no course question: e.g. "Hi", "How are you?", "What's up?", "Good morning", "Thanks", "Nice to meet you".

Example: "Can you clarify the concept of Bias-Variance tradeoff and what's the weather like today?" → No (it asks to clarify an ML concept).

User message: {query}

Answer with exactly one word: Yes or No."""


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
        self.obnoxious_agent = None
        self.small_talk_agent = None
        self.context_rewriter = None
        self.query_agent = None
        self.relevant_agent = None
        self.answering_agent = None

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.small_talk_agent = Small_Talk_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.client, None)
        self.query_agent.namespace = self.namespace
        self.relevant_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def respond(self, user_message, conv_history=None, stream=False):
        """Returns (response_text_or_stream, agent_path, source_passages).
        If stream=True and we generate an answer, first element is a generator; else the full string.
        """
        conv_history = conv_history or []
        agent_path = []
        if self.obnoxious_agent is None:
            self.setup_sub_agents()

        # First gate: polite and not obnoxious (single Obnoxious_Agent does both)
        if self.obnoxious_agent.check_query(user_message):
            agent_path.append("Obnoxious_Agent")
            return REFUSE_OBNOXIOUS, " -> ".join(agent_path), None

        # Fast path: common greetings/small talk (normalize: strip, lower, drop trailing ?!., so "How are you?" matches)
        _raw = (user_message or "").strip().lower()
        _msg = _raw.rstrip("?!.,; ")
        _exact = {"hi", "hello", "hey", "howdy", "greetings", "how are you", "how are you today", "how are you doing", "how's it going", "how is it going", "how's your day", "how's you day", "what's up", "how do you do", "good morning", "good afternoon", "good evening", "nice to meet you", "good to see you", "hi there", "hello there", "hey there", "thanks", "thank you"}
        _starts = ("how are you", "what's up", "how's it going", "how is it going", "how's your day", "how's you day", "good morning", "good afternoon", "good evening", "nice to meet you", "good to see you")
        if _msg in _exact or any(_msg.startswith(p) for p in _starts):
            agent_path.append("Small_Talk")
            return SMALL_TALK_RESPONSE, " -> ".join(agent_path), None

        # Do not treat as small talk if the message clearly contains an ML/course question (hybrid)
        _ml_keywords = ("clarify", "explain the concept", "bias", "variance", "tradeoff", "overfitting", "logistic regression", "neural network", "what is ", "define ")
        if not any(k in _raw for k in _ml_keywords) and self.small_talk_agent.is_small_talk(user_message):
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

        # Fast path: obviously on-topic (e.g. "What is machine learning?") so topic_check LLM cannot misclassify
        _q = effective_query.strip().lower().rstrip("?!.,; ")
        _on_topic_starts = ("what is machine learning", "what's machine learning", "what is ml", "what's ml", "explain machine learning", "explain ml", "define machine learning", "define ml", "what is overfitting", "what is logistic regression", "what is neural network", "what is supervised learning", "what is unsupervised learning")
        if any(_q == p or _q.startswith(p + " ") or _q.startswith(p + "?") for p in _on_topic_starts) or _q.startswith("what is ") and ("machine learning" in _q or " ml " in _q or _q == "what is ml"):
            pass  # skip topic check, clearly on-topic
        elif not self.query_agent.is_query_on_topic(effective_query):
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
        # Fallback: if retrieval returned nothing, try a shorter key phrase (e.g. "machine learning") for broad questions
        if not docs and effective_query.strip():
            _q = effective_query.strip().lower()
            if "machine learning" in _q or _q.startswith("what is ml") or _q.startswith("what's ml"):
                docs = self.query_agent.query_vector_store("machine learning", k=8)
            elif "ml " in _q or _q.startswith("ml "):
                docs = self.query_agent.query_vector_store("machine learning", k=8)
        agent_path.append("Query_Agent(retrieve)")
        if not docs:
            return REFUSE_NO_RELEVANT_DOCS, " -> ".join(agent_path), None

        rel = self.relevant_agent.get_relevance({"query": effective_query, "docs": docs})
        # Fallback: when relevance says No, ask LLM for a short search phrase and retry retrieval (works for any topic)
        if rel != "Yes" and effective_query.strip():
            key_phrase = self.query_agent.get_search_phrase(effective_query)
            if key_phrase:
                docs2 = self.query_agent.query_vector_store(key_phrase, k=8)
                if docs2:
                    rel2 = self.relevant_agent.get_relevance({"query": effective_query, "docs": docs2})
                    if rel2 == "Yes":
                        docs, rel = docs2, "Yes"
        if rel != "Yes":
            agent_path.append("Relevant_Documents_Agent")
            return REFUSE_NO_RELEVANT_DOCS, " -> ".join(agent_path), None

        agent_path.append("Relevant_Documents_Agent")
        agent_path.append("Answering_Agent")
        source_passages = [d.page_content for d in docs]
        path_str = " -> ".join(agent_path)
        if stream:
            return self.answering_agent.generate_response_stream(effective_query, docs, conv_history, k=8), path_str, source_passages
        answer = self.answering_agent.generate_response(effective_query, docs, conv_history, k=8)
        return answer, path_str, source_passages

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
