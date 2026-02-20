"""
Part 3: Multi-Agent Chatbot — all agents and Head_Agent controller.
Use with: from part3_agents import Head_Agent, Obnoxious_Agent, ...
"""
from openai import OpenAI

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
DEFAULT_RELEVANCE_PROMPT = """You are a relevance judge. Given a user query and a set of retrieved document snippets, determine whether these documents are relevant to answering the user's query.

User query: {query}

Retrieved documents:
{docs}

Answer with ONLY one word: "Yes" if the documents are relevant to the query, or "No" if they are not relevant (e.g., off-topic or useless for answering).

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
REPHRASE_PROMPT = """Given the conversation history and the user's latest (possibly vague) message, rephrase the latest message into a standalone, clear question. If already clear, return it unchanged. Output ONLY the rephrased question.

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
TOPIC_PROMPT = """Does the following user query ask about machine learning, statistics, or the content of a machine learning textbook? Answer only "Yes" or "No".

Query: {query}

Answer:"""


class DocResult:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


def _embed_text(client, text, model="text-embedding-3-small"):
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


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
ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context. If the context does not contain enough information, say so. Be concise.

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
REFUSE_OBNOXIOUS = "I won't respond to that. Please ask in a respectful way."
REFUSE_IRRELEVANT = "I can only answer questions about the course material (e.g. machine learning). Your question seems out of scope."
REFUSE_NO_RELEVANT_DOCS = "I couldn't find relevant material to answer that. Try rephrasing or asking about machine learning topics."


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name, namespace="ns2500") -> None:
        from pinecone import Pinecone
        self.client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(pinecone_index_name)
        self.namespace = namespace
        self.obnoxious_agent = None
        self.context_rewriter = None
        self.query_agent = None
        self.relevant_agent = None
        self.answering_agent = None

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.client, None)
        self.query_agent.namespace = self.namespace
        self.relevant_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def respond(self, user_message, conv_history=None):
        conv_history = conv_history or []
        agent_path = []
        if self.obnoxious_agent is None:
            self.setup_sub_agents()

        if self.obnoxious_agent.check_query(user_message):
            agent_path.append("Obnoxious_Agent")
            return REFUSE_OBNOXIOUS, " -> ".join(agent_path)

        effective_query = self.context_rewriter.rephrase(conv_history, user_message)
        if not effective_query.strip():
            return "Could you rephrase that?", "Context_Rewriter"

        if not self.query_agent.is_query_on_topic(effective_query):
            agent_path.append("Query_Agent(topic_check)")
            return REFUSE_IRRELEVANT, " -> ".join(agent_path)

        docs = self.query_agent.query_vector_store(effective_query, k=5)
        agent_path.append("Query_Agent(retrieve)")
        if not docs:
            return REFUSE_NO_RELEVANT_DOCS, " -> ".join(agent_path)

        rel = self.relevant_agent.get_relevance({"query": effective_query, "docs": docs})
        if rel != "Yes":
            agent_path.append("Relevant_Documents_Agent")
            return REFUSE_NO_RELEVANT_DOCS, " -> ".join(agent_path)

        agent_path.append("Relevant_Documents_Agent")
        agent_path.append("Answering_Agent")
        answer = self.answering_agent.generate_response(effective_query, docs, conv_history, k=5)
        return answer, " -> ".join(agent_path)

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
