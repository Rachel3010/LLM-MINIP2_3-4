"""
Part 3 Streamlit demo: Multi-Agent Chatbot (ML topic).
Run: streamlit run streamlit_app.py
Set OPENAI_API_KEY and PINECONE_API_KEY in environment or in Streamlit secrets.
"""
import os
import streamlit as st
from part3_agents import Head_Agent

# Page config
st.set_page_config(page_title="Multi-Agent Chatbot (Part 3)", layout="centered")
st.title("Multi-Agent Chatbot (ML Topic)")
st.caption("Obnoxious check → Topic check → Retrieve → Relevance check → Answer")

# API keys: env or Streamlit secrets
openai_key = os.environ.get("OPENAI_API_KEY") or (getattr(st.secrets, "OPENAI_API_KEY", None) or "")
pinecone_key = os.environ.get("PINECONE_API_KEY") or (getattr(st.secrets, "PINECONE_API_KEY", None) or "")
index_name = os.environ.get("PINECONE_INDEX_NAME", "machine-learning-textbook")
namespace = os.environ.get("PINECONE_NAMESPACE", "ns2500")

if not openai_key or not pinecone_key:
    st.warning("Set OPENAI_API_KEY and PINECONE_API_KEY (env or Streamlit secrets) to run the chatbot.")
    st.stop()

# Init Head_Agent once per session
@st.cache_resource
def get_head_agent():
    return Head_Agent(openai_key, pinecone_key, index_name, namespace=namespace)

head = get_head_agent()

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_paths" not in st.session_state:
    st.session_state.agent_paths = []
if "source_passages" not in st.session_state:
    st.session_state.source_passages = []


def get_conversation(messages=None):
    """Build conversation history: one string with all messages, each line prefixed by role (User / Assistant)."""
    if messages is None:
        messages = st.session_state.messages
    lines = []
    for role, content in messages:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines) if lines else "(No messages yet)"


# Sidebar: "Chats" list style — pill-shaped items, last item selected (light blue)
SIDEBAR_PREVIEW_LEN = 80

def _escape(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

with st.sidebar:
    st.markdown(
        """
        <style>
        .chats-header { font-weight: 600; color: #1e293b; margin-bottom: 0.75rem; font-size: 1rem; }
        .chat-item { padding: 0.5rem 0.75rem; margin-bottom: 0.35rem; border-radius: 9999px; font-size: 0.875rem;
                     color: #334155; background: transparent; }
        .chat-item.selected { background: #e0f2fe; color: #0369a1; }
        .chat-item .role { font-weight: 600; margin-right: 0.25rem; }
        .chat-item .preview { color: inherit; opacity: 0.95; }
        .chat-item { display: block; text-decoration: none; }
        .chat-item:hover { opacity: 0.9; }
        </style>
        <div class="chats-header">Chats</div>
        """,
        unsafe_allow_html=True,
    )
    if not st.session_state.messages:
        st.caption("No messages yet. Start chatting in the main area.")
    else:
        n = len(st.session_state.messages)
        for i, (role, content) in enumerate(st.session_state.messages):
            role_label = "User" if role == "user" else "Assistant"
            preview = (content[:SIDEBAR_PREVIEW_LEN].strip() + "…") if len(content) > SIDEBAR_PREVIEW_LEN else content
            selected = " selected" if i == n - 1 else ""
            # Clickable link: jumps to the message in the main area (anchor #msg-{i})
            html = f'<a href="#msg-{i}" class="chat-item{selected}"><span class="role">{_escape(role_label)}</span><span class="preview">{_escape(preview)}</span></a>'
            st.markdown(html, unsafe_allow_html=True)

# Main area: each message has an anchor so sidebar links can jump here
for i, (role, content) in enumerate(st.session_state.messages):
    st.markdown(f'<span id="msg-{i}"></span>', unsafe_allow_html=True)
    with st.chat_message(role):
        role_label = "**User**" if role == "user" else "**Assistant**"
        st.markdown(role_label)
        st.markdown(content)
        if role == "assistant":
            path_idx = (i - 1) // 2
            if path_idx < len(st.session_state.agent_paths):
                st.caption(f"Path: {st.session_state.agent_paths[path_idx]}")
            if path_idx < len(st.session_state.source_passages) and st.session_state.source_passages[path_idx]:
                with st.expander("Source passages (from PDF)"):
                    for j, passage in enumerate(st.session_state.source_passages[path_idx], 1):
                        st.text_area(f"Passage {j}", value=passage[:2000] + ("..." if len(passage) > 2000 else ""), height=120, disabled=True, key=f"src_{i}_{j}")

# Input
if prompt := st.chat_input("Ask about machine learning..."):
    st.session_state.messages.append(("user", prompt))
    conv_history = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if i + 1 < len(st.session_state.messages):
            u, a = st.session_state.messages[i], st.session_state.messages[i + 1]
            if u[0] == "user" and a[0] == "assistant":
                conv_history.append((u[1], a[1]))
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, path, sources = head.respond(prompt, conv_history)
        st.markdown("**Assistant**")
        st.markdown(response)
        st.caption(f"Path: {path}")
        if sources:
            with st.expander("Source passages (from PDF)"):
                for j, passage in enumerate(sources, 1):
                    st.text_area(f"Passage {j}", value=passage[:2000] + ("..." if len(passage) > 2000 else ""), height=120, disabled=True, key=f"src_new_{j}")
    st.session_state.messages.append(("assistant", response))
    st.session_state.agent_paths.append(path)
    st.session_state.source_passages.append(sources if sources else [])
    st.rerun()
