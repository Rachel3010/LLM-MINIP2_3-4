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

# Show history
for i, (role, content) in enumerate(st.session_state.messages):
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            path_idx = (i - 1) // 2
            if path_idx < len(st.session_state.agent_paths):
                st.caption(f"Path: {st.session_state.agent_paths[path_idx]}")

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
            response, path = head.respond(prompt, conv_history)
        st.markdown(response)
        st.caption(f"Path: {path}")
    st.session_state.messages.append(("assistant", response))
    st.session_state.agent_paths.append(path)
    st.rerun()
