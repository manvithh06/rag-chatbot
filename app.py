"""
app.py — Streamlit chatbot interface.
Run with: streamlit run app.py
"""

import streamlit as st
import os
from rag_engine import RAGEngine



# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="ML Knowledge Base Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────
st.title("🤖 ML Knowledge Base Chatbot")
st.markdown(
    "Ask questions about **Machine Learning, Deep Learning, NLP, "
    "Neural Networks**, and related topics. The bot answers strictly "
    "from its document corpus — it will tell you when it doesn't know."
)
st.divider()

# ── API Key Input ───────────────────────────────────────────
# WHY in sidebar: keeps main area clean, standard UX pattern
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Try environment variable first (for deployment)
    # Fall back to user input (for local testing)
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get a free key at console.groq.com"
        )
    else:
        st.success("✅ API key loaded from environment")
    
    st.divider()
    st.header("📚 Knowledge Base")
    st.markdown("""
    **Corpus:** 15 Wikipedia articles on ML topics
    
    **Topics covered:**
    - Machine Learning fundamentals
    - Deep Learning & Neural Networks
    - NLP & Transformers
    - Training techniques (gradient descent, overfitting)
    - Model architectures (CNN, RNN, GAN)
    - Transfer Learning
    """)
    
    st.divider()
    st.markdown("**Try these questions:**")
    
    sample_questions = [
        "What is gradient descent?",
        "How do transformers work?",
        "What is overfitting and how to prevent it?",
        "Explain transfer learning",
        "What is the difference between CNN and RNN?"
    ]
    
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state['sample_q'] = q

# ── Initialise RAG Engine ───────────────────────────────────
@st.cache_resource  # cache so it loads only once
def load_engine(key):
    """Load RAG engine — cached across reruns."""
    return RAGEngine(api_key=key)

# ── Chat history ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources used", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**{i}. [{src['title']}]({src['url']})**  \n"
                        f"Relevance: `{src['relevance']}`  \n"
                        f"*{src['chunk']}*"
                    )

# ── Handle sample question buttons ─────────────────────────
if 'sample_q' in st.session_state:
    query = st.session_state.pop('sample_q')
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        engine = load_engine(api_key)
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = engine.generate(query)
            st.markdown(result['answer'])
            with st.expander("📄 Sources used", expanded=False):
                for i, src in enumerate(result['sources'], 1):
                    st.markdown(
                        f"**{i}. [{src['title']}]({src['url']})**  \n"
                        f"Relevance: `{src['relevance']}`  \n"
                        f"*{src['chunk']}*"
                    )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "sources": result['sources']
        })
    st.rerun()

# ── Chat Input ──────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about machine learning..."):
    
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate RAG response
    engine = load_engine(api_key)
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            result = engine.generate(prompt)
        
        st.markdown(result['answer'])
        
        # Show sources in expandable section
        with st.expander(
            f"📄 {result['chunks_retrieved']} sources retrieved", 
            expanded=False
        ):
            for i, src in enumerate(result['sources'], 1):
                relevance_color = (
                    "🟢" if src['relevance'] > 0.7 
                    else "🟡" if src['relevance'] > 0.4 
                    else "🔴"
                )
                st.markdown(
                    f"{relevance_color} **{i}. [{src['title']}]({src['url']})**  \n"
                    f"Relevance score: `{src['relevance']}`  \n"
                    f"*{src['chunk']}*"
                )
    
    # Add to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "sources": result['sources']
    })