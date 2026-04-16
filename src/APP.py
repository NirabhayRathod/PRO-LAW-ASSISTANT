import os
import streamlit as st
import requests
from dotenv import load_dotenv
import time

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (UNCHANGED)
st.markdown("""<style>
/* (your full CSS unchanged) */
</style>""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚖️ Legal AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Intelligent Legal Research Partner • Powered by FAISS & Groq Llama 3.1</div>', unsafe_allow_html=True)

# Load env
load_dotenv()

# 🔥 CHANGE: FastAPI URL (UPDATE THIS LATER FOR EC2)
FASTAPI_URL = "http://localhost:8000/query"

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_connected" not in st.session_state:
    st.session_state.db_connected = False

# Connect button (NOW JUST CHECKS API)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if not st.session_state.db_connected:
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h3 style='color: #2C3E50; margin-bottom: 1rem;'>🚀 Ready to Explore Legal Knowledge</h3>
            <p style='color: #5D6D7E; font-size: 1.1rem;'>Connect to backend API</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔗 Connect to Legal Database", use_container_width=True):
            try:
                # simple health check
                res = requests.get("http://localhost:8000/")
                if res.status_code == 200:
                    st.session_state.db_connected = True
                    st.success("✅ Connected to backend!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Backend not responding")
            except:
                st.error("❌ FastAPI server not running")

# Chat UI
if st.session_state.db_connected:
    st.markdown("---")

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">⚖️ {message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt_text := st.chat_input("Ask your legal question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        with chat_container:
            st.markdown(f'<div class="user-message">👤 {prompt_text}</div>', unsafe_allow_html=True)

            with st.spinner("⚖️ Researching legal documents..."):
                try:
                    # 🔥 CALL FASTAPI
                    response = requests.post(
                        FASTAPI_URL,
                        json={"text": prompt_text}
                    )

                    result = response.json()

                    answer = "\n\n".join(result["results"])

                except Exception as e:
                    answer = "❌ Error connecting to backend. Please try again."

                st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()

    # Sidebar
    if st.session_state.messages:
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-content">
                <h3>📚 Legal Resources</h3>
                <p>Backend powered results</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

else:
    # Feature UI (UNCHANGED)
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <h3 style='color: #2C3E50; margin-bottom: 2rem;'>🌟 Why Choose Legal AI Assistant?</h3>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7F8C8D; margin-top: 3rem; padding: 1rem; font-size: 0.9rem;'>"
    "Built with ❤️ using Streamlit, LangChain, and Groq"
    "</div>",
    unsafe_allow_html=True
)