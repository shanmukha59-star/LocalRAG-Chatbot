# app.py
import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot")

API_URL = "http://127.0.0.1:8000/ask"

if "history" not in st.session_state:
    st.session_state.history = []    

# Display past messages
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])

# Chat input box
query = st.chat_input("Ask a question about your documents...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "history": st.session_state.history},
                    timeout=120,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
                st.write(answer)
            except requests.RequestException:
                answer = "The RAG API is unavailable. Start main.py and try again."
                st.error(answer)

    st.session_state.history.append({"user": query, "assistant": answer})