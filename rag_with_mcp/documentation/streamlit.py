import streamlit as st
import asyncio
from main import get_docs  

st.title("🔍 AI Doc Assistant")

query = st.text_input("Enter your question:")
library = st.selectbox("Choose a library:", ["langchain", "openai", "llama_index"])

if st.button("Search"):
    if query and library:
        with st.spinner("Searching..."):
            result = asyncio.run(get_docs(query, library))
            st.text_area("📄 Result:", result, height=400)
