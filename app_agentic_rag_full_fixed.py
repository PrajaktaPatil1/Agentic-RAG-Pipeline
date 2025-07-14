
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Import the compiled agentic pipeline
from langchain_agentic_pipeline import app as agentic_app

st.set_page_config(page_title="Agentic RAG Full", layout="wide")
st.title(" Agentic RAG with Multi-Tool Reasoning")

st.markdown("This app classifies your question, runs the right reasoning tool (RAG / LLM / Web), validates the result, and gives a concise final answer.")

query = st.text_input("Ask me anything...", placeholder="e.g. What is the net worth of Elon Musk?")

if query:
    st.info(" Processing through Agentic RAG Workflow...")
    with st.spinner("Invoking tools and reasoning..."):
        result = agentic_app.invoke({"messages": [query]})

    if result:
        # Attempt to extract a more meaningful output
        final_message = result.get("messages", [])[-1] if isinstance(result.get("messages"), list) else result.get("messages")
        full_answer = result.get("answer") or final_message or str(result)

        st.success("Final Answer:")
        st.markdown(full_answer if isinstance(full_answer, str) else str(full_answer))

        # Optional debug section
        with st.expander("Full Result Object (Debug)", expanded=False):
            st.json(result)
    else:
        st.warning(" No answer generated. Try again with a different query.")
