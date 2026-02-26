import os
import streamlit as st
from dotenv import load_dotenv

# --- 2026 MODERN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- NEW 2026 CLASSIC PATHS (Must have langchain-classic installed) ---
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Setup Environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# 2. UI Configuration
st.set_page_config(page_title="2026 Agentic RAG", page_icon="🤖")
st.title("🤖 Live AI Agentic RAG")

@st.cache_resource
def setup_rag():
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = setup_rag()

# 3. Robust Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Chat Logic
user_input = st.text_input("Ask about LLM Agents:")

if st.button("Run Agent"):
    if user_input:
        with st.spinner("Llama-3 is searching..."):
            try:
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

                response = rag_chain.invoke({"input": user_input})
                st.subheader("Answer")
                st.success(response["answer"])
            except Exception as e:
                st.error(f"Execution Error: {e}")
