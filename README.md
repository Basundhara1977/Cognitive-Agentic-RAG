# 🧠 Cognitive Agentic RAG System

A Retrieval-Augmented Generation (RAG) based AI system that integrates document retrieval with large language model reasoning to generate context-aware, grounded responses.

This project demonstrates an agentic workflow where retrieval and reasoning are combined to improve answer quality and factual consistency.

---

## 📌 Overview

The system loads a document (PDF), processes it into semantic chunks, generates embeddings, stores them in a vector store, and retrieves relevant context during query time. The retrieved context is then passed to a language model to generate accurate and grounded responses.

The project includes both a notebook implementation and a runnable Python application.

---

## 🚀 Features

* Document ingestion from PDF
* Text chunking and embedding generation
* Semantic similarity-based retrieval
* LLM-powered response generation
* Modular RAG pipeline architecture
* Notebook and application-based implementation

---

## 🛠 Tech Stack

* Python
* LangChain
* Groq LLM
* Vector Embeddings
* Jupyter Notebook

---

## 📂 Project Structure

```
Cognitive-Agentic-RAG/
│
├── agentic_rag.ipynb
├── app.py
├── requirements.txt
├── 2026 Agentic RAG.pdf
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Basundhara1977/Cognitive-Agentic-RAG.git
cd Cognitive-Agentic-RAG
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in the project root directory:

```
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

Do not commit the `.env` file to version control.

---

## ▶ Running the Project

To run the notebook:

```
jupyter notebook
```

To run the application:

```
python app.py
```

---

## 🧠 How It Works

1. Load and preprocess the PDF document
2. Split text into manageable chunks
3. Generate embeddings for each chunk
4. Store embeddings in a vector database
5. Retrieve relevant chunks based on user query
6. Pass retrieved context to the LLM for response generation

This ensures responses are grounded in the provided document rather than relying solely on model memory.

---

## 📈 Future Enhancements

* Multi-document support
* Conversational memory integration
* Web interface improvements
* Cloud deployment

---

## 👩‍💻 Author

Basundhara Maity
