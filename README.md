
# 📄 Local AI Research Assistant (RAG Pipeline)

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) application. This tool allows you to upload complex academic PDFs (such as two-column IEEE papers), semantically parse their layout, and query their contents using an open-source Large Language Model—all running entirely on your local hardware without relying on cloud APIs.

## ✨ Features

* **100% Local & Private:** No data is sent to OpenAI, Google, or any external API. Your documents stay on your machine.
* **Layout-Aware Parsing:** Utilizes the `unstructured` library with a `fast` strategy to intelligently parse complex PDF layouts, capturing titles, methodologies, and section headers without scrambling multi-column text.
* **Semantic Chunking:** Documents are split by their logical sections (headers/titles) rather than arbitrary character counts, with a fallback character limiter to prevent context window overflow.
* **Optimized Local Embeddings:** Powered by `BAAI/bge-small-en-v1.5` and `FAISS` for lightning-fast, highly accurate semantic search.
* **Modern LangChain Architecture:** Built using the latest LangChain ecosystem routing and retrieval chains.

---

## 📂 Project Structure

* `main.py` - The Streamlit frontend interface. Handles file uploads, session state, and user chat interactions.
* `document_embedding.py` - Manages the `UnstructuredPDFLoader`. Extracts layout elements, applies semantic chunking, and enforces the fallback length limits.
* `vector_storage.py` - Initializes the HuggingFace embeddings (`bge-small`) and creates the local FAISS vector database.
* `rag_backend.py` - Connects to the local Ollama instance, sets the system prompt rules, and chains the retriever with the LLM.

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **Ollama:** The backend engine to run the Large Language Model locally.

---

## 📥 Installation & Setup

### 1. Install Dependencies
Ensure you have created a virtual environment, then install the required packages using pip by running the following command:

```bash
pip install -r "requirements.txt"
```

### 2. Install and Configure Ollama
This project specifically uses Llama 3.1 (8 Billion parameters) for its excellent reasoning and concise question-answering capabilities.

1. Download and install Ollama from ollama.com.
2. Open your terminal or command prompt.
3. Pull the specific Llama 3.1 model by running the following command:

```bash 
ollama pull llama3.1:8b
 ````

4. Note: The download is approximately 4.7GB. Once it finishes, Ollama runs automatically in the background as a local host service (defaulting to http://localhost:11434).

---

## 🚀 Running the Application

1. Ensure the Ollama application is running in the background.
2. Open your terminal in the project directory.
3. Launch the Streamlit interface:

```bash
streamlit run main.py
```

4. Open the provided Local URL in your browser (usually http://localhost:8501).
5. Upload your PDF via the sidebar, click Process Document, and wait for the success message.
6. Start asking questions about your document!

---

## 📦 Dependencies
* Python 3.10+
* Ollama
* Streamlit >= 1.0
* LangChain Ecosystem
* FAISS (faiss* cpu)
* Unstructured / PyPDF
* Sentence Transformers

---

## ⚖️ License

MIT License

Author: Ramz P

Year: 2026

---