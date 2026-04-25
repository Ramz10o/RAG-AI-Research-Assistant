import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import tempfile
from document_embedding import process_document
from vector_storage import create_vector_store
from rag_backend import setup_rag_chain

print("Initializing Streamlit app....")

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("📄 AI Research Assistant")
st.markdown("""
Upload a PDF document, and ask questions about its content. The AI will retrieve relevant sections and provide answers based on the document's information.
""")

# Initialize session state for the RAG chain
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to analyze", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing and Embedding..."):
            # Save uploaded file temporarily to pass to PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Execute the pipeline
            chunks = process_document(tmp_path)
            vector_store = create_vector_store(chunks)
            st.session_state.rag_chain = setup_rag_chain(vector_store)
            
            os.remove(tmp_path) # Cleanup
            st.success("Document successfully processed!")

# Main chat interface
user_query = st.text_input("Ask a question about the document:")

if user_query:
    if st.session_state.rag_chain is None:
        st.warning("Please upload and process a pdf document first.")
    else:
        with st.spinner("Searching for answers..."):
            response = st.session_state.rag_chain.invoke({"input": user_query})
            
            st.markdown("### Answer:")
            st.write(response["answer"])
            
            with st.expander("View Source Context"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1}:** {doc.page_content}")