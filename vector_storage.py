# Import the HuggingFace embeddings wrapper for local vectorization
from langchain_huggingface import HuggingFaceEmbeddings
# Import the FAISS vector database wrapper for efficient similarity search
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    # Initialize the embedding model (BAAI/bge-small-en-v1.5) optimized for semantic retrieval
    # 'device': 'cuda' maps the embedding mathematical workload directly to the GPU for faster processing
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cuda'}
    )
    
    # Convert the text chunks into mathematical vectors and store them in a local FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Return the populated vector database instance
    return vector_store
