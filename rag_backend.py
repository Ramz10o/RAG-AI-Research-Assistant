# Import the Ollama wrapper to communicate with locally hosted LLMs
from langchain_ollama import ChatOllama
# Import the retrieval chain module (using the classic path)
from langchain_classic.chains.retrieval import create_retrieval_chain
# Import the chain module that stuffs all retrieved documents into the prompt context
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# Import the template class to structure the final message sent to the LLM
from langchain_core.prompts import ChatPromptTemplate

def setup_rag_chain(vector_store):
    # Initialize the local Llama 3.1 8B model via the Ollama application
    # temperature=0 ensures the model gives deterministic, factual answers without creative hallucination
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    
    # Define the system prompt dictating the AI's persona, boundaries, and formatting instructions
    # {context} is a placeholder variable where LangChain will inject the retrieved FAISS chunks
    system_prompt = (
        "You are a research assistant for question-answering tasks as well as a general purpose language model. "
        "Use the data from the following pieces of retrieved context to answer the question if it is based on the context. "
        "If you don't know the answer, just reply with 'I don't know the answer to the question.'. "
        "If the user asks a general conversational question, answer normally without relying on the context. "
        "Write in an engaging, professional tone, using structured paragraphs or bullet points where helpful.\n\n"
        "{context}"
    )
    
    # Structure the conversation flow: first the system rules, then the user's specific query ({input})
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the specific chain that handles formatting the prompt and passing it to the LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Convert the FAISS database into a retriever object configured to return the top 3 most relevant chunks (k=3)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    
    # Combine the retriever and the QA chain into one overarching RAG pipeline
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Return the fully assembled RAG chain, ready to be invoked with a user's query
    return rag_chain