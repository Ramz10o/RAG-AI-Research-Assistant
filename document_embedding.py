# Import the layout-aware PDF loader from LangChain's community package
from langchain_community.document_loaders import UnstructuredPDFLoader
# Import the Document schema used to store text and its associated metadata
from langchain_core.documents import Document
# Import the text splitter that intelligently breaks down text based on character limits
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(file_path):
    # Initialize the PDF loader with the specified file path
    # 'mode="elements"' tells the loader to return individual document components (titles, paragraphs, lists)
    # 'strategy="fast"' bypasses heavy vision models and uses PDF metadata/font sizes for fast layout analysis
    loader = UnstructuredPDFLoader(
        file_path,
        mode="elements",
        strategy="fast", 
    )
    
    # Execute the loading process, extracting the categorized layout elements from the PDF
    elements = loader.load()
    
    # Initialize a fallback title in case the parser cannot find one on the first page
    document_title = "Unknown Title"
    # Create an empty list to store our final structured document chunks
    processed_chunks = []
    # Create a string variable to temporarily hold text as we build a single semantic chunk
    current_chunk_text = ""
    
    # Iterate through every extracted layout element (e.g., a single paragraph or a single heading)
    for el in elements:
        # Safely retrieve the layout category type (e.g., 'Title', 'NarrativeText') from the element's metadata
        category = el.metadata.get("category", "")
        
        # 1. Dynamically capture the main Title
        # If we hit the first 'Title' element and haven't set a title yet, assume it's the paper's main title
        if category == "Title" and document_title == "Unknown Title":
            document_title = el.page_content
            
        # 2. Semantic Chunking Strategy
        # If we encounter a new 'Title' (like a section header) and we already have text in our buffer...
        if category == "Title" and current_chunk_text != "":
            # Package the accumulated text into a LangChain Document object
            # Prepend the main paper title to the text to ensure semantic context is never lost during retrieval
            new_doc = Document(
                page_content=f"Paper Title: {document_title}\n---\n{current_chunk_text}",
                # Store the title and file source in the official metadata dictionary for potential future filtering
                metadata={"title": document_title, "source": file_path}
            )
            # Add the completed section to our list of processed chunks
            processed_chunks.append(new_doc)
            
            # Reset the buffer and start it with the new section header we just found
            current_chunk_text = el.page_content + "\n"
        else:
            # If it's just regular text (not a new section header), append it to the current buffer
            current_chunk_text += el.page_content + " "
            
    # Append the final chunk after the loop finishes
    # This catches whatever text was left in the buffer when the document ended
    if current_chunk_text.strip() != "":
         new_doc = Document(
            page_content=f"Paper Title: {document_title}\n---\n{current_chunk_text}",
            metadata={"title": document_title, "source": file_path}
        )
         processed_chunks.append(new_doc)
         
    # 3. Fallback Length Limiter
    # Initialize a secondary chunker to ensure massive semantic sections don't blow up the LLM's context window
    # It splits text strictly at 1200 characters, with a 250-character overlap to preserve sentence flow
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=250
    )
    
    # Pass our semantically grouped documents through the length-limiter
    final_chunks = text_splitter.split_documents(processed_chunks)
         
    # Return the fully processed, enriched, and sized chunks ready for embedding
    return final_chunks