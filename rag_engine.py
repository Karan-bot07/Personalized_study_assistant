import io
import pypdf
import google.generativeai as genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def process_uploaded_file(uploaded_file):
    """
    Parses an uploaded file (PDF or TXT) and returns a list of Document objects with page/source metadata.
    """
    documents = []
    file_name = uploaded_file.name
    
    if file_name.lower().endswith('.pdf'):
        # Read PDF bytes
        pdf_file = io.BytesIO(uploaded_file.read())
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # Store metadata for citation
                metadata = {
                    "source": file_name,
                    "page": page_num
                }
                documents.append(Document(page_content=text, metadata=metadata))
    else:
        # Read text file
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        if text.strip():
            metadata = {
                "source": file_name,
                "page": 1
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
    return documents

def chunk_documents(documents, chunk_size=2000, chunk_overlap=300):
    """
    Splits larger documents into smaller, overlapping chunks suitable for embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vector_db(chunks, api_key):
    """
    Generates embeddings via the first available Gemini embedding model and builds an in-memory FAISS index.
    """
    models_to_try = [
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/embedding-001"
    ]
    
    embeddings = None
    last_error = None
    
    for model_name in models_to_try:
        try:
            temp_embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
            # Try a test query to verify model availability
            temp_embeddings.embed_query("test")
            embeddings = temp_embeddings
            # Success!
            break
        except Exception as e:
            last_error = e
            continue
            
    if embeddings is None:
        if last_error:
            raise last_error
        raise ValueError("Could not initialize any Google Generative AI embedding models.")
        
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_best_llm_model(api_key):
    """
    Queries the Google API using the student's API key to find the best available LLM model.
    Avoids hardcoding to prevent 404 NOT_FOUND errors.
    """
    try:
        genai.configure(api_key=api_key)
        # List models and find those supporting text generation
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Preference list (we prefer newer flash models because they are fast and free)
        preferences = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-pro",
            "gemini-1.5-flash",
            "gemini-pro"
        ]
        
        for pref in preferences:
            if pref in available_models:
                return pref
            # Fallback check without models/ prefix if list has short names
            short_pref = pref.replace("models/", "")
            if short_pref in available_models:
                return short_pref
                
        # Fallback to any model with "flash" in its name
        for m in available_models:
            if "flash" in m.lower():
                return m
                
        if available_models:
            return available_models[0]
            
    except Exception as e:
        print(f"Error listing generative models: {e}")
        
    return "gemini-1.5-flash"

def chat_with_docs(vector_store, chat_history, query, api_key):
    """
    Retrieves the most relevant document chunks and answers the user's query using Gemini,
    returning the answer text and a list of citation sources.
    """
    # Retrieve top 5 most relevant document chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    
    # Construct context string and collect citations
    context_parts = []
    citations = []
    
    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Unknown File")
        page_num = doc.metadata.get("page", 1)
        source_id = f"{source_name} (Page {page_num})"
        
        context_parts.append(f"Source: {source_id}\nContent:\n{doc.page_content}\n")
        
        # Add to unique citations
        citation_info = {
            "source": source_name,
            "page": page_num,
            "snippet": doc.page_content
        }
        if citation_info not in citations:
            citations.append(citation_info)
            
    context_text = "\n---\n".join(context_parts)
    
    # Initialize the LLM dynamically
    model_name = get_best_llm_model(api_key)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )
    
    # Prompt forcing citations
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an empathetic, highly structured, and expert student study assistant. "
            "Use the provided context from the student's study notes to answer the question. "
            "Ensure you synthesize the information clearly, using bold text, headings, and bullet points. "
            "Critically, always cite the source file and page when stating facts (e.g., 'As stated in [Filename (Page X)]...'). "
            "If the answer cannot be found in the context, clearly state that but try to provide any relevant "
            "information from the notes that might help them. Do not make up facts outside the notes.\n\n"
            "STUDY NOTES CONTEXT:\n{context}"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{query}")
    ])
    
    chain = prompt_template | llm
    
    response = chain.invoke({
        "context": context_text,
        "chat_history": chat_history,
        "query": query
    })
    
    return response.content, citations
