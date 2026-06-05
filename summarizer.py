from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from rag_engine import get_best_llm_model

def generate_summary(vector_store, topic, api_key):
    """
    Retrieves chunks relevant to a topic and generates a structured, student-focused study guide summary.
    """
    # Retrieve relevant text chunks for the summary topic
    # If the user asks for a general summary, we retrieve overall content
    search_query = f"Detailed explanations, key terms, definitions, and summaries of: {topic}"
    if topic.lower() in ["general", "all", "everything", "whole document"]:
        search_query = "Detailed summary of the entire document, outlining all major topics and chapters"
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    retrieved_docs = retriever.invoke(search_query)
    
    # Aggregate context
    context_parts = []
    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Notes")
        page_num = doc.metadata.get("page", 1)
        context_parts.append(f"Source: {source_name} (Page {page_num})\nContent:\n{doc.page_content}")
        
    context = "\n---\n".join(context_parts)
    
    # Initialize LLM dynamically
    model_name = get_best_llm_model(api_key)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3  # Low temperature for factual accuracy and consistency
    )
    
    # Prompt template for study summary
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert tutor. Synthesize the provided study notes context into a highly effective, "
            "clear, and structured study summary on the topic '{topic}'.\n\n"
            "Format the summary strictly using markdown with these specific sections:\n"
            "## 📌 Topic Overview\n"
            "Provide a clear, high-level summary (2-4 sentences) of what this topic is about and why it matters.\n\n"
            "## 🔑 Key Terms & Definitions\n"
            "Extract and define the most important terms, abbreviations, or formulas in this section as a bulleted glossary. "
            "Format as: **Term**: definition.\n\n"
            "## 💡 Core Concepts & Explanations\n"
            "Detail the main mechanisms, processes, historical context, or arguments. Use numbered lists, sub-bullets, "
            "or subsections where appropriate to make it extremely easy to read, scan, and study.\n\n"
            "## 📝 Key Takeaways & Exam Tips\n"
            "List 3-5 quick-reference bullet points of the absolute most critical points that are likely to appear on a test or exam. "
            "Highlight what a student should focus on.\n\n"
            "Base this summary strictly on the provided context. Cite the source files and pages where appropriate (e.g. '[Filename (Page X)]'). "
            "Do not make up facts outside the notes.\n\n"
            "STUDY NOTES CONTEXT:\n{context}"
        )),
        ("human", "Generate the study summary now.")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "topic": topic,
        "context": context
    })
    
    return response.content
