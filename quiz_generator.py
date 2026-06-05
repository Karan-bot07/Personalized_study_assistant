import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from rag_engine import get_best_llm_model

def generate_quiz(vector_store, topic, num_questions, api_key):
    """
    Retrieves chunks relevant to a topic, generates a set of multiple choice questions (MCQs)
    using Gemini, and returns a parsed JSON list of question dictionaries.
    """
    # Retrieve relevant text chunks for the quiz topic
    # If the user asks for a general quiz, we retrieve overall content
    search_query = f"Core concepts, definitions, details, and facts about: {topic}"
    if topic.lower() in ["general", "all", "everything", "whole document"]:
        search_query = "Main summary, key terms, definitions, and major highlights of the document"
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    retrieved_docs = retriever.invoke(search_query)
    
    # Aggregate context
    context_parts = []
    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Notes")
        page_num = doc.metadata.get("page", 1)
        context_parts.append(f"Source: {source_name} (Page {page_num})\nContent:\n{doc.page_content}")
        
    context = "\n---\n".join(context_parts)
    
    # Configure Gemini with native JSON output formatting dynamically
    model_name = get_best_llm_model(api_key)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,  # Higher temperature allows for creative and diverse questions
        model_kwargs={"response_mime_type": "application/json"}
    )
    
    # Prompt template for quiz generation
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert educator who designs high-quality, challenging multiple-choice quizzes for students. "
            "Your task is to generate a quiz of exactly {num_questions} multiple-choice questions (MCQs) based ONLY on the provided context "
            "about the topic '{topic}'. "
            "Ensure the questions test actual conceptual understanding and factual knowledge, rather than minor trivia. "
            "You MUST return the output as a valid JSON array of objects. Do not wrap the JSON in Markdown code blocks or any other text. "
            "Each object in the array must follow this exact schema:\n"
            "[\n"
            "  {{\n"
            "    \"question\": \"Question text here?\",\n"
            "    \"options\": [\n"
            "      \"Option A\",\n"
            "      \"Option B\",\n"
            "      \"Option C\",\n"
            "      \"Option D\"\n"
            "    ],\n"
            "    \"answer_index\": 0, // 0-based index of the correct option in the options array\n"
            "    \"explanation\": \"A thorough explanation explaining why this option is correct based on the text.\"\n"
            "  }}\n"
            "]\n\n"
            "STUDY NOTES CONTEXT:\n{context}"
        )),
        ("human", "Generate the quiz now in JSON.")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "num_questions": num_questions,
            "topic": topic,
            "context": context
        })
        
        content = response.content.strip()
        
        # Strip markdown markers if the model accidentally included them
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
            
        quiz_data = json.loads(content)
        
        # Ensure it's a list
        if not isinstance(quiz_data, list):
            if isinstance(quiz_data, dict) and "quiz" in quiz_data:
                quiz_data = quiz_data["quiz"]
            else:
                return None
                
        return quiz_data
    except Exception as e:
        print(f"Error generating or parsing quiz: {e}")
        # Return none to let frontend display an error message
        return None
