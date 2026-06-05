import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import backend modules
from rag_engine import process_uploaded_file, chunk_documents, create_vector_db, chat_with_docs
from summarizer import generate_summary
from quiz_generator import generate_quiz

# -------------------------------------------------------------
# Configuration and Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Personalized Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables (e.g. local GEMINI_API_KEY)
load_dotenv()

# Inject Custom Premium Styles (Midnight Study Theme)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
    /* Global Styles & Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Elegant Title Gradient */
    .main-title {
        background: linear-gradient(135deg, #a855f7, #6366f1, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px rgba(255, 255, 255, 0.08) solid;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }
    .card-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-body {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Styled Submit Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1, #3b82f6);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4f46e5, #2563eb);
        color: white;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0px 24px;
        background-color: transparent;
        color: #94a3b8;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-weight: 600;
    }
    
    /* Chat custom background and bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Score display */
    .score-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 500px;
    }
    .score-num {
        font-size: 3.5rem;
        font-weight: 800;
        color: #818cf8;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "summary_output" not in st.session_state:
    st.session_state.summary_output = None
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_idx" not in st.session_state:
    st.session_state.quiz_idx = 0
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = {}
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.markdown("## 📄 Study Notes & Files")

# Load Gemini API Key silently from environment variables / .env
active_key = os.getenv("GEMINI_API_KEY")

# Note Uploads
st.sidebar.markdown("### 📄 Upload Study Notes")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF or Plain Text)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload PDFs or text notes to create your vector database."
)

# Document processing
if uploaded_files:
    # Check if files uploaded are different from processed
    current_file_names = [f.name for f in uploaded_files]
    needs_processing = current_file_names != st.session_state.processed_files
    
    if needs_processing:
        st.sidebar.info("Files loaded. Click process to index.")
        if st.sidebar.button("⚡ Process & Index Notes"):
            if not active_key:
                st.sidebar.error("Cannot process: Gemini API key missing!")
            else:
                # Use interactive steps instead of a single static spinner
                status_placeholder = st.sidebar.empty()
                
                status_placeholder.info("📖 Step 1/3: Reading and parsing PDF/Text files...")
                all_docs = []
                for uf in uploaded_files:
                    # Process files (PDF/Text)
                    docs = process_uploaded_file(uf)
                    all_docs.extend(docs)
                
                if all_docs:
                    status_placeholder.info("✂️ Step 2/3: Splitting text into study chunks...")
                    chunks = chunk_documents(all_docs)
                    
                    status_placeholder.info("⚡ Step 3/3: Generating embeddings & indexing database...")
                    vector_db = create_vector_db(chunks, active_key)
                    
                    # Store in state
                    st.session_state.vector_store = vector_db
                    st.session_state.processed_files = current_file_names
                    
                    # Reset child features state
                    st.session_state.chat_history = []
                    st.session_state.summary_output = None
                    st.session_state.quiz_questions = None
                    
                    status_placeholder.empty()
                    st.sidebar.success(f"Successfully indexed {len(all_docs)} pages into {len(chunks)} chunks!")
                    st.rerun()
                else:
                    status_placeholder.empty()
                    st.sidebar.error("Could not extract any readable text from the uploaded files.")
    else:
        st.sidebar.success("✅ Notes are fully indexed.")
else:
    if st.session_state.processed_files:
        # User removed files, reset
        st.session_state.processed_files = []
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.summary_output = None
        st.session_state.quiz_questions = None
        st.sidebar.info("Index cleared.")
        st.rerun()

# Display active files list
if st.session_state.processed_files:
    st.sidebar.markdown("#### Indexed Files:")
    for fn in st.session_state.processed_files:
        st.sidebar.markdown(f"- 📄 `{fn}`")

# -------------------------------------------------------------
# Main Application Layout
# -------------------------------------------------------------
st.markdown('<div class="main-title">Personalized Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An intelligent companion that helps you master your course notes.</div>', unsafe_allow_html=True)

# Check if database is ready
if not st.session_state.vector_store:
    # Showcase dashboard layout for onboarding
    st.markdown("### Getting Started:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-header">💬 Chat with Notes</div>
            <div class="card-body">
                Ask specific questions about your notes and get direct answers. 
                Our assistant will read through pages and provide answers along with <b>exact citations and page sources</b> so you can double check.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-header">📝 Topic Summaries</div>
            <div class="card-body">
                Feeling overwhelmed? Enter a topic or ask for a general overview, and get a structured revision guide complete with <b>Glossaries, Core Concepts</b> and <b>Exam Takeaways</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-header">🧠 Practice Quizzes</div>
            <div class="card-body">
                Test your knowledge! Instantly generate multiple-choice quizzes (MCQs) of varying lengths on specific topics. Answer interactively and receive <b>instant explanations</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.info("👈 First, upload your PDF or Text notes in the sidebar to begin!")
    
else:
    # Core Application Tabs
    tab_chat, tab_summary, tab_quiz = st.tabs([
        "💬 Chat Assistant", 
        "📝 Topic Summarizer", 
        "🧠 Practice Quiz"
    ])
    
    # -------------------------------------------------------------
    # Tab 1: Chat Assistant
    # -------------------------------------------------------------
    with tab_chat:
        st.markdown("### 💬 Chat with your Notes")
        st.write("Ask questions about any concepts in your uploaded material. The assistant will answer using context and cite references.")
        
        # Display chat logs
        for idx, message in enumerate(st.session_state.chat_history):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
                # Show citations if present in the message metadata
                if hasattr(message, "additional_kwargs") and "citations" in message.additional_kwargs:
                    citations = message.additional_kwargs["citations"]
                    if citations:
                        with st.expander("🔍 Citations & Sources", expanded=False):
                            for citation in citations:
                                st.markdown(f"**📄 {citation['source']} (Page {citation['page']})**")
                                st.caption(f"\"{citation['snippet']}\"")
                                
        # Chat input
        user_query = st.chat_input("Ask something (e.g. 'Explain the main concepts in chapter 2')")
        
        if user_query:
            if not active_key:
                st.error("Google Gemini API Key missing! Please configure GEMINI_API_KEY in your local .env file.")
            else:
                # 1. Render User Message
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Update history (HumanMessage)
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                
                # 2. Get AI Response
                with st.chat_message("assistant"):
                    with st.spinner("Searching notes and formulating answer..."):
                        # Format chat history for LangChain
                        lc_history = []
                        # Limit history depth to keep context window small
                        for msg in st.session_state.chat_history[-6:-1]: 
                            lc_history.append(msg)
                            
                        # Run RAG
                        answer, citations = chat_with_docs(
                            st.session_state.vector_store,
                            lc_history,
                            user_query,
                            active_key
                        )
                        
                        st.markdown(answer)
                        
                        # Render citations
                        if citations:
                            with st.expander("🔍 Citations & Sources", expanded=False):
                                for citation in citations:
                                    st.markdown(f"**📄 {citation['source']} (Page {citation['page']})**")
                                    st.caption(f"\"{citation['snippet']}\"")
                
                # Update history (AIMessage with citations stored in metadata)
                ai_message = AIMessage(content=answer, additional_kwargs={"citations": citations})
                st.session_state.chat_history.append(ai_message)
                st.rerun()

    # -------------------------------------------------------------
    # Tab 2: Topic Summarizer
    # -------------------------------------------------------------
    with tab_summary:
        st.markdown("### 📝 Generate Topic Summaries")
        st.write("Input any topic covered in your notes to generate a structured, revision-ready study guide summary.")
        
        topic_input = st.text_input(
            "What concept or topic would you like to summarize?", 
            placeholder="e.g. Overview of notes, DNA Replication, Photosynthesis, Chapter 1 summary"
        )
        
        # Quick suggestions
        st.markdown("<small>Quick Suggestions:</small>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3, _ = st.columns([1, 1, 1, 2])
        with col_s1:
            if st.button("📌 General Notes Overview"):
                topic_input = "General Overview of Notes"
        with col_s2:
            if st.button("🔑 Key Glossary"):
                topic_input = "Glossary of Key Terms"
        with col_s3:
            if st.button("💡 Core Theories"):
                topic_input = "Main Concepts and Core Theories"
                
        if st.button("📝 Generate Revision Summary"):
            if not topic_input:
                st.warning("Please enter a topic first.")
            elif not active_key:
                st.error("Google Gemini API Key missing! Please configure GEMINI_API_KEY in your local .env file.")
            else:
                with st.spinner(f"Synthesizing study guide for '{topic_input}'..."):
                    summary = generate_summary(st.session_state.vector_store, topic_input, active_key)
                    st.session_state.summary_output = {
                        "topic": topic_input,
                        "content": summary
                    }
                    st.rerun()
                    
        # Render active summary
        if st.session_state.summary_output:
            summary_data = st.session_state.summary_output
            st.markdown("---")
            st.markdown(f"### Study Guide: **{summary_data['topic']}**")
            
            # Action buttons
            col_dl, col_clr = st.columns([1, 4])
            with col_dl:
                st.download_button(
                    label="💾 Download Summary (Markdown)",
                    data=summary_data["content"],
                    file_name=f"summary_{summary_data['topic'].replace(' ', '_').lower()}.md",
                    mime="text/markdown"
                )
            with col_clr:
                if st.button("🗑️ Clear Summary"):
                    st.session_state.summary_output = None
                    st.rerun()
            
            # Display contents in a container
            st.markdown(summary_data["content"])

    # -------------------------------------------------------------
    # Tab 3: Practice Quiz
    # -------------------------------------------------------------
    with tab_quiz:
        st.markdown("### 🧠 Practice Quiz Generator")
        st.write("Generate interactive Multiple Choice Questions (MCQs) on any topic in your notes to test your understanding.")
        
        # State: quiz not yet generated
        if not st.session_state.quiz_questions:
            col_topic, col_num = st.columns([3, 1])
            with col_topic:
                quiz_topic = st.text_input(
                    "What topic do you want the quiz to cover?", 
                    value="General",
                    placeholder="e.g. Mitochondria, Cellular Respiration, Chapter 1, General"
                )
            with col_num:
                quiz_num = st.slider("Number of Questions", min_value=3, max_value=10, value=5)
                
            if st.button("🧠 Generate Quiz"):
                if not active_key:
                    st.error("Google Gemini API Key missing! Please configure GEMINI_API_KEY in your local .env file.")
                else:
                    with st.spinner(f"Generating a {quiz_num}-question quiz on '{quiz_topic}'..."):
                        quiz_data = generate_quiz(st.session_state.vector_store, quiz_topic, quiz_num, active_key)
                        if quiz_data:
                            st.session_state.quiz_questions = quiz_data
                            st.session_state.quiz_idx = 0
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = {}
                            st.session_state.quiz_score = 0
                            st.rerun()
                        else:
                            st.error("❌ Failed to generate the quiz. This could be due to API limits or parsing issues. Please try again.")
        else:
            # Quiz is in progress or completed
            questions = st.session_state.quiz_questions
            total_questions = len(questions)
            curr_idx = st.session_state.quiz_idx
            
            if curr_idx < total_questions:
                # Active quiz question
                q = questions[curr_idx]
                
                # Question header
                st.markdown(f"#### Question {curr_idx + 1} of {total_questions}")
                
                # Question container
                st.markdown(f"### {q['question']}")
                
                # Manage choices
                is_sub = st.session_state.quiz_submitted.get(curr_idx, False)
                selected_ans = st.session_state.quiz_answers.get(curr_idx, None)
                
                # Render options
                if is_sub:
                    # Question is submitted - show answers static
                    correct_idx = q["answer_index"]
                    
                    for opt_idx, option in enumerate(q["options"]):
                        if opt_idx == correct_idx:
                            # Highlight correct green
                            st.markdown(f"✅ **{option}** *(Correct)*")
                        elif opt_idx == selected_ans:
                            # Highlight selected incorrect red
                            st.markdown(f"❌ ~~{option}~~ *(Your answer)*")
                        else:
                            # Standard format
                            st.markdown(f"⚪ {option}")
                    
                    # Score and Explanation Box
                    if selected_ans == correct_idx:
                        st.success("🎉 Correct!")
                    else:
                        st.error("❌ Incorrect.")
                        
                    st.markdown(f"""
                    <div class="quiz-feedback">
                        <strong>Explanation:</strong> {q['explanation']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Next button
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_next, _ = st.columns([1, 4])
                    with col_next:
                        next_label = "See Results 🏆" if curr_idx == total_questions - 1 else "Next Question ➡️"
                        if st.button(next_label):
                            st.session_state.quiz_idx += 1
                            st.rerun()
                else:
                    # Question not yet submitted - render options select
                    # To prevent streamlit reset issues we use radio
                    # radio needs integer index or value
                    options_list = q["options"]
                    user_selection = st.radio(
                        "Choose the best option:",
                        options=range(len(options_list)),
                        format_func=lambda x: options_list[x],
                        index=None if selected_ans is None else selected_ans,
                        key=f"q_radio_{curr_idx}"
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_submit, _ = st.columns([1, 4])
                    with col_submit:
                        if st.button("Submit Answer ⚡"):
                            if user_selection is None:
                                st.warning("Please select an answer first.")
                            else:
                                st.session_state.quiz_answers[curr_idx] = user_selection
                                st.session_state.quiz_submitted[curr_idx] = True
                                
                                # If correct, increment score
                                if user_selection == q["answer_index"]:
                                    st.session_state.quiz_score += 1
                                st.rerun()
            else:
                # Quiz complete - show Scorecard
                final_score = st.session_state.quiz_score
                percentage = int((final_score / total_questions) * 100)
                
                # Display Scorecard
                st.markdown('<div class="score-container">', unsafe_allow_html=True)
                st.markdown(f"<h2>Quiz Completed!</h2>", unsafe_allow_html=True)
                st.markdown(f'<div class="score-num">{final_score} / {total_questions}</div>', unsafe_allow_html=True)
                st.markdown(f"<h4>Score: {percentage}%</h4>", unsafe_allow_html=True)
                
                # Grade message
                if percentage == 100:
                    st.markdown("<p style='color:#34d399; font-weight:600;'>Perfect Score! 🏆 You have mastered this topic!</p>", unsafe_allow_html=True)
                elif percentage >= 70:
                    st.markdown("<p style='color:#60a5fa; font-weight:600;'>Great job! 👍 You have a strong grasp of these concepts.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color:#f87171; font-weight:600;'>Keep studying! 📚 Review your notes and try again.</p>", unsafe_allow_html=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Review Questions details
                with st.expander("🔍 Review Quiz Questions & Explanations", expanded=False):
                    for idx, q in enumerate(questions):
                        selected = st.session_state.quiz_answers.get(idx, -1)
                        correct = q["answer_index"]
                        
                        st.markdown(f"**Q{idx+1}: {q['question']}**")
                        st.markdown(f"- Correct Answer: `{q['options'][correct]}`")
                        st.markdown(f"- Your Answer: `{q['options'][selected] if selected != -1 else 'None'}`")
                        st.info(f"**Explanation:** {q['explanation']}")
                        if idx < len(questions) - 1:
                            st.markdown("---")
                
                # Restart buttons
                col_reset, col_new, _ = st.columns([1, 1, 3])
                with col_reset:
                    if st.button("🔄 Restart Quiz"):
                        st.session_state.quiz_idx = 0
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = {}
                        st.session_state.quiz_score = 0
                        st.rerun()
                with col_new:
                    if st.button("🗑️ Create New Quiz"):
                        st.session_state.quiz_questions = None
                        st.session_state.quiz_idx = 0
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = {}
                        st.session_state.quiz_score = 0
                        st.rerun()
