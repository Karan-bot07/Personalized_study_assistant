# 🎓 Personalized Study Assistant

A premium, AI-powered study companion designed for students. Upload your study notes (PDFs or Text files) to chat with them, generate structured study guides/summaries, and test your knowledge with interactive, auto-generated quizzes.

Built with a gorgeous, dark-themed **Midnight Study** UI, glassmorphic elements, and micro-interactions.

---

## ⚡ Features

1. **💬 Chat Assistant (RAG)**
   - Ask questions about your notes.
   - Get responses with **exact citations** indicating the source file and page numbers.
   - Collapsible references let you review the exact context text.

2. **📝 Topic Summarizer**
   - Input any specific topic or ask for a general notes overview.
   - Generates structured revision guides containing:
     - **Topic Overview**: High-level context.
     - **Key Terms & Definitions**: A glossary of essential concepts.
     - **Core Concepts & Explanations**: Detailed breakdowns with numbered/bulleted steps.
     - **Key Takeaways & Exam Tips**: Cheat-sheet bullet points for test prep.
   - Export and download your summaries as Markdown (`.md`) files.

3. **🧠 Practice Quiz Generator**
   - Auto-generate interactive Multiple Choice Questions (MCQs) for any topic in your notes.
   - Select answer choices and receive **instant visual feedback** (green for correct, red for incorrect).
   - Provides **detailed explanation cards** explaining *why* the correct answer is right.
   - View your final score, check a progress bar, and review all questions with a collapsible scorecard at the end.

---

## 🛠️ Tech Stack & Design Choices

To ensure maximum reliability and speed on modern Python environments (including Python 3.14.2 on Windows):
* **Frontend**: [Streamlit](https://streamlit.io/) with custom HTML/CSS injections for a premium "Midnight Study" theme.
* **Orchestration**: [LangChain](https://www.langchain.com/) for document loading, splitting, and RAG chains.
* **LLM**: **Google Gemini 1.5 Flash** (fast, accurate, and generous free tier).
* **Embeddings**: **Google Gemini Embeddings (`embedding-001`)** via API. This avoids large local downloads (~1GB PyTorch model) and ensures rapid startup.
* **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) for efficient, zero-setup, in-memory local indexing.
* **Parser**: `pypdf` (pure Python parser, highly stable).

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have **Python 3.9+** installed. (Tested up to **Python 3.14.2**).

### 2. Installation
Clone this repository and set up a virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it:
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Configuration
1. Get a free Gemini API key from the [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root directory and add your key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   *Note: If no `.env` file is present, you can also enter your API key directly in the application's sidebar.*

### 4. Running the Web App
Run the Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to start studying!
