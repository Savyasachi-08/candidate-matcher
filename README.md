# Candidate Matcher — Hybrid Scoring Engine (Gemini + OpenRouter)

A prototype that ranks candidate resumes against a job description using a hybrid scoring model that combines local semantic matching with contextual LLM qualitative assessment.

This repository contains **two implementations**:

- **Baseline Matcher (main branch):**  
  Direct embedding comparison between JD and full resumes + LLM explanations.

- **LLM-Structured Matcher (chunking_resume_before_embedding branch):**  
  JD and resumes are semantically **chunked into sections** by an LLM before embedding → more accurate, section-aware matching.

> **Note:** This repository requires API keys for full functionality.

---

## Features (Baseline System — `main`)
- **Hybrid Scoring:**  
  Final score = **50% Vector Similarity + 50% LLM Score**.
- **Multi-LLM Support:**  
  Works with **Gemini** or **OpenRouter** for JD parsing and candidate explanations.
- **Flexible Input:**  
  Upload files OR paste multiple resumes directly.
- **Local Semantic Search:**  
  Uses `all-mpnet-base-v2` + cosine similarity.
- **Streamlined UX:**  
  Centralized loader (`st.status`) showing each processing phase.

---

## Additional Features (Advanced Structured Version — `chunking_resume_before_embedding`)

- LLM-based **chunking of JD + resumes** into sections: Skills, Experience, Education, Projects, etc.
- **Section-to-Section Matching** with weighted scoring.
- Embedding model: `all-mpnet-base-v2`.
- Final Hybrid Score = **30% Vector Score + 70% LLM Score**.
- Displays **per-section strengths** to explain why a candidate ranked higher.

---

## Quick start (local)

1. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS / Linux
   venv\Scripts\activate.bat     # Windows (powershell/cmd)

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Create a .env file in the project root:
    ```bash
    OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    
4. Run the app:
    ```bash
    streamlit run app.py

## Sample Input Files (for Testing)

The repository includes an `input_files/` directory containing sample documents you can use to test the app immediately:
