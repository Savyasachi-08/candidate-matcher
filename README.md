# Candidate Matcher — Hybrid Scoring Engine (Gemini + OpenRouter)

A prototype that ranks candidate resumes against a job description using a hybrid scoring model that combines local semantic matching with contextual LLM qualitative assessment.

> **Note:** This repository requires API keys for two services to enable full functionality.

## Features
- **Hybrid Scoring:** Calculates a **Final Hybrid Score** by combining the Vector Similarity Score (50%) and the LLM's Qualitative Match Score (50%).
- **Multi-LLM Support:** Toggle and use **Gemini** or **OpenRouter** models for JD parsing and qualitative scoring.
- **Flexible Input:** Upload the JD and resumes via file upload, or paste **multiple resumes** directly into the text input fields.
- **Core Matching:** Local semantic matching using `sentence-transformers/all-MiniLM-L6-v2` + cosine similarity.
- **Enhanced UX:** Uses a single, centralized Streamlit progress loader (`st.status`) for cleaner processing feedback.

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


