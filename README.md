# Candidate Matcher — Streamlit + OpenRouter (Free)

A small prototype that ranks candidate resumes against a job description using local embeddings
(`sentence-transformers/all-MiniLM-L6-v2`) and optional OpenRouter LLM (free models) for
parsing job descriptions and generating explanations.

> **Note:** This repo does **not** contain any API keys. Set `OPENROUTER_API_KEY` as an environment
> variable or GitHub secret.

## Features
- Upload a job description + multiple resumes (PDF/DOCX/TXT) via Streamlit UI.
- Local semantic matching using sentence-transformers + cosine similarity.
- Optional LLM parsing & explanations using OpenRouter free models.
- Toggle LLM usage in the UI.

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
    OPENROUTER_API_KEY="API KEY"

4. Run the app:
    ```bash
    streamlit run app.py


