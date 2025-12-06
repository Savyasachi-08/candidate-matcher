import os
import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
from io import BytesIO
import requests
import traceback
from dotenv import load_dotenv
import re
import html


load_dotenv() 

# ------------------- Configuration -------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free model recommended (may change over time) - you can change this to any free OpenRouter model
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"


# ------------------- Helper functions -------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        elif name.endswith(".docx") or name.endswith('.doc'):
            uploaded_file.seek(0)
            doc = docx.Document(BytesIO(uploaded_file.read()))
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs)
        else:
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"Failed to extract text from {uploaded_file.name}: {e}")
        return ""


def call_openrouter(model: str, system_prompt: str, user_prompt: str, max_retries: int = 1) -> str:
    """Call OpenRouter chat completion endpoint and return assistant text. Requires OPENROUTER_API_KEY env var."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set. Set the environment variable to enable LLM calls.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt >= max_retries:
                raise
            else:
                continue

def parse_jd_with_llm(jd_text: str) -> dict:
    """
    Ask the LLM to produce structured JSON. Robustly extract JSON if the model
    wraps it in text. Returns the parsed dict or a safe empty structure and
    logs the raw output to the Streamlit UI for debugging.
    """
    system_prompt = (
        "You are a recruitment assistant. Given a job description, output STRICT JSON only. "
        "The JSON must have keys: must_have, important, nice_to_have, implicit. "
        "Each key maps to an array of objects with schema: "
        "{\"skill\": str, \"description\": str, \"weight\": float, \"min_years\": int|null}. "
        "Return only valid JSON — do NOT include any explanation or surrounding text."
    )
    user_prompt = f"JOB DESCRIPTION:\n\n{jd_text}\n\nReturn JSON only."

    raw = call_openrouter(OPENROUTER_MODEL, system_prompt, user_prompt)

    #unescape HTML entities and strip surrounding code fences/backticks
    raw_clean = html.unescape(raw).strip()
    # Remove common markdown code fences if present
    raw_clean = re.sub(r"^```(?:json)?\\n?|\\n?```$", "", raw_clean).strip()

    first_brace = raw_clean.find("{")
    last_brace = raw_clean.rfind("}")
    candidate_json = None

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate_json = raw_clean[first_brace:last_brace+1]

    if not candidate_json:
        first_brack = raw_clean.find("[")
        last_brack = raw_clean.rfind("]")
        if first_brack != -1 and last_brack != -1 and last_brack > first_brack:
            candidate_json = raw_clean[first_brack:last_brack+1]

    if candidate_json:
        try:
            parsed = json.loads(candidate_json)
            return parsed
        except Exception:
            pass

    # If we get here: parsing failed. Expose raw output to the UI for debugging,
    # and fall back to an empty structure to keep app working.
    try:
        st.warning("OpenRouter output couldn't be parsed as JSON. Showing raw LLM output below for debugging.")
        st.code(raw_clean, language="json")
    except Exception:
        # In case Streamlit cannot show code for some reason, just print
        print("LLM raw output (couldn't parse JSON):", raw_clean)

    return {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}

def explain_candidate_with_llm(jd_struct: dict, resume_text: str, base_score: float) -> str:
    system_prompt = (
        "You are an ATS explainability assistant. Given structured job requirements (JSON) and a candidate resume, "
        "produce a concise (<=200 words) explanation of why the candidate matches or doesn't, listing matched skills, missing skills, and an overall score 0-100."
    )
    user_prompt = (
        f"JOB REQUIREMENTS JSON:\n{json.dumps(jd_struct, indent=2)}\n\nRESUME TEXT:\n{resume_text[:6000]}\n\nBASE_SIM_SCORE:{base_score:.4f}"
    )
    return call_openrouter(OPENROUTER_MODEL, system_prompt, user_prompt)


def compute_similarity(jd_text: str, resume_text: str) -> float:
    """
    Compute cosine similarity between JD and resume embeddings.
    Uses .item() on the result to avoid the numpy scalar conversion warning.
    """
    jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
    cv_emb = embedder.encode(resume_text, convert_to_tensor=True)
    cos = util.cos_sim(jd_emb, cv_emb)
    # cos might be a 1x1 tensor — use .item() to get a Python float safely
    try:
        sim = float(cos.cpu().item())
    except Exception:
        # Fallback: convert to numpy and take first element
        sim = float(cos.cpu().numpy().reshape(-1)[0])
    return sim



# ------------------- Streamlit App -------------------
st.set_page_config(layout="wide", page_title="Candidate Matcher (OpenRouter Free)")
st.title("Candidate Matcher — Streamlit (OpenRouter free models)")

col1, col2 = st.columns([2, 1])

with col1:
    jd_input_type = st.radio("Job description input:", ["Paste text", "Upload file"], index=0)
    if jd_input_type == "Paste text":
        jd_text = st.text_area("Paste Job Description", height=300)
    else:
        jd_file = st.file_uploader("Upload JD file (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jdfile")
        jd_text = ""
        if jd_file:
            jd_text = extract_text_from_file(jd_file)

    st.markdown("---")
    resumes = st.file_uploader("Upload candidate resumes (multiple)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    st.markdown("---")
    run_btn = st.button("Run Matching")

with col2:
    st.markdown("**Settings**")
    use_llm_for_parsing = st.checkbox("Use OpenRouter LLM to parse JD to structured requirements", value=True)
    use_llm_for_explanations = st.checkbox("Use OpenRouter LLM to generate explanations", value=True)
    top_k = st.slider("Top K candidates to show explanations for", 1, 10, 5)


if run_btn:
    if not jd_text or jd_text.strip() == "":
        st.error("Please provide a job description (paste text or upload file).")
    elif not resumes or len(resumes) == 0:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Parsing job description..."):
            if use_llm_for_parsing:
                if OPENROUTER_API_KEY:
                    try:
                        jd_struct = parse_jd_with_llm(jd_text)
                    except Exception as e:
                        st.error("OpenRouter call failed for JD parsing: %s" % str(e))
                        st.exception(traceback.format_exc())
                        jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}
                else:
                    st.warning("OPENROUTER_API_KEY not set — skipping LLM parsing. Falling back to empty structure.")
                    jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}
            else:
                jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}

        # Show parsed JD structure
        st.subheader("Parsed Job Requirements (Preview)")
        st.json(jd_struct)

        # Compute embeddings & similarity locally
        candidates = []
        progress_bar = st.progress(0)
        total = len(resumes)
        for i, f in enumerate(resumes):
            try:
                txt = extract_text_from_file(f)
                sim = compute_similarity(jd_text, txt)
                candidates.append({"name": f.name, "text": txt, "sim": sim})
            except Exception as e:
                st.warning(f"Failed to process {f.name}: {e}")
            progress_bar.progress((i + 1) / total)

        candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)

        st.subheader("Ranked candidates (by semantic similarity)")
        for idx, c in enumerate(candidates):
            st.markdown(f"**{idx+1}. {c['name']}** — base sim: {c['sim']:.4f}")
            with st.expander("Preview resume text"):
                st.write(c["text"][:2000])

        # Generate explanations for top_k
        if use_llm_for_explanations and OPENROUTER_API_KEY:
            st.subheader(f"Top {min(top_k, len(candidates))} explanations (from OpenRouter)")
            for c in candidates[:top_k]:
                try:
                    with st.spinner(f"Explaining {c['name']}..."):
                        explanation = explain_candidate_with_llm(jd_struct, c['text'], c['sim'])
                    st.markdown(f"**{c['name']}** — explanation:")
                    st.write(explanation)
                except Exception as e:
                    st.warning(f"Failed to get explanation for {c['name']}: {e}")
        elif use_llm_for_explanations and not OPENROUTER_API_KEY:
            st.warning("OPENROUTER_API_KEY not configured — enable it to generate natural-language explanations.")

        st.success("Done")

# Footer - quick tips
st.markdown("---")
st.markdown(
    "**Quick tips:** Use the checkboxes to toggle whether to call the OpenRouter LLM. If you don't set `OPENROUTER_API_KEY`, the app will still run local semantic matching but won't call the LLM for parsing or explanations."
)
