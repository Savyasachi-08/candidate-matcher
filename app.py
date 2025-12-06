import os
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from io import BytesIO
import requests
import traceback
from dotenv import load_dotenv
import re
import html
import numpy as np
import pathlib
import shutil
import hashlib
from typing import List, Dict, Any

# chroma import
try:
    import chromadb
except Exception:
    chromadb = None

load_dotenv()

# ------------------- Configuration -------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"

# Directories for Chroma persistence and local store
CHROMA_DB_DIR = os.path.expanduser("~/.local/share/candidate_matcher/chroma_db")
LOCAL_STORE_DIR = os.path.join(os.getcwd(), "chroma_store")
CHROMA_COLLECTION = "resumes"

os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(LOCAL_STORE_DIR, exist_ok=True)

# ------------------- Embedding model -------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
EMBED_DIM = 384

# ------------------- Local store files -------------------
EMBED_FILE = os.path.join(LOCAL_STORE_DIR, "embeddings.npy")
META_FILE = os.path.join(LOCAL_STORE_DIR, "metadata_chunks.json")
HASH_FILE = os.path.join(LOCAL_STORE_DIR, "hash_index.json") 
ID_LIST_FILE = os.path.join(LOCAL_STORE_DIR, "id_list_chunks.json") 

# in-memory structures
_local_embeddings, _local_metadata = None, None 
_id_list: List[int] = [] 
_next_id = 0 
_hash_to_id = {} 

# ------------------- Chroma client placeholders -------------------
_chroma_client = None
_chroma_collection = None

# ------------------- Helpers: local store persistence -------------------

def _load_local_store():
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id
    try:
        if os.path.exists(EMBED_FILE) and os.path.exists(META_FILE) and os.path.exists(ID_LIST_FILE):
            emb = np.load(EMBED_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(ID_LIST_FILE, "r", encoding="utf-8") as f:
                id_list = json.load(f)
            if os.path.exists(HASH_FILE):
                with open(HASH_FILE, "r", encoding="utf-8") as f:
                    hash_map = json.load(f)
            else:
                hash_map = {}
            
            meta = {int(k): v for k, v in meta.items()}
            id_list = [int(x) for x in id_list]
            _local_embeddings = emb.astype(np.float32)
            _local_metadata = meta
            _id_list = id_list
            _hash_to_id = {k: int(v) for k, v in hash_map.items()}
            all_ids = list(_id_list) + list(_hash_to_id.values())
            _next_id = max(all_ids) + 1 if all_ids else 0
            return
    except Exception as e:
        # NOTE: This warning is kept for developer debugging but is not part of the final UI goal
        # st.warning(f"Failed to load local store (continuing with empty store): {e}") 
        pass 
        
    # defaults
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}

def _save_local_store():
    global _local_embeddings, _local_metadata, _id_list, _hash_to_id
    try:
        if _local_embeddings is None:
            arr = np.zeros((0, EMBED_DIM), dtype=np.float32)
        else:
            arr = _local_embeddings.astype(np.float32)
        np.save(EMBED_FILE, arr)
        
        meta_str_keys = {str(k): v for k, v in _local_metadata.items()}
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta_str_keys, f, ensure_ascii=False, indent=2)
            
        with open(ID_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump([int(x) for x in _id_list], f)
            
        with open(HASH_FILE, "w", encoding="utf-8") as f:
            json.dump({k: int(v) for k, v in _hash_to_id.items()}, f)
            
    except Exception as e:
        pass

# initialize local store
_load_local_store()

# ------------------- Chroma init (Suppressed Output) -------------------

def init_chroma(prefers_home: bool = True):
    global _chroma_client, _chroma_collection, CHROMA_DB_DIR

    if chromadb is None:
        st.info("Chroma not installed — running local-only.")
        return

    success_path = None
    candidates = [CHROMA_DB_DIR]

    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            testfile = os.path.join(p, ".write_test")
            with open(testfile, "w") as f:
                f.write("ok")
            os.remove(testfile)

            try:
                client = chromadb.PersistentClient(path=p)
            except Exception:
                client = chromadb.Client()

            try:
                coll = client.get_collection(CHROMA_COLLECTION)
            except Exception:
                coll = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})

            _chroma_client = client
            _chroma_collection = coll
            success_path = p
            break
        except Exception as e:
            continue

    if success_path:
        pass
    else:
        st.warning("Chroma could not be initialized — falling back to local-only.")
        _chroma_client = None
        _chroma_collection = None


def reset_chroma_collection(force_delete_dir: bool = False):
    global _chroma_client, _chroma_collection
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id

    try:
        for f in [EMBED_FILE, META_FILE, ID_LIST_FILE, HASH_FILE]:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print("Warning while clearing local files:", e)

    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}

    if chromadb is None:
        return

    client = _chroma_client

    if client is not None:
        try:
            client.delete_collection(collection_name=CHROMA_COLLECTION)
        except Exception:
            pass
        
        try:
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            _chroma_collection = _chroma_client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
            return
        except Exception:
            pass

    if force_delete_dir:
        try:
            if os.path.exists(CHROMA_DB_DIR):
                shutil.rmtree(CHROMA_DB_DIR)
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            init_chroma()
        except Exception as e:
            print("Full Chroma dir reset failed:", e)
            _chroma_client = None
            _chroma_collection = None

init_chroma()

# ------------------- Utility functions -------------------

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

def extract_sections_from_text(full_text: str) -> List[Dict[str, str]]:
    """Splits the full resume text into sections based on common headers."""
    section_headers = [
        "experience", "work experience", "professional experience",
        "education", "skills", "technical skills", "projects",
        "certifications", "summary", "profile", "contact"
    ]
    pattern = r"^\s*(" + "|".join(section_headers) + r")\s*[:\s]*(\n|$)"
    
    chunks = []
    parts = re.split(pattern, full_text, flags=re.IGNORECASE | re.MULTILINE)
    
    current_content = parts[0].strip()
    if current_content:
        chunks.append({'header': 'Summary/Contact', 'content': current_content})

    current_header = None
    current_content = ""
    for i in range(1, len(parts)):
        is_header = any(re.match(h, parts[i].strip(), re.IGNORECASE) for h in section_headers)
        
        if is_header and parts[i].strip():
            if current_header and current_content.strip():
                chunks.append({'header': current_header.title(), 'content': current_content.strip()})
            
            current_header = parts[i].strip()
            current_content = ""
        elif parts[i] is not None:
            current_content += parts[i]
            
    if current_header and current_content.strip():
        chunks.append({'header': current_header.title(), 'content': current_content.strip()})
    
    if not chunks and full_text.strip():
         chunks.append({'header': 'Full Text (No Sections Found)', 'content': full_text.strip()})
         
    final_chunks = []
    seen_content = set()
    for chunk in chunks:
        if chunk['content'] and len(chunk['content']) > 50 and chunk['content'] not in seen_content:
            final_chunks.append(chunk)
            seen_content.add(chunk['content'])
            
    return final_chunks

def embed_text_np(text: str) -> np.ndarray:
    vec = embedder.encode(text, convert_to_tensor=False)
    return np.asarray(vec, dtype=np.float32)

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ------------------- Add / search / reset functions -------------------

def add_resume_to_store(filename: str, full_text: str) -> int:
    """Add resume chunks to local store and to chroma (best-effort). Returns assigned *resume* id."""
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id, _chroma_collection

    h = _text_hash(full_text)
    if h in _hash_to_id:
        existing_id = _hash_to_id[h]
        st.info(f"Skipped duplicate upload (already stored): {filename} -> id {existing_id}")
        return existing_id

    resume_id = _next_id
    _next_id += 1

    chunks = extract_sections_from_text(full_text)
    if not chunks:
        st.warning(f"Could not extract sections from {filename}. Storing full text as single chunk.")
        chunks = [{'header': 'Full Document', 'content': full_text}]
        
    chunk_ids = []
    chunk_embeddings = []
    chunk_metadatas = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_rid = _next_id
        _next_id += 1
        
        chunk_text = chunk['content']
        chunk_header = chunk['header']
        
        emb = embed_text_np(chunk_text)
        preview = chunk_text[:500]
        
        # Local store update
        if _local_embeddings.size == 0:
            _local_embeddings = emb.reshape(1, -1)
        else:
            _local_embeddings = np.vstack([_local_embeddings, emb.reshape(1, -1)])
            
        _local_metadata[chunk_rid] = {
            "name": filename, 
            "preview": preview, 
            "resume_id": resume_id, 
            "section_header": chunk_header
        }
        _id_list.append(chunk_rid)
        
        # Collect data for Chroma batch insert
        chunk_ids.append(str(chunk_rid))
        chunk_embeddings.append(emb.tolist())
        chunk_metadatas.append({
            "name": filename, 
            "hash": h, 
            "resume_id": str(resume_id), 
            "section_header": chunk_header
        })
    
    _hash_to_id[h] = resume_id
    _save_local_store()

    if _chroma_collection is not None and chunk_ids:
        try:
            # FIX: Build the 'documents' list using the actual chunk data
            chroma_documents = [
                f"{c['header']}: {c['content'][:500]}"
                for c in chunks 
            ]
            
            _chroma_collection.add(
                ids=chunk_ids,
                documents=chroma_documents, 
                metadatas=chunk_metadatas,
                embeddings=chunk_embeddings,
            )
            try:
                _chroma_client.persist()
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Warning: adding chunks to Chroma failed for {filename}: {e}")

    return resume_id

def cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.shape[0] == 0:
        return np.array([])
    if a.ndim == 1:
        a_norm = a / (np.linalg.norm(a) + 1e-12)
    else:
        a_norm = a.reshape(-1) / (np.linalg.norm(a) + 1e-12)
    
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    
    sims = (B_norm @ a_norm).reshape(-1)
    return sims


def search_local_by_cosine(jd_text: str, top_k: int = 10):
    """Fallback search against local chunks."""
    if _local_embeddings.size == 0:
        return []
        
    qvec = embed_text_np(jd_text)
    sims = cosine_sim(qvec, _local_embeddings)
    
    chunk_results = []
    for pos, sim_val in enumerate(sims):
        chunk_id = _id_list[pos]
        meta = _local_metadata.get(chunk_id, {})
        resume_id = meta.get("resume_id")

        if resume_id is not None:
             chunk_results.append({
                "resume_id": resume_id, 
                "score": float(sim_val), 
                "payload": meta, 
                "preview": meta.get("section_header", "") + ": " + meta.get("preview", "")
            })
    return chunk_results


def search_via_chroma_first_then_local(jd_text: str, top_k: int = 10):
    """Aggregates score by simple average of all chunks with positive similarity."""
    
    chunk_results = []
    qvec = embed_text_np(jd_text)
    qvec_list = qvec.tolist()

    if _chroma_collection is not None:
        try:
            resp = _chroma_collection.query(
                query_embeddings=[qvec_list],
                n_results=100, 
                include=["documents", "metadatas", "distances"],
            )
            
            ids = resp.get("ids", [[]])[0] if isinstance(resp.get("ids", None), list) else []
            metadatas = resp.get("metadatas", [[]])[0] if isinstance(resp.get("metadatas", None), list) else []
            distances = resp.get("distances", [[]])[0] if isinstance(resp.get("distances", None), list) else []

            for i, sid in enumerate(ids):
                meta = metadatas[i]
                resume_id = int(meta.get("resume_id")) if meta and meta.get("resume_id") else None
                
                sim_val = 0.0
                if i < len(distances) and distances[i] is not None:
                    try:
                        sim_val = float(1.0 - float(distances[i]))
                    except Exception:
                        pass
                
                if resume_id is not None:
                    doc_preview = resp.get("documents", [[]])[0][i] if resp.get("documents") else ""
                    chunk_results.append({
                        "resume_id": resume_id, 
                        "score": sim_val, 
                        "payload": meta, 
                        "preview": doc_preview
                    })
        except Exception as e:
            st.warning(f"Chroma query failed (falling back to local): {e}")
            chunk_results = search_local_by_cosine(jd_text, top_k=100)
    else:
        chunk_results = search_local_by_cosine(jd_text, top_k=100)


    # --- STEP 3: Aggregate Chunk Scores (Average of All Positive Chunks) ---
    
    resume_chunk_scores: Dict[int, List[float]] = {}
    resume_details: Dict[int, Dict[str, Any]] = {}
    chunk_previews: Dict[int, List[Dict[str, Any]]] = {} 

    for r in chunk_results:
        rid = r["resume_id"]
        score = r["score"]
        name = r["payload"].get("name", f"resume_{rid}")
        
        if score > 0: # Only consider positive scores
            if rid not in resume_chunk_scores:
                resume_chunk_scores[rid] = []
                resume_details[rid] = {"name": name}
                chunk_previews[rid] = []
            
            resume_chunk_scores[rid].append(score)
            chunk_previews[rid].append({"score": score, "preview": r["preview"]})

    final_results = []
    
    for rid, scores in resume_chunk_scores.items():
        if not scores:
            continue
            
        average_score = sum(scores) / len(scores)
            
        chunk_previews[rid].sort(key=lambda x: x["score"], reverse=True)
        TOP_N_DISPLAY = 3
        
        top_previews = [
            f"**{i+1}.** [{p['score']:.4f}] {p['preview']}" 
            for i, p in enumerate(chunk_previews[rid][:TOP_N_DISPLAY])
        ]
        
        score_note = f"\n(Score is the average of **{len(scores)}** positive-scoring sections.)"
        
        final_results.append({
            "id": rid, 
            "score": average_score, 
            "payload": {"name": resume_details[rid]["name"]},
            "preview": "\n\n---\n\n".join(top_previews) + score_note, 
            "full_resume_text_temp": "" 
        })
        
    final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
    return final_results[:top_k]


def dedupe_results_by_id_keep_best(results: List[Dict[str, Any]]):
    return results

def reset_local_store():
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id
    try:
        for f in [EMBED_FILE, META_FILE, ID_LIST_FILE, HASH_FILE]:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print("Warning while clearing local files:", e)
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}
    _save_local_store()
    st.info("Local embedding store reset (files cleared).")


# ------------------- OpenRouter LLM helpers (NO CACHING DECORATOR) -------------------

import time

def call_openrouter(model: str, system_prompt: str, user_prompt: str, max_retries: int = 4, base_backoff: float = 1.0):
    if not OPENROUTER_API_KEY:
        return None

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

    last_error = None
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content")
                if isinstance(choice, dict) and "text" in choice:
                    return choice.get("text")
            return json.dumps(data)
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            last_error = he
            if status == 429 or (status and 500 <= status < 600):
                backoff = base_backoff * (2 ** (attempt - 1))
                jitter = backoff * 0.1 * (0.5 - (time.time() % 1))
                sleep_t = min(backoff + jitter, 30.0)
                time.sleep(sleep_t)
                continue
            try:
                print("OpenRouter HTTP error body:", he.response.text)
            except Exception:
                pass
            st.error(f"OpenRouter HTTP error {status}: {he}")
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_e:
            last_error = conn_e
            backoff = base_backoff * (2 ** (attempt - 1))
            time.sleep(min(backoff, 30.0))
            continue
        except Exception as e:
            last_error = e
            break

    st.warning("OpenRouter is currently unavailable or rate-limited. Using local fallback. (Retries exhausted.)")
    return None


def parse_jd_with_llm(jd_text: str) -> dict:
    safe_empty = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}
    system_prompt = (
        "You are a recruitment assistant. Given a job description, output STRICT JSON only. "
        "The JSON must have keys: must_have, important, nice_to_have, implicit. "
        "Each key maps to an array of objects with schema: "
        "{\"skill\": str, \"description\": str, \"weight\": float, \"min_years\": int|null}. "
        "Return only valid JSON — do NOT include any explanation or surrounding text."
    )
    user_prompt = f"JOB DESCRIPTION:\n\n{jd_text}\n\nReturn JSON only."
    raw = None
    try:
        # LLM parsing is ALWAYS attempted here as per the prompt's implied need
        raw = call_openrouter(OPENROUTER_MODEL, system_prompt, user_prompt)
    except Exception as e:
        st.warning(f"OpenRouter call raised an exception: {e}")
        raw = None

    if not raw:
        lines = [l.strip("-• \t") for l in jd_text.splitlines() if l.strip()]
        heur = [{"skill": (line[:80] + "...") if len(line) > 80 else line, "description": line, "weight": 0.5, "min_years": None} for line in lines[:4]]
        if heur:
            return {"must_have": heur, "important": [], "nice_to_have": [], "implicit": []}
        return safe_empty

    try:
        raw_clean = html.unescape(raw).strip()
        raw_clean = re.sub(r"^```(?:json)?\n?|\n?```$", "", raw_clean).strip()
        first_brace = raw_clean.find("{")
        last_brace = raw_clean.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            raw_json = raw_clean[first_brace:last_brace+1]
        else:
            raw_json = raw_clean
        parsed = json.loads(raw_json)
        return parsed
    except Exception:
        try:
            st.warning("LLM output couldn't be parsed as JSON. Showing raw LLM output below for debugging.")
            st.code(raw, language="json")
        except Exception:
            print("LLM raw output:", raw)
        return safe_empty

def explain_candidate_with_llm(jd_struct: dict, resume_text: str, base_score: float) -> str:
    system_prompt = (
        "You are an ATS explainability assistant. Given structured job requirements (JSON) and a candidate resume, "
        "produce a concise (<=200 words) explanation of why the candidate matches or doesn't, listing matched skills, missing skills, and an overall score 0-100."
    )
    user_prompt = (
        f"JOB REQUIREMENTS JSON:\n{json.dumps(jd_struct, indent=2)}\n\nRESUME TEXT:\n{resume_text[:6000]}\n\nAVERAGE_SIM_SCORE:{base_score:.4f}"
    )
    raw = call_openrouter(OPENROUTER_MODEL, system_prompt, user_prompt)
    if raw:
        return raw
    
    # fallback succinct explanation
    matched = []
    missing = []
    try:
        for group in ["must_have", "important", "nice_to_have"]:
            for item in jd_struct.get(group, []):
                sk = item.get("skill", "")
                if sk and sk.lower() in resume_text.lower():
                    matched.append(sk)
                else:
                    missing.append(sk)
    except Exception:
        pass
    matched = matched[:6]
    missing = missing[:6]
    return (
        f"(LLM unavailable) Heuristic summary — avg_sim: {base_score:.3f}\n"
        f"Matched skills: {', '.join(matched) if matched else 'None'}\n"
        f"Missing skills (sample): {', '.join([m for m in missing if m]) if missing else 'None'}"
    )


# ------------------- Streamlit UI (FINAL MODIFIED BLOCK) -------------------

st.set_page_config(layout="wide", page_title="Candidate Matcher (Chunking & Average Scoring)")
st.title("Candidate Matcher — Streamlit (Chunking & Average Scoring)")

col1, col2 = st.columns([2, 1])

# Set LLM usage flags to True (always try to use them if key is available)
USE_LLM_FOR_PARSING = True
USE_LLM_FOR_EXPLANATIONS = True

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
    # ONLY SLIDER REMAINS
    top_k = st.slider("Top K candidates to show", 1, 10, 5)

if run_btn:
    if not jd_text or jd_text.strip() == "":
        st.error("Please provide a job description (paste text or upload file).")
    elif not resumes or len(resumes) == 0:
        st.error("Please upload at least one resume.")
    else:
        reset_local_store()
        reset_chroma_collection(force_delete_dir=False)

        st.info("Embedding and storing resume chunks (local files + Chroma if available)...")
        
        uploaded_file_map = {} 
        for f in resumes:
            try:
                txt = extract_text_from_file(f)
                rid = add_resume_to_store(f.name, txt)
                uploaded_file_map[rid] = {"name": f.name, "full_text": txt}
            except Exception as e:
                st.warning(f"Failed to process {f.name}: {e}")
                
        if not uploaded_file_map:
             st.error("No resumes were successfully processed or stored.")
             st.stop()
             
        # Parse JD (always attempt LLM parsing)
        with st.spinner("Parsing job description..."):
            if USE_LLM_FOR_PARSING and OPENROUTER_API_KEY:
                try:
                    jd_struct = parse_jd_with_llm(jd_text)
                except Exception as e:
                    st.error(f"OpenRouter call failed for JD parsing: {e}")
                    st.exception(traceback.format_exc())
                    jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}
            else:
                if USE_LLM_FOR_PARSING and not OPENROUTER_API_KEY:
                    st.warning("OPENROUTER_API_KEY not set — skipping LLM parsing and using heuristic fallback.")
                jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}

        st.subheader("Parsed Job Requirements (Preview)")
        st.json(jd_struct)

        # Search against chunks and aggregate
        results = search_via_chroma_first_then_local(jd_text, top_k=max(top_k, len(resumes)))
        results = dedupe_results_by_id_keep_best(results) 

        candidates = []
        for r in results:
            rid = r.get("id")
            score = r.get("score", 0.0) if r.get("score") is not None else 0.0
            
            file_info = uploaded_file_map.get(rid, {})
            name = file_info.get("name", f"resume_{rid}")
            full_text = file_info.get("full_text", "")

            candidates.append({
                "name": name, 
                "full_text": full_text, 
                "best_chunks_preview": r.get("preview", ""), 
                "sim": float(score)
            })

        candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)

        st.subheader("Ranked candidates (by Average of All Positive Chunks)")
        for idx, c in enumerate(candidates):
            st.markdown(f"**{idx+1}. {c['name']}** — **AVERAGE SIM SCORE**: {c['sim']:.4f}")
            with st.expander("Top 3 Matching Chunks & Scores"):
                st.markdown(c["best_chunks_preview"])
                
        # Explanations (always attempt LLM explanations)
        if USE_LLM_FOR_EXPLANATIONS and OPENROUTER_API_KEY:
            st.subheader(f"Top {min(top_k, len(candidates))} explanations (from OpenRouter)")
            for c in candidates[:top_k]:
                try:
                    with st.spinner(f"Explaining {c['name']}..."):
                        explanation = explain_candidate_with_llm(jd_struct, c['full_text'], c['sim']) 
                    st.markdown(f"**{c['name']}** — explanation:")
                    st.write(explanation)
                except Exception as e:
                    st.warning(f"Failed to get explanation for {c['name']}: {e}")
        elif USE_LLM_FOR_EXPLANATIONS and not OPENROUTER_API_KEY:
            st.warning("OPENROUTER_API_KEY not configured — explanation LLM skipped.")

        st.success("Done")

# Footer
st.markdown("---")
st.markdown(
    "Notes: This app uses **resume chunking** and an **average similarity score of all positive-matching chunks** for ranking.\n"
    "Embeddings are persisted in local files and Chroma.\n"
    "Each run clears the previous store."
)