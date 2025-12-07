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
import tempfile
import errno
import shutil
import time

# chroma import
try:
    import chromadb
except Exception:
    chromadb = None

load_dotenv()

# ------------------- Configuration -------------------
# LLM SERVICE 1: OpenRouter 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"

# LLM SERVICE 2: Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # User must set this in .env
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
GEMINI_MODEL = "gemini-2.5-flash"

# Directories for Chroma persistence and local store
CHROMA_DB_DIR = os.path.expanduser("~/.local/share/candidate_matcher/chroma_db")
LOCAL_STORE_DIR = os.path.join(os.getcwd(), "chroma_store")
CHROMA_COLLECTION = "chunks" # Renamed collection for chunk storage

os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(LOCAL_STORE_DIR, exist_ok=True)

# Define mandatory sections for structured matching
MANDATORY_SECTIONS = ["Skills_Technical", "Skills_Other", "Experience", "Education", "Projects"]

# ------------------- Embedding model -------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

embedder = load_embedder()
EMBED_DIM = 768

# ------------------- Local store files -------------------
EMBED_FILE = os.path.join(LOCAL_STORE_DIR, "embeddings_chunks.npy") 
META_FILE = os.path.join(LOCAL_STORE_DIR, "metadata_chunks.json") 
HASH_FILE = os.path.join(LOCAL_STORE_DIR, "hash_index.json")
ID_LIST_FILE = os.path.join(LOCAL_STORE_DIR, "id_list_chunks.json") 
FULL_DOC_MAP_FILE = os.path.join(LOCAL_STORE_DIR, "full_doc_map.json") 

# in-memory structures
_local_embeddings, _local_metadata = None, None
_id_list: List[int] = [] # Stores chunk IDs (CIDs)
_next_id = 0
_hash_to_id = {} 
_full_doc_map = {} 

# ------------------- Chroma client placeholders -------------------
_chroma_client = None
_chroma_collection = None

# ------------------- Helpers: local store persistence -------------------

def _load_local_store():
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id, _full_doc_map
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
            if os.path.exists(FULL_DOC_MAP_FILE):
                 with open(FULL_DOC_MAP_FILE, "r", encoding="utf-8") as f:
                    _full_doc_map = json.load(f)

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
        st.warning(f"Failed to load local store (continuing with empty store): {e}")
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}
    _full_doc_map = {}

def _save_local_store():
    global _local_embeddings, _local_metadata, _id_list, _hash_to_id, _full_doc_map
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
        with open(FULL_DOC_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(_full_doc_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass

# initialize local store
_load_local_store()

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
    """
    Reset/clear the Chroma collection in a safe way.
    """
    global _chroma_client, _chroma_collection
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id, _full_doc_map

    try:
        for f in [EMBED_FILE, META_FILE, ID_LIST_FILE, HASH_FILE, FULL_DOC_MAP_FILE]:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print("Warning while clearing local files:", e)

    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}
    _full_doc_map = {}

    if chromadb is None:
        return

    client = _chroma_client
    if client is not None:
        try:
            # Delete and recreate collection
            if hasattr(client, "delete_collection"):
                try:
                    client.delete_collection(collection_name=CHROMA_COLLECTION)
                except Exception:
                    pass
            
            # Recreate the client and collection
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR) if hasattr(chromadb, 'PersistentClient') else chromadb.Client()
            _chroma_collection = _chroma_client.create_collection(CHROMA_COLLECTION)
            return
        except Exception as e:
            print("Chroma reset failed:", e) 
            _chroma_client = None
            _chroma_collection = None
    
    if force_delete_dir:
        try:
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            init_chroma()
        except Exception as e:
            print("Full Chroma dir reset failed:", e)

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

# ------------------- Unified LLM Helpers -------------------
import time
from functools import lru_cache 

# Small LRU cache for explanations & JD parsing
@lru_cache(maxsize=64)
def _cached_call_llm_cache_key(service, model, system_prompt, user_prompt):
    return None

def call_openrouter(system_prompt: str, user_prompt: str, max_retries: int = 4, base_backoff: float = 1.0):
    """Robust OpenRouter caller with exponential backoff."""
    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API Key is not configured.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
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
                return choice.get("message", {}).get("content") or choice.get("text")
            return json.dumps(data)
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            last_error = he
            if status == 429 or (status and 500 <= status < 600):
                backoff = base_backoff * (2 ** (attempt - 1))
                time.sleep(min(backoff * (1 + 0.1 * (0.5 - (time.time() % 1))), 30.0))
                continue
            st.error(f"OpenRouter HTTP error {status}: {getattr(he.response, 'text', None)}")
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_e:
            last_error = conn_e
            time.sleep(min(base_backoff * (2 ** (attempt - 1)), 30.0))
            continue
        except Exception as e:
            last_error = e
            break

    st.warning(f"OpenRouter is unavailable or rate-limited. (Retries exhausted.) Last error: {repr(last_error)}")
    return None

def call_gemini(system_prompt: str, user_prompt: str):
    """Simple synchronous Gemini API caller."""
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is not configured.")
        return None

    url = GEMINI_URL.format(GEMINI_MODEL)

    headers = {
        "Content-Type": "application/json",
    }
    
    # Gemini API uses contents for chat/system prompts
    payload = {
        "contents": [
            {"role": "user", "parts": [
                {"text": f"SYSTEM PROMPT: {system_prompt}\n\nUSER PROMPT: {user_prompt}"}
            ]}
        ],
        "generationConfig": { 
            "temperature": 0.0 
        }
    }
    
    try:
        resp = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        if 'candidates' in data and data['candidates']:
            candidate = data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        st.warning(f"Gemini API returned no text or unexpected structure: {json.dumps(data)}")
        return None
        
    except requests.exceptions.HTTPError as he:
        status = getattr(he.response, "status_code", None)
        body = getattr(he.response, 'text', 'No body')
        st.error(f"Gemini HTTP error {status}: {body}")
        return None
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return None

def call_llm(service: str, system_prompt: str, user_prompt: str):
    """Routes the call to the selected LLM service."""
    if service == "OpenRouter":
        return call_openrouter(system_prompt, user_prompt)
    elif service == "Gemini":
        return call_gemini(system_prompt, user_prompt)
    return None

def parse_jd_with_llm(jd_text: str, service: str) -> dict:
    
    safe_empty = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}

    if not service:
        lines = [l.strip("-• \t") for l in jd_text.splitlines() if l.strip()]
        heur = [{"skill": (line[:80] + "...") if len(line) > 80 else line, "description": line, "weight": 0.5, "min_years": None} for line in lines[:4]]
        if heur:
            return {"must_have": heur, "important": [], "nice_to_have": [], "implicit": []}
        return safe_empty

    system_prompt = (
        "You are a recruitment assistant. Given a job description, output STRICT JSON only. "
        "The JSON must have keys: must_have, important, nice_to_have, implicit. "
        "Each key maps to an array of objects with schema: "
        "{\"skill\": str, \"description\": str, \"weight\": float, \"min_years\": int|null}. "
        "Return only valid JSON — do NOT include any explanation or surrounding text."
    )
    user_prompt = f"JOB DESCRIPTION:\n\n{jd_text}\n\nReturn JSON only."
    
    raw = call_llm(service, system_prompt, user_prompt)
    
    # Attempt to parse as before
    if raw:
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
                st.warning(f"{service} output couldn't be parsed as JSON. Showing raw LLM output below for debugging.")
                st.code(raw, language="json")
            except Exception:
                print(f"{service} LLM raw output:", raw)
    return safe_empty


def explain_candidate_with_llm(jd_struct: dict, resume_text: str, base_score: float, service: str) -> str:
    if not service or (service == "OpenRouter" and not OPENROUTER_API_KEY) or (service == "Gemini" and not GEMINI_API_KEY):
        return "(LLM unavailable) Cannot generate explanation." 

    system_prompt = (
        "You are an ATS explainability assistant. Given structured job requirements (JSON) and a candidate resume, "
        "produce a concise (<=200 words) explanation of why the candidate matches or doesn't, listing matched skills, missing skills, and an overall score 0-100. "
        "Crucially, **the last line of your output MUST be only the numeric match score (0-100)**, followed by no other text."
    )
    user_prompt = (
        f"JOB REQUIREMENTS JSON:\n{json.dumps(jd_struct, indent=2)}\n\nRESUME TEXT:\n{resume_text[:6000]}\n\nAVERAGE_SIM_SCORE:{base_score:.4f}"
    )
    
    raw = call_llm(service, system_prompt, user_prompt)
    
    if raw:
        return raw
    
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
    
    fallback_llm_score = min(100, int(base_score * 100 * 1.5))
    
    return (
        f"({service} call failed) Heuristic summary — base_sim: {base_score:.3f}\n"
        f"Matched skills: {', '.join(matched) if matched else 'None'}\n"
        f"Missing skills (sample): {', '.join([m for m in missing if m]) if missing else 'None'}\n"
        f"{fallback_llm_score}" 
    )


def structured_chunk_document_with_llm(doc_text: str, doc_name: str, doc_hash: str, doc_type: str, service: str) -> List[Dict[str, Any]]:
    """
    Uses LLM to chunk a document (JD or Resume) into mandatory, semantic sections.
    Returns a list of chunk dictionaries.
    """
    if not service:
        return []

    mandatory_sections_str = ", ".join(MANDATORY_SECTIONS)
    
    system_prompt = (
        f"You are a document chunking expert. Your task is to extract relevant sections from the provided {doc_type} "
        f"and output the result as STRICT JSON ONLY. The document MUST be broken down into these sections if present: "
        f"{mandatory_sections_str}. "
        f"For RESUME chunks, the 'weight' should be 1.0. For JD chunks, the 'weight' should reflect importance (0.1 to 1.0). "
        f"The section_name must be one of {mandatory_sections_str}. "
        f"Use a list named 'sections' at the root of the JSON. Each object in 'sections' must have keys: "
        f"'section_name', 'chunk_text', and 'weight'."
    )
    user_prompt = f"{doc_type} TEXT:\n\n{doc_text}\n\nReturn JSON only."

    raw = call_llm(service, system_prompt, user_prompt)

    if not raw:
        return []

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
        sections = parsed.get("sections", [])
        
        valid_chunks = []
        for section in sections:
            s_name = section.get("section_name", "").strip()
            c_text = section.get("chunk_text", "").strip()
            if s_name in MANDATORY_SECTIONS and c_text:
                valid_chunks.append({
                    "doc_hash": doc_hash, 
                    "doc_name": doc_name,
                    "section": s_name,
                    "text": c_text,
                    "weight": float(section.get("weight", 1.0)),
                    "doc_type": doc_type
                })
        
        if not valid_chunks:
            pass
        
        return valid_chunks
        
    except Exception as e:
        return []


def add_resume_to_store_and_chunk(filename: str, full_text: str, service: str):
    """
    Chunks resume using LLM and adds each chunk to the local store and Chroma.
    """
    doc_hash = _text_hash(full_text)
    
    if doc_hash in _full_doc_map:
        return

    _full_doc_map[doc_hash] = {"name": filename, "full_text": full_text}
    _save_local_store() 

    chunks = structured_chunk_document_with_llm(full_text, filename, doc_hash, "Resume", service)
    
    if not chunks:
        return

    global _local_embeddings, _local_metadata, _id_list, _next_id, _chroma_collection
    
    new_embeddings = []
    
    for chunk in chunks:
        cid = _next_id
        _next_id += 1
        
        emb = embed_text_np(chunk["text"])
        
        new_embeddings.append(emb.reshape(1, -1))
        
        metadata = {
            "name": chunk["doc_name"], 
            "doc_hash": chunk["doc_hash"], 
            "section": chunk["section"], 
            "weight": chunk["weight"],
            "preview": chunk["text"][:200]
        }
        _local_metadata[cid] = metadata
        _id_list.append(cid)
        _hash_to_id[f'{doc_hash}_{chunk["section"]}'] = cid 
        
        if _chroma_collection is not None:
            try:
                _chroma_collection.add(
                    ids=[str(cid)],
                    documents=[chunk["text"]],
                    metadatas=metadata,
                    embeddings=[emb.tolist()],
                )
            except Exception as e:
                pass 

    if new_embeddings:
        new_embeddings_array = np.vstack(new_embeddings)
        if _local_embeddings.size == 0:
            _local_embeddings = new_embeddings_array.astype(np.float32)
        else:
            _local_embeddings = np.vstack([_local_embeddings, new_embeddings_array]).astype(np.float32)

    _save_local_store()
    

    return doc_hash


def search_chunks_and_aggregate(jd_text: str, service: str, top_k: int = 10):
    """
    Performs chunking on the JD, then for each JD chunk, searches only the corresponding 
    Resume chunks, aggregates scores by original document hash, and returns the top documents.
    """
    if _local_embeddings.size == 0:
        return [], {}
    
    # 1. Chunk the Job Description using the LLM
    jd_doc_hash = _text_hash(jd_text)
    jd_chunks = structured_chunk_document_with_llm(jd_text, "Job Description", jd_doc_hash, "JD", service)
    
    if not jd_chunks:
        st.error("JD chunking failed or returned no sections. Cannot perform section-based matching.")
        return [], {}

    # Map: {doc_hash: {section: weighted_score}}
    doc_scores: Dict[str, Dict[str, float]] = {}
    
    # Organize local metadata by section
    resume_sections: Dict[str, List[Dict[str, Any]]] = {sec: [] for sec in MANDATORY_SECTIONS}
    
    # Populate resume_sections with chunk metadata and embedding index
    for i, cid in enumerate(_id_list):
        meta = _local_metadata[cid]
        if meta.get("section") in resume_sections:
            meta["index"] = i
            meta["cid"] = cid
            resume_sections[meta["section"]].append(meta)

    
    # Iterate through JD chunks and find best match in corresponding resume section
    for jd_chunk in jd_chunks:
        section = jd_chunk["section"]
        jd_weight = jd_chunk["weight"]
        qvec = embed_text_np(jd_chunk["text"])
        
        if section not in resume_sections:
            continue
            
        resume_chunks_in_section = resume_sections[section]
        if not resume_chunks_in_section:
            continue
            
        # Get all embeddings for this section
        indices = [rc['index'] for rc in resume_chunks_in_section]
        if not indices: continue
        
        B_section = _local_embeddings[indices]
        
        # Calculate cosine similarity of JD chunk against all resume chunks in this section
        sims = cosine_sim(qvec, B_section)
        
        # Assign best match score back to the original document hash
        for i, rc in enumerate(resume_chunks_in_section):
            doc_hash = rc["doc_hash"]
            score = sims[i]
            
            # Weighted score for this section match
            weighted_score = score * jd_weight
            
            if doc_hash not in doc_scores:
                doc_scores[doc_hash] = {}
            
            key = section
            
            if key not in doc_scores[doc_hash] or weighted_score > doc_scores[doc_hash][key]:
                doc_scores[doc_hash][key] = weighted_score

    
    # 3. Aggregate final score and format results
    final_results = []
    granular_match_details = {} # New dictionary to hold detailed scores

    # Calculate the normalized total possible weight for the JD
    total_jd_weight = sum(c['weight'] for c in jd_chunks)
    if total_jd_weight == 0: total_jd_weight = 1.0
    
    for doc_hash, scores in doc_scores.items():
        # Sum of the BEST weighted match scores for this document across all JD sections
        total_matched_score = sum(v for k, v in scores.items())
        
        # Normalized final score (0 to 1)
        normalized_score = total_matched_score / total_jd_weight
        
        doc_info = _full_doc_map.get(doc_hash, {"name": f"Unknown Document ({doc_hash[:8]})", "full_text": ""})
        
        final_results.append({
            "id": doc_hash, 
            "score": normalized_score,
            "payload": {"name": doc_info["name"]},
            "preview": doc_info["full_text"][:1000] 
        })
        
        # Store the granular scores for display
        granular_match_details[doc_hash] = [
            {"section": k, "weighted_score": v} 
            for k, v in scores.items()
        ]
        
    # Sort and return top K
    final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
    
    # Return both the main results and the detailed breakdown
    return final_results[:top_k], granular_match_details

# ------------------- Add / search / reset functions (API) -------------------

def reset_local_store():
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id, _full_doc_map
    try:
        for f in [EMBED_FILE, META_FILE, ID_LIST_FILE, HASH_FILE, FULL_DOC_MAP_FILE]:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print(f"Warning while clearing local files: {e}")
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}
    _full_doc_map = {}
    _save_local_store()


# Renamed entry point for adding/chunking resumes
def add_resume_entry_point(filename: str, full_text: str, service: str):
    return add_resume_to_store_and_chunk(filename, full_text, service)


# Renamed entry point for search
def search_entry_point(jd_text: str, service: str, top_k: int = 10):
    # This now runs the new chunking and aggregation logic
    return search_chunks_and_aggregate(jd_text, service, top_k)


# ------------------- Streamlit UI -------------------

# Initialize session state for resume text areas
if 'resume_count' not in st.session_state:
    st.session_state.resume_count = 1

def add_resume_field():
    st.session_state.resume_count += 1

def remove_resume_field():
    if st.session_state.resume_count > 1:
        last_key = f"resume_text_area_{st.session_state.resume_count - 1}"
        if last_key in st.session_state:
            del st.session_state[last_key]
        st.session_state.resume_count -= 1


st.set_page_config(layout="wide", page_title="Candidate Matcher (LLM Structured)")
st.title("Candidate Matcher — LLM Structured Matching")

col1, col2 = st.columns([2, 1])

service_options = []
if GEMINI_API_KEY:
    service_options.append("Gemini")
if OPENROUTER_API_KEY:
    service_options.append("OpenRouter")

LLM_AVAILABLE = bool(service_options)

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

    # --- Resume Inputs ---
    st.subheader("2. Candidate Resumes")
    resume_input_type = st.radio("Resume input method:", ["Paste text", "Upload files"], index=0, key="resume_input_type")
        
    pasted_resumes = []
    uploaded_files = []

    if resume_input_type == "Paste text":
        st.markdown("**Paste Text Resumes:**")
        
        # Initialization block: Ensure session state keys exist *before* the loop
        for i in range(st.session_state.resume_count):
            key = f"resume_text_area_{i}"
            if key not in st.session_state:
                st.session_state[key] = "" 
        
        # Display existing text areas (Now only instantiating the widget, not assigning to state)
        for i in range(st.session_state.resume_count):
            key = f"resume_text_area_{i}"
            
            st.text_area(f"Resume {i+1} Text", height=200, key=key)
            
            # Collect the valid text from session state
            if st.session_state[key].strip():
                pasted_resumes.append((f"Pasted Resume {i+1}", st.session_state[key]))

        # Add/Remove buttons
        col_add, col_remove = st.columns([1, 6])
        with col_add:
            st.button("➕ Add Resume", on_click=add_resume_field, key="add_resume_btn")
        with col_remove:
            if st.session_state.resume_count > 1:
                st.button("➖ Remove Last", on_click=remove_resume_field, key="remove_resume_btn")

    else:
        uploaded_files = st.file_uploader("Upload candidate resumes (PDF/DOCX/TXT)", 
                                           type=["pdf", "docx", "txt"], 
                                           accept_multiple_files=True, 
                                           key="resume_file_uploader")

    st.markdown("---")
    run_btn = st.button("Run Matching")

with col2:
    st.markdown("**Settings**") 
    
    llm_service = None
    if LLM_AVAILABLE:
        if 'llm_service' not in st.session_state:
            st.session_state.llm_service = service_options[0]
            
        llm_service = st.radio("Select LLM Service:", service_options, key="llm_service")
        
        if llm_service == "OpenRouter":
            st.info(f"OpenRouter Model: {OPENROUTER_MODEL}")
        elif llm_service == "Gemini":
            st.info(f"Gemini Model: {GEMINI_MODEL}")
            
        use_llm_for_parsing = True
        use_llm_for_explanations = True
    else:
        use_llm_for_parsing = False
        use_llm_for_explanations = False
        st.warning("No LLM API keys configured. LLM-based chunking/parsing is disabled.")
        
    st.markdown("---") 
    top_k = st.slider("Top K candidates to show explanations for", 1, 10, 5)


if run_btn:
    if not use_llm_for_parsing:
        st.error("LLM API is required for the structured chunking and matching approach. Please configure an API key.")
    elif not jd_text or jd_text.strip() == "":
        st.error("Please provide a job description (paste text or upload file).")
    else:
        all_resumes_to_process = []
        # 1. Process uploaded files
        if uploaded_files:
            for f in uploaded_files:
                try:
                    txt = extract_text_from_file(f)
                    if txt.strip():
                        all_resumes_to_process.append((f.name, txt))
                except Exception as e:
                    st.warning(f"Failed to process uploaded file {f.name}: {e}")

        # 2. Process pasted resumes
        if pasted_resumes:
            for name, text in pasted_resumes:
                if text.strip():
                    all_resumes_to_process.append((name, text))
            
        if not all_resumes_to_process:
            st.error("Please provide at least one resume (paste text or upload file).")
            st.stop()
            
        # --------------------- Main Processing Block (Loader) ---------------------
        with st.status("Processing and Matching Candidates...", expanded=True) as status:
            
            status.update(label="1. Resetting Data Store and Chroma...", state="running")
            # Reset store
            reset_local_store()
            reset_chroma_collection(force_delete_dir=False)
            
            status.update(label="2. Chunking Resumes and Generating Embeddings...", state="running")
            # Add all collected resumes (files and pasted text)
            for name, txt in all_resumes_to_process:
                try:
                    add_resume_entry_point(name, txt, llm_service)
                except Exception as e:
                    st.warning(f"Failed to process resume {name}: {e}")
                    
            status.update(label="3. Parsing Job Description Requirements...", state="running")
            # Parse JD for the explanation format (must_have/important/etc)
            jd_struct = parse_jd_with_llm(jd_text, llm_service)
            
            status.update(label="4. Running Section-to-Section Semantic Match Aggregation...", state="running")
            # Run the new chunking and aggregation search
            results, granular_scores_map = search_entry_point(jd_text, llm_service, top_k=max(top_k, len(all_resumes_to_process)))
            
            status.update(label="5. Preparing Final Scores...", state="running")
            
            # Results are already de-duplicated and aggregated by document hash
            final_candidates_data = []
            candidates = []
            
            # Pre-calculate overall matched skills for efficiency
            all_jd_skills = []
            for group in jd_struct.values():
                if isinstance(group, list):
                    for item in group:
                        skill = item.get("skill", "").strip()
                        if skill:
                            all_jd_skills.append(skill)


            for r in results:
                doc_hash = r.get("id")
                score = r.get("score", 0.0) 
                doc_info = _full_doc_map.get(doc_hash, {})
                
                resume_text_lower = doc_info.get("full_text", "").lower()
                matched_skills_summary = [
                    skill for skill in all_jd_skills 
                    if skill.lower() in resume_text_lower
                ]


                candidate_data = {
                    "name": doc_info.get("name", r["payload"].get("name", f"Resume {doc_hash[:8]}")), 
                    "text": doc_info.get("full_text", "Full text unavailable for LLM explanation."), 
                    "score_vector": float(score),
                    "score_llm_normalized": 0.0, 
                    "score_llm_raw": 0.0, 
                    "score_final": 0.0, 
                    "explanation": "LLM explanation pending.", 
                    "granular_scores": granular_scores_map.get(doc_hash, []), 
                    "matched_skills_summary": matched_skills_summary 
                }
                candidates.append(candidate_data)
            
            
            # --------------------- LLM Explanation and Final Score Calculation ---------------------

            if use_llm_for_explanations:
                status.update(label=f"6. Generating LLM Explanations for Top {min(top_k, len(candidates))} Candidates...", state="running")
                
                # Loop only through the top_k candidates for explanation
                for i, c in enumerate(candidates[:top_k]):
                    try:
                        explanation_raw = explain_candidate_with_llm(jd_struct, c['text'], c['score_vector'], llm_service)
                        
                        # 1. Extract LLM Score (last line of output)
                        parts = explanation_raw.strip().rsplit('\n', 1)
                        llm_score_text = parts[-1].strip()
                        explanation_display = explanation_raw if len(parts) == 1 else parts[0]
                        
                        llm_score_raw = 0.0
                        try:
                            llm_score_raw = float(re.sub(r'[^0-9.]', '', llm_score_text)) 
                            # Normalize LLM score from 0-100 to 0.0-1.0
                            llm_score_normalized = min(1.0, max(0.0, llm_score_raw / 100.0))
                        except ValueError:
                            st.warning(f"Could not parse numeric LLM score for {c['name']}. Using 0.0.")
                            llm_score_normalized = 0.0
                            explanation_display = explanation_raw 

                        # 2. Calculate Final Hybrid Score (70% LLM + 30% Vector)
                        score_vector = c['score_vector']
                        
                        # Weighting: 30% Vector (0.3) + 70% LLM (0.7)
                        final_score = (0.3 * score_vector) + (0.7 * llm_score_normalized)

                        c['score_llm_raw'] = llm_score_raw
                        c['score_llm_normalized'] = llm_score_normalized
                        c['score_final'] = final_score
                        c['explanation'] = explanation_display
                        
                    except Exception as e:
                        st.warning(f"Failed to get explanation for {c['name']}: {e}")
                        c['explanation'] = f"Error: {e}"
            
            status.update(label="7. Final Results Ready!", state="complete")


        # --------------------- INITIAL RANKED LIST (Vector Score) ---------------------
        st.subheader("Ranked Candidates (by Weighted Section Similarity)")
        
        for idx, c in enumerate(candidates):
            st.markdown(f"**{idx+1}. {c['name']}** — Vector Match Score: **{c['score_vector']:.4f}**")
        
            if c['granular_scores']:
                with st.container():
                    st.markdown("**:dart: Top Section Strengths (Score):**")
                    
                    # Sort the granular scores by weighted score (descending)
                    granular_scores_sorted = sorted(c['granular_scores'], key=lambda x: x['weighted_score'], reverse=True)
                    
                    strength_data = []
            
                    
                    for rank, item in enumerate(granular_scores_sorted[:3]):
                        strength_data.append({
                            "Rank": rank + 1,
                            "JD Section": item['section'].replace('_', ' '),
                            "Score": f"{item['weighted_score']:.4f}",
                        })
                    
                    st.dataframe(strength_data, hide_index=True, use_container_width=True)

        # --------------------- LLM Explanation and Final Score Calculation ---------------------

        if use_llm_for_explanations:
            st.subheader(f"Top {min(top_k, len(candidates))} LLM Explanations and Hybrid Scoring (from {llm_service})")
            
            # Loop only through the top_k candidates for explanation (using updated scores)
            for i, c in enumerate(candidates[:top_k]):
                if c.get('score_llm_raw', 0) > 0: 
                    st.markdown(f"**{i+1}. {c['name']}** — Final Hybrid Score: **{c['score_final']:.4f}**")
                    st.write(c['explanation'])
        else:
             st.warning("Skipping LLM explanations and hybrid scoring as no LLM service is configured.")


        # --------------------- Final Score Table ---------------------
        st.markdown("---")
        st.subheader("📊 Final Hybrid Matching Results")
        
        # Prepare data for the table, including all candidates whether explained or not
        table_data = []
        for c in candidates:
            final_score = c['score_final'] if c['score_final'] > 0.0 else c['score_vector']
            
            table_data.append({
                "Rank": "-",
                "Candidate Name": c['name'],
                "Vector Match (0.0-1.0)": f"{c['score_vector']:.4f}",
                "LLM Match (0-100)": f"{c['score_llm_raw']:.1f}" if c['score_llm_raw'] > 0 else "N/A",
                "Final Hybrid Score (0.0-1.0)": f"{final_score:.4f}",
            })
            
        # Sort the table by Final Hybrid Score
        def get_sort_key(item):
             try:
                 return float(item["Final Hybrid Score (0.0-1.0)"])
             except ValueError:
                 return 0.0

        table_data_sorted = sorted(table_data, key=get_sort_key, reverse=True)

        table_data_final_display = table_data_sorted[:top_k]
        
        for i, item in enumerate(table_data_final_display):
            item["Rank"] = i + 1
            
        st.dataframe(table_data_final_display, use_container_width=True)


        # --------------------- Raw Preview (for reference) ---------------------
        st.markdown("---")
        st.subheader("Raw Resume Previews")
        
        for c in candidates:
             with st.expander(f"Preview: {c['name']}"):
                st.write(c["text"][:2000] if len(c["text"]) > 2000 else c["text"])
        
        st.success("Matching Complete!")

st.markdown("---")
st.markdown("Notes: This application uses a Hybrid Matching system to accurately identify candidates whose resumes professionally align with the Job Description requirements.")