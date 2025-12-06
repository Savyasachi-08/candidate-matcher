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
META_FILE = os.path.join(LOCAL_STORE_DIR, "metadata.json")
HASH_FILE = os.path.join(LOCAL_STORE_DIR, "hash_index.json")
ID_LIST_FILE = os.path.join(LOCAL_STORE_DIR, "id_list.json")

# in-memory structures
_local_embeddings, _local_metadata = None, None  # will load below
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
            # convert meta keys to ints
            meta = {int(k): v for k, v in meta.items()}
            id_list = [int(x) for x in id_list]
            _local_embeddings = emb.astype(np.float32)
            _local_metadata = meta
            _id_list = id_list
            _hash_to_id = {k: int(v) for k, v in hash_map.items()}
            _next_id = max(_id_list) + 1 if _id_list else 0
            return
    except Exception as e:
        st.warning(f"Failed to load local store (continuing with empty store): {e}")
    # defaults
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}

def _save_local_store():
    global _local_embeddings, _local_metadata, _id_list, _hash_to_id
    try:
        # ensure shapes
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
        st.warning(f"Failed to save local store: {e}")

# initialize local store
_load_local_store()

# ------------------- Chroma init (modern PersistentClient) -------------------

import tempfile
import errno

import tempfile

def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, f".perm_test_{os.getpid()}")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return True
    except Exception:
        return False


import shutil
import time

def init_chroma(prefers_home: bool = True):
    global _chroma_client, _chroma_collection, CHROMA_DB_DIR

    if chromadb is None:
        _chroma_client = None
        _chroma_collection = None
        st.info("Chroma not installed — running local-only.")
        return

    candidates = [
        CHROMA_DB_DIR,
        os.path.expanduser("~/.local/share/candidate_matcher/chroma_db"),
        os.path.join(tempfile.gettempdir(), "candidate_matcher_chroma_db")
    ]

    success_path = None

    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
        except:
            continue

        # test-writable
        try:
            testfile = os.path.join(p, ".write_test")
            with open(testfile, "w") as f:
                f.write("ok")
            os.remove(testfile)
        except:
            continue

        try:
            try:
                client = chromadb.PersistentClient(path=p)
            except:
                client = chromadb.Client()

            try:
                coll = client.get_collection(CHROMA_COLLECTION)
            except:
                coll = client.create_collection(CHROMA_COLLECTION)

            _chroma_client = client
            _chroma_collection = coll
            success_path = p
            break
        except:
            continue

    if success_path:
        st.success(f"Chroma initialized at: {success_path}")
    else:
        st.warning("Chroma could not be initialized — falling back to local-only.")
        _chroma_client = None
        _chroma_collection = None


def reset_chroma_collection(force_delete_dir: bool = False):
    """
    Reset/clear the Chroma collection in a safe way:
    1) If chroma client exists, try to delete the collection via API (preferred).
    2) Re-create an empty collection (client.create_collection).
    3) If client doesn't exist or deletion via API fails, attempt to remove DB files
       under CHROMA_DB_DIR safely (but only when force_delete_dir=True).
    4) Always clear local numpy store files and in-memory structures.
    """
    global _chroma_client, _chroma_collection
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id

    # -------- CLEAR LOCAL NUMPY STORE --------
    try:
        for f in [EMBED_FILE, META_FILE, ID_LIST_FILE, HASH_FILE]:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print("Warning while clearing local files:", e)

    # reset in-memory
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}

    # -------- CHROMA CLIENT RESET --------
    if chromadb is None:
        return

    client = _chroma_client

    # If no client, try to create one
    if client is None:
        try:
            try:
                client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            except:
                client = chromadb.Client()
        except:
            client = None

    # Try API delete
    if client is not None:
        try:
            if hasattr(client, "delete_collection"):
                try:
                    client.delete_collection(collection_name=CHROMA_COLLECTION)
                except TypeError:
                    client.delete_collection(CHROMA_COLLECTION)
        except Exception as e:
            print("Chroma API deletion failed:", e)

        # Try to recreate collection
        try:
            try:
                client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            except:
                client = chromadb.Client()
            _chroma_client = client
            _chroma_collection = client.create_collection(CHROMA_COLLECTION)
            return
        except Exception:
            pass  # fallback below

    # -------- FAILOVER: DELETE DIRECTORY --------
    if force_delete_dir:
        try:
            if os.path.exists(CHROMA_DB_DIR):
                shutil.rmtree(CHROMA_DB_DIR)
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)

            try:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            except:
                _chroma_client = chromadb.Client()

            _chroma_collection = _chroma_client.create_collection(CHROMA_COLLECTION)
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

def embed_text_np(text: str) -> np.ndarray:
    vec = embedder.encode(text, convert_to_tensor=False)
    return np.asarray(vec, dtype=np.float32)

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ------------------- Add / search / reset functions -------------------

def add_resume_to_store(filename: str, full_text: str) -> int:
    """Add resume to local store and to chroma (best-effort). Returns assigned id."""
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id, _chroma_collection

    h = _text_hash(full_text)
    if h in _hash_to_id:
        existing_id = _hash_to_id[h]
        st.info(f"Skipped duplicate upload (already stored): {filename} -> id {existing_id}")
        return existing_id

    rid = _next_id
    _next_id += 1

    emb = embed_text_np(full_text)
    preview = full_text[:1000]

    # append to local matrix
    if _local_embeddings.size == 0:
        _local_embeddings = emb.reshape(1, -1)
    else:
        _local_embeddings = np.vstack([_local_embeddings, emb.reshape(1, -1)])

    _local_metadata[rid] = {"name": filename, "preview": preview}
    _id_list.append(rid)
    _hash_to_id[h] = rid

    # persist local store
    _save_local_store()

    # try to add to chroma collection (best-effort)
    if _chroma_collection is not None:
        try:
            _chroma_collection.add(
                ids=[str(rid)],
                documents=[preview],
                metadatas=[{"name": filename, "hash": h}],
                embeddings=[emb.tolist()],
            )
            try:
                _chroma_client.persist()
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Warning: adding to Chroma failed for {filename}: {e}")

    return rid

def cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a_norm = a / (np.linalg.norm(a) + 1e-12)
    else:
        a_norm = a.reshape(-1) / (np.linalg.norm(a) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sims = (B_norm @ a_norm).reshape(-1)
    return sims

def search_local_by_cosine(jd_text: str, top_k: int = 10):
    if _local_embeddings.size == 0:
        return []
    qvec = embed_text_np(jd_text)
    sims = cosine_sim(qvec, _local_embeddings)
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for pos in top_idx:
        if pos >= len(_id_list):
            continue
        rid = _id_list[pos]
        meta = _local_metadata.get(rid, {})
        results.append({"id": rid, "score": float(sims[pos]), "payload": meta, "preview": meta.get("preview", "")})
    return results

def search_via_chroma_first_then_local(jd_text: str, top_k: int = 10):
    if _chroma_collection is not None:
        try:
            qvec = embed_text_np(jd_text).tolist()
            resp = _chroma_collection.query(
                query_embeddings=[qvec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            ids = resp.get("ids", [[]])[0] if isinstance(resp.get("ids", None), list) else []
            docs = resp.get("documents", [[]])[0] if isinstance(resp.get("documents", None), list) else []
            metadatas = resp.get("metadatas", [[]])[0] if isinstance(resp.get("metadatas", None), list) else []
            distances = resp.get("distances", [[]])[0] if isinstance(resp.get("distances", None), list) else []

            results = []
            for i, sid in enumerate(ids):
                rid = None
                try:
                    rid = int(sid)
                except Exception:
                    try:
                        rid = int(str(sid))
                    except Exception:
                        rid = None
                payload = metadatas[i] if i < len(metadatas) else {}
                doc_preview = docs[i] if i < len(docs) else payload.get("preview", "")
                sim_val = None
                if rid is not None and rid in _local_metadata:
                    pos = _id_list.index(rid)
                    emb = _local_embeddings[pos]
                    qemb = embed_text_np(jd_text)
                    sim_val = float(cosine_sim(qemb, emb.reshape(1, -1))[0])
                else:
                    if i < len(distances) and distances[i] is not None:
                        try:
                            sim_val = float(1.0 - float(distances[i]))
                        except Exception:
                            sim_val = 0.0
                    else:
                        sim_val = 0.0
                results.append({"id": rid, "score": sim_val, "payload": payload, "preview": doc_preview})
            if not results:
                return search_local_by_cosine(jd_text, top_k)
            results = sorted(results, key=lambda x: (x["score"] if x["score"] is not None else -999.0), reverse=True)
            return results[:top_k]
        except Exception as e:
            st.warning(f"Chroma query failed (falling back to local): {e}")
            return search_local_by_cosine(jd_text, top_k)
    else:
        return search_local_by_cosine(jd_text, top_k)

def dedupe_results_by_id_keep_best(results: List[Dict[str, Any]]):
    best = {}
    for r in results:
        rid = r.get("id")
        if rid is None:
            rid = r.get("payload", {}).get("name") or r.get("preview")
        score = r.get("score") if r.get("score") is not None else 0.0
        if rid not in best or score > best[rid]["score"]:
            best[rid] = {"item": r, "score": float(score)}
    deduped = [v["item"] for v in sorted(best.values(), key=lambda x: x["score"], reverse=True)]
    return deduped

def reset_local_store():
    global _local_embeddings, _local_metadata, _id_list, _next_id, _hash_to_id
    try:
        if os.path.exists(EMBED_FILE):
            os.remove(EMBED_FILE)
        if os.path.exists(META_FILE):
            os.remove(META_FILE)
        if os.path.exists(ID_LIST_FILE):
            os.remove(ID_LIST_FILE)
        if os.path.exists(HASH_FILE):
            os.remove(HASH_FILE)
    except Exception as e:
        st.warning(f"Failed to remove local store files: {e}")
    _local_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _local_metadata = {}
    _id_list = []
    _next_id = 0
    _hash_to_id = {}
    _save_local_store()
    st.info("Local embedding store reset (files cleared).")


# ------------------- OpenRouter LLM helpers (unchanged) -------------------

import time
from functools import lru_cache

# Small LRU cache for explanations & JD parsing to avoid repeated LLM calls during development
# (maxsize can be tuned).
@lru_cache(maxsize=64)
def _cached_call_openrouter_cache_key(model, system_prompt, user_prompt):
    # we won't call this directly — wrapper will call requests after checking cache
    return None

def call_openrouter(model: str, system_prompt: str, user_prompt: str, max_retries: int = 4, base_backoff: float = 1.0):
    """
    Robust OpenRouter caller with exponential backoff.
    Does NOT show intermediate retry messages — only shows a single final UI message:
      - success: returns text (no message shown here, caller may st.success)
      - failure after retries: shows ONE st.warning and returns None
    """
    if not OPENROUTER_API_KEY:
        # caller handles missing key
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
            # normal response shape
            if "choices" in data and len(data["choices"]) > 0:
                # handle both chat-style and legacy shapes
                choice = data["choices"][0]
                if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content")
                if isinstance(choice, dict) and "text" in choice:
                    return choice.get("text")
            # fallback: return full json as string
            return json.dumps(data)
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            last_error = he
            # treat 429 and 5xx as retryable
            if status == 429 or (status and 500 <= status < 600):
                backoff = base_backoff * (2 ** (attempt - 1))
                # jitter
                jitter = backoff * 0.1 * (0.5 - (time.time() % 1))
                sleep_t = min(backoff + jitter, 30.0)
                time.sleep(sleep_t)
                continue
            # non-retryable HTTP error -> break and return None
            try:
                body = he.response.text
            except Exception:
                body = None
            st.error(f"OpenRouter HTTP error {status}: {body}")
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_e:
            last_error = conn_e
            backoff = base_backoff * (2 ** (attempt - 1))
            time.sleep(min(backoff, 30.0))
            continue
        except Exception as e:
            last_error = e
            # unexpected -> stop retrying
            break

    # retries exhausted — show a single friendly message and return None
    st.warning("OpenRouter is currently unavailable or rate-limited. Using local fallback. (Retries exhausted.)")
    # optionally print last error to console for debugging (not UI)
    try:
        print("OpenRouter last error:", repr(last_error))
    except Exception:
        pass
    return None


def parse_jd_with_llm(jd_text: str) -> dict:
    """
    Use LLM to parse JD -> structured JSON. If LLM call fails, return empty safe structure.
    The function will show the raw LLM output in the UI if parsing fails, for debugging.
    """
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
        raw = call_openrouter(OPENROUTER_MODEL, system_prompt, user_prompt)
    except Exception as e:
        st.warning(f"OpenRouter call raised an exception: {e}")
        raw = None

    if not raw:
        # fallback: try simple heuristic splitting by lines with bullets or return safe empty
        lines = [l.strip("-• \t") for l in jd_text.splitlines() if l.strip()]
        # pick some lines as "must_have" heuristically (first 4 lines)
        heur = [{"skill": (line[:80] + "...") if len(line) > 80 else line, "description": line, "weight": 0.5, "min_years": None} for line in lines[:4]]
        if heur:
            return {"must_have": heur, "important": [], "nice_to_have": [], "implicit": []}
        return safe_empty

    # attempt to extract JSON defensively
    try:
        raw_clean = html.unescape(raw).strip()
        # strip markdown fences
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
        # show raw LLM output for debugging (but don't crash)
        try:
            st.warning("OpenRouter output couldn't be parsed as JSON. Showing raw LLM output below for debugging.")
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
        f"JOB REQUIREMENTS JSON:\n{json.dumps(jd_struct, indent=2)}\n\nRESUME TEXT:\n{resume_text[:6000]}\n\`nBASE_SIM_SCORE:{base_score:.4f}"
    )
    raw = call_openrouter(OPENROUTER_MODEL, system_prompt, user_prompt)
    if raw:
        return raw
    # fallback succinct explanation
    matched = []
    missing = []
    try:
        # quick heuristic: check if skill names appear in resume
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
        f"(LLM unavailable) Heuristic summary — base_sim: {base_score:.3f}\n"
        f"Matched skills: {', '.join(matched) if matched else 'None'}\n"
        f"Missing skills (sample): {', '.join([m for m in missing if m]) if missing else 'None'}"
    )

# ------------------- Streamlit UI -------------------

st.set_page_config(layout="wide", page_title="Candidate Matcher (Chroma + file store)")
st.title("Candidate Matcher — Streamlit (Chroma + file store)")

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
        # AUTOMATIC CLEAR: reset local store + chroma so only newly uploaded resumes are present
        reset_local_store()
        reset_chroma_collection(force_delete_dir=False)

        st.info("Embedding and storing resumes (local files + Chroma if available)...")
        for f in resumes:
            try:
                txt = extract_text_from_file(f)
                add_resume_to_store(f.name, txt)
            except Exception as e:
                st.warning(f"Failed to process {f.name}: {e}")

        # Parse JD
        with st.spinner("Parsing job description..."):
            if use_llm_for_parsing and OPENROUTER_API_KEY:
                try:
                    jd_struct = parse_jd_with_llm(jd_text)
                except Exception as e:
                    st.error(f"OpenRouter call failed for JD parsing: {e}")
                    st.exception(traceback.format_exc())
                    jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}
            else:
                if use_llm_for_parsing and not OPENROUTER_API_KEY:
                    st.warning("OPENROUTER_API_KEY not set — skipping LLM parsing.")
                jd_struct = {"must_have": [], "important": [], "nice_to_have": [], "implicit": []}

        st.subheader("Parsed Job Requirements (Preview)")
        st.json(jd_struct)

        results = search_via_chroma_first_then_local(jd_text, top_k=max(top_k, len(resumes)))
        # dedupe
        results = dedupe_results_by_id_keep_best(results)

        candidates = []
        for r in results:
            rid = r.get("id")
            score = r.get("score", 0.0) if r.get("score") is not None else 0.0
            payload = r.get("payload") or {}
            preview = r.get("preview") or payload.get("preview", "")
            if (not preview) and (rid in _local_metadata):
                preview = _local_metadata[rid].get("preview", "")
            name = payload.get("name") if isinstance(payload, dict) else None
            name = name or (_local_metadata[rid]["name"] if rid in _local_metadata else f"resume_{rid}")
            candidates.append({"name": name, "text": preview, "sim": float(score)})

        candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)

        st.subheader("Ranked candidates (by cosine similarity)")
        for idx, c in enumerate(candidates):
            st.markdown(f"**{idx+1}. {c['name']}** — base sim: {c['sim']:.4f}")
            with st.expander("Preview resume text"):
                st.write(c["text"][:2000])

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

# Footer
st.markdown("---")
st.markdown(
    "Notes: This app persists embeddings and metadata in ./chroma_store/ (files) and attempts to add them into a Chroma collection in ./chroma_db/.\n"
    "Each run clears the previous store so only newly uploaded resumes are compared."
)
