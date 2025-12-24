#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable

import numpy as np
from tqdm import tqdm

# --- Third-party deps ---
try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from bson import ObjectId
except Exception as e:
    print("[error] Missing or failed to import pymongo. Please run: pip install pymongo", file=sys.stderr)
    raise

try:
    import faiss  # provided by faiss-cpu
except Exception:
    print("[error] Missing or failed to import faiss. Please run: pip install faiss-cpu", file=sys.stderr)
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("[error] Missing or failed to import sentence-transformers. Please run: pip install sentence-transformers", file=sys.stderr)
    raise

# llama-cpp-python is optional until you actually query
try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # Defer error until query time


# --- Constants & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
# Default to a fast 3B model for CPU-only laptops.
# Download it once via: python download_llm_model.py
DEFAULT_MODEL_PATH = MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"

# Generation defaults tuned for CPU-only hardware
DEFAULT_LLM_N_CTX = int(os.getenv("LLM_N_CTX", "2048"))
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))
DEFAULT_LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", "256"))
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss.index"
ID_MAP_PATH = PROJECT_ROOT / "id_map.json"

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_db"
COLLECTION_NAME = "passages"

# Embedding configuration: prefer local llama.cpp embeddings if available
# Default to the local nomic-embed-text-v1.5 GGUF model, with a
# SentenceTransformers model as a fallback when the GGUF is missing.
EMBED_MODEL_NAME = "nomic-embed-text-v1.5"  # fallback SentenceTransformers model name
EMBED_MODEL_PATH = MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf"  # local embedding .gguf


# --- Utilities ---
@dataclass
class Passage:
    mongo_id: str
    source_id: str
    text: str


def log(msg: str) -> None:
    print(msg, flush=True)


def get_collection() -> Collection:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Force connection test
        _ = client.admin.command("ping")
    except Exception as e:
        log("[error] Could not connect to MongoDB at mongodb://localhost:27017")
        log(f"        {e}")
        log("        Make sure MongoDB Community Server is installed and running.")
        sys.exit(2)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def ensure_dirs() -> None:
    (PROJECT_ROOT / "data" / "docs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)


#############################
# Multi-format ingestion    #
#############################
# Optional imports; absence will silently skip related formats.
try:
    import PyPDF2  # PDF text extraction
except Exception:
    PyPDF2 = None
try:
    import docx  # python-docx for .docx
except Exception:
    docx = None
try:
    import pandas as pd  # csv/xlsx
except Exception:
    pd = None
try:
    import pytesseract  # OCR
    # Set Tesseract path for Windows if installed in default location
    import os
    default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract
except Exception:
    pytesseract = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    from pdf2image import convert_from_path  # scanned PDF pages to images
except Exception:
    convert_from_path = None
try:
    import oracledb  # Oracle DB ingestion (optional)
except Exception:
    oracledb = None
import xml.etree.ElementTree as ET

SUPPORTED_FILE_EXT = {
    ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".json", ".xml",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}

def _extract_txt(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"[warn] txt read failed {fp}: {e}")
        return ""

def _ocr_image(fp: Path) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(fp)
        txt = pytesseract.image_to_string(img)
        return txt.strip()
    except Exception as e:
        log(f"[warn] image OCR failed {fp}: {e}")
        return ""

def _extract_docx(fp: Path) -> str:
    if docx is None:
        return ""
    try:
        d = docx.Document(str(fp))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        log(f"[warn] docx read failed {fp}: {e}")
        return ""

def _extract_tabular(fp: Path) -> str:
    if pd is None:
        return ""
    try:
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp, dtype=str, keep_default_na=False, encoding="utf-8")
        else:
            df = pd.read_excel(fp, dtype=str, keep_default_na=False)
        rows: List[str] = []
        for _, row in df.iterrows():
            vals = [v.strip() for v in row.tolist() if isinstance(v, str) and v.strip()]
            if vals:
                rows.append(" | ".join(vals))
        return "\n".join(rows)
    except Exception as e:
        log(f"[warn] tabular read failed {fp}: {e}")
        return ""

def _extract_json(fp: Path) -> str:
    try:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
    except Exception as e:
        log(f"[warn] json parse failed {fp}: {e}")
        return ""
    collected: List[str] = []
    def walk(x: Any):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (str, int, float)):
            s = str(x).strip()
            if s:
                collected.append(s)
    walk(obj)
    return "\n".join(collected)

def _extract_xml(fp: Path) -> str:
    try:
        tree = ET.parse(fp)
        root = tree.getroot()
    except Exception as e:
        log(f"[warn] xml parse failed {fp}: {e}")
        return ""
    texts: List[str] = []
    for elem in root.iter():
        if elem.text:
            t = elem.text.strip()
            if t:
                texts.append(t)
    return "\n".join(texts)

def _ocr_pdf_pages(fp: Path) -> str:
    if pytesseract is None or convert_from_path is None or Image is None:
        return ""
    try:
        pages = convert_from_path(str(fp))
    except Exception as e:
        log(f"[warn] pdf rasterization failed {fp}: {e}")
        return ""
    out: List[str] = []
    for page_img in pages:
        try:
            txt = pytesseract.image_to_string(page_img)
            if txt.strip():
                out.append(txt)
        except Exception:
            continue
    if out:
        log(f"[ingest] OCR fallback used for scanned PDF: {fp.name}")
    return "\n".join(out)

def _extract_pdf(fp: Path) -> str:
    # Try embedded text first
    if PyPDF2 is None:
        return ""
    parts: List[str] = []
    try:
        with fp.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                    if t.strip():
                        parts.append(t)
                except Exception:
                    continue
    except Exception as e:
        log(f"[warn] pdf read failed {fp}: {e}")
        return ""
    if parts:
        return "\n".join(parts)
    # Fallback OCR if no embedded text
    return _ocr_pdf_pages(fp)

EXTRACTORS: Dict[str, Callable[[Path], str]] = {
    ".txt": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".csv": _extract_tabular,
    ".xlsx": _extract_tabular,
    ".json": _extract_json,
    ".xml": _extract_xml,
    ".png": _ocr_image,
    ".jpg": _ocr_image,
    ".jpeg": _ocr_image,
    ".tif": _ocr_image,
    ".tiff": _ocr_image,
}

def discover_documents(folder: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for fp in sorted(folder.iterdir()):
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        if ext not in SUPPORTED_FILE_EXT:
            continue
        extractor = EXTRACTORS.get(ext)
        if extractor is None:
            continue
        text = extractor(fp)
        if not text.strip():
            log(f"[warn] Skipping empty/unsupported {fp.name}")
            continue
        out.append((fp, text))
    return out


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        chunk_size = 250
    if overlap < 0:
        overlap = 0
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


# --- Ingest ---

def ingest_files() -> None:
    """Ingest multi-format documents in data/docs/ into MongoDB."""
    ensure_dirs()
    coll = get_collection()
    log("[ingest] Scanning data/docs/ for txt/pdf/docx/csv/xlsx/json/xml/images ...")
    docs = discover_documents(DATA_DIR)
    if not docs:
        log(f"[ingest] No supported files found in {DATA_DIR}")
        return
    to_insert: List[Dict[str, Any]] = []
    total_chunks = 0
    for fp, text in docs:
        chunks = chunk_text(text, chunk_size=250, overlap=50)
        for idx, chunk in enumerate(chunks):
            source_id = f"{fp.name}#{idx}"
            to_insert.append({
                "source": fp.name,
                "source_id": source_id,
                "text": chunk,
                "created_at": datetime.utcnow(),
            })
        total_chunks += len(chunks)
    if not to_insert:
        log("[ingest] No chunks produced from input files.")
        return
    log(f"[ingest] Inserting {len(to_insert)} chunks ({total_chunks} chunks total) ...")
    try:
        coll.insert_many(to_insert)
    except Exception as e:
        log(f"[error] Failed to insert into MongoDB: {e}")
        sys.exit(3)
    log(f"[ingest] Inserted {total_chunks} chunks from {len(docs)} files into rag_db.passages")

def ingest_oracle(dsn: str, user: str, password: str, query: str) -> None:
    """Ingest rows from an Oracle table into MongoDB."""
    if oracledb is None:
        log("[oracle] oracledb not installed. Run: pip install oracledb")
        return
    ensure_dirs()
    coll = get_collection()
    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
    except Exception as e:
        log(f"[oracle] Connection failed: {e}")
        return
    rows: List[Tuple[Any, ...]] = []
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    except Exception as e:
        log(f"[oracle] Query failed: {e}")
        return
    inserts: List[Dict[str, Any]] = []
    for r in rows:
        try:
            # Expecting (id, title, body) minimal schema; adapt as needed.
            doc_id, title, body = r[:3]
            text = f"{title}\n{body}" if body else str(title)
            for idx, chunk in enumerate(chunk_text(text)):
                source_id = f"oracle:{doc_id}#{idx}"
                inserts.append({
                    "source": f"oracle:{doc_id}",
                    "source_id": source_id,
                    "text": chunk,
                    "created_at": datetime.utcnow(),
                })
        except Exception:
            continue
    if inserts:
        coll.insert_many(inserts)
    log(f"[oracle] Inserted {len(inserts)} chunks from {len(rows)} rows")

def ingest_mongo_source(uri: str, db_name: str, collection: str, text_field: str = "text", id_field: Optional[str] = None) -> None:
    """Ingest existing documents from another MongoDB collection.
    Each document must contain a text_field; optional id_field used for source_id prefix."""
    ensure_dirs()
    target = get_collection()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
    except Exception as e:
        log(f"[mongo-src] Connection failed: {e}")
        return
    src_coll = client[db_name][collection]
    cursor = src_coll.find({},{text_field:1, id_field:1 if id_field else 1})
    inserts: List[Dict[str, Any]] = []
    count = 0
    for doc in cursor:
        raw = doc.get(text_field, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        base = str(doc.get(id_field)) if id_field else str(doc.get("_id"))
        for idx, chunk in enumerate(chunk_text(raw)):
            inserts.append({
                "source": f"mongo:{collection}",
                "source_id": f"mongo:{base}#{idx}",
                "text": chunk,
                "created_at": datetime.utcnow(),
            })
            count += 1
    if inserts:
        target.insert_many(inserts)
    log(f"[mongo-src] Inserted {count} chunks from collection {collection}")


# --- Build FAISS Index ---

def _load_embedder():
    """
    Return an embedder with an .encode(texts, normalize_embeddings=True, convert_to_numpy=True) API.

    Preference order:
      1) Local llama.cpp embedding using EMBED_MODEL_PATH (.gguf) via llama-cpp-python
      2) SentenceTransformers model (fallback)
    """
    # Try llama.cpp embeddings first if Llama is available and model file exists
    if Llama is not None and EMBED_MODEL_PATH.exists():
        try:
            log(f"[index] Loading local llama.cpp model for embeddings: {EMBED_MODEL_PATH} ...")
            llm = Llama(model_path=str(EMBED_MODEL_PATH), n_ctx=2048, embedding=True)

            class LlamaEmbedder:
                def encode(self, texts: List[str], normalize_embeddings: bool = True, convert_to_numpy: bool = True, **kwargs):
                    vecs: List[np.ndarray] = []
                    for t in texts:
                        out = llm.create_embedding(t)
                        emb = np.array(out["data"][0]["embedding"], dtype="float32")
                        if normalize_embeddings:
                            n = np.linalg.norm(emb)
                            if n > 0:
                                emb = emb / n
                        vecs.append(emb)
                    arr = np.vstack(vecs)
                    return arr if convert_to_numpy else arr.tolist()

            log("[index] Using llama.cpp for embeddings")
            return LlamaEmbedder()
        except Exception as e:
            log(f"[index] Failed to initialize llama.cpp embeddings, falling back: {e}")

    # Fallback: SentenceTransformers
    log(f"[index] Loading embedding model: {EMBED_MODEL_NAME} (may download on first use) ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model


def build_index(batch_size: int = 64, incremental: bool = True) -> None:
    """Build or extend the FAISS index.

    If incremental is True and existing artifacts (faiss.index, id_map.json) exist,
    only new MongoDB passages (those whose _id is not already present) are embedded
    and appended. Otherwise a full rebuild is performed.
    """
    ensure_dirs()
    coll = get_collection()

    existing_ids: set[str] = set()
    id_map: List[Dict[str, str]] = []
    index: Optional[faiss.Index] = None

    use_incremental = incremental and FAISS_INDEX_PATH.exists() and ID_MAP_PATH.exists()
    if use_incremental:
        try:
            # Load existing index & id map
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            raw_map = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(raw_map, list):
                for entry in raw_map:
                    if isinstance(entry, dict) and "mongo_id" in entry:
                        mid = entry.get("mongo_id")
                        if mid:
                            existing_ids.add(mid)
                id_map = raw_map  # continue extending
            log(f"[index] Loaded existing index with {len(id_map)} vectors for incremental update.")
        except Exception as e:
            log(f"[index] Failed to load existing index/id_map; falling back to full rebuild: {e}")
            use_incremental = False  # force full rebuild

    # Query passages: all if full rebuild else only new ones
    log("[index] Fetching passages from MongoDB ...")
    mongo_query: Dict[str, Any] = {}
    if use_incremental and existing_ids:
        # Exclude already indexed ids
        mongo_query = {"_id": {"$nin": [ObjectId(x) for x in existing_ids]}}

    try:
        cursor = coll.find(mongo_query, {"text": 1, "source_id": 1})
        passages = list(cursor)
    except Exception as e:
        log(f"[error] Failed to query MongoDB: {e}")
        sys.exit(4)

    if not passages:
        if use_incremental:
            log("[index] No new passages to add. Index is up to date.")
        else:
            log("[index] No passages found. Run --ingest first.")
        return

    texts = [p.get("text", "") for p in passages]
    src_ids = [p.get("source_id", "") for p in passages]
    mongo_ids = [str(p.get("_id")) for p in passages]

    model = _load_embedder()
    log("[index] Encoding passages (normalized embeddings, cosine via inner product) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")

    if index is None:
        # Full rebuild path
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        log("[index] Creating new FAISS index ...")
    else:
        log("[index] Appending to existing FAISS index ...")

    # Add vectors
    index.add(embeddings)  # type: ignore[arg-type]

    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
    except Exception as e:
        log(f"[error] Failed to write FAISS index to {FAISS_INDEX_PATH}: {e}")
        sys.exit(5)

    # Extend mapping in same order
    for mid, sid in zip(mongo_ids, src_ids):
        id_map.append({"mongo_id": mid, "source_id": sid})
    try:
        ID_MAP_PATH.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[error] Failed to write id_map.json: {e}")
        sys.exit(6)

    action_word = "Appended" if use_incremental else "Built"
    log(f"[index] {action_word} {len(mongo_ids)} passages. Total vectors = {len(id_map)}")
    log(f"[index] Saved index -> {FAISS_INDEX_PATH}")
    log(f"[index] Saved id_map -> {ID_MAP_PATH}")


# --- Retrieve ---

def _load_index_and_map() -> Tuple[faiss.Index, List[Dict[str, str]]]:
    if not FAISS_INDEX_PATH.exists():
        log(f"[retrieve] Missing FAISS index at {FAISS_INDEX_PATH}. Run --build-index first.")
        sys.exit(7)
    if not ID_MAP_PATH.exists():
        log(f"[retrieve] Missing id_map at {ID_MAP_PATH}. Run --build-index first.")
        sys.exit(8)
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
    except Exception as e:
        log(f"[error] Failed to read FAISS index: {e}")
        sys.exit(9)

    try:
        id_map = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
        assert isinstance(id_map, list)
    except Exception as e:
        log(f"[error] Failed to read/parse id_map.json: {e}")
        sys.exit(10)
    return index, id_map


def retrieve(query: str, top_k: int = 5) -> List[Passage]:
    coll = get_collection()
    index, id_map = _load_index_and_map()

    model = _load_embedder()
    log("[retrieve] Encoding query ...")
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    log(f"[retrieve] Searching top-{top_k} passages ...")
    # Ensure query embedding contiguous before search
    q_emb = np.ascontiguousarray(q_emb, dtype="float32")
    # Perform search; Pylance may mis-report missing parameters (k, distances, labels)
    scores, idxs = index.search(q_emb, top_k)  # type: ignore[call-arg]
    hits = idxs[0].tolist()

    passages: List[Passage] = []
    for rank, pos in enumerate(hits):
        if pos < 0 or pos >= len(id_map):
            continue
        meta = id_map[pos]
        mid = meta.get("mongo_id")
        sid = meta.get("source_id", "")
        try:
            doc = coll.find_one({"_id": ObjectId(mid)}, {"text": 1, "source_id": 1})
        except Exception as e:
            log(f"[warn] Failed to fetch Mongo doc {mid}: {e}")
            continue
        if not doc:
            continue
        passages.append(Passage(mongo_id=mid, source_id=sid or doc.get("source_id", ""), text=doc.get("text", "")))

    if not passages:
        log("[retrieve] No passages retrieved.")
    else:
        log("[retrieve] Retrieved passages:")
        for i, p in enumerate(passages, 1):
            preview = (p.text[:120] + "...") if len(p.text) > 120 else p.text
            log(f"  {i}. [{p.source_id}] {preview}")

    return passages


# --- Generation with llama-cpp-python ---

def _load_llm(model_path: Path, n_ctx: int = DEFAULT_LLM_N_CTX) -> Any:
    global _CACHED_LLM
    if Llama is None:
        log("[error] llama-cpp-python not installed or failed to import. Install via: pip install llama-cpp-python")
        sys.exit(11)
    if not model_path.exists():
        log(f"[error] Missing LLM model file at: {model_path}")
        log("        Place a compatible .gguf model at that path, e.g., models/model.gguf")
        sys.exit(12)

    # Reuse the loaded model across requests (important for FastAPI/web UI).
    # This keeps the first call slow (initial load) but makes subsequent calls much faster.
    with _CACHED_LLM_LOCK:
        if _CACHED_LLM is not None:
            cached_path, cached_ctx, cached_batch, cached_threads, cached_llm = _CACHED_LLM
            if (
                cached_llm is not None
                and cached_path == model_path.resolve()
                and cached_ctx == int(n_ctx)
                and cached_batch == int(DEFAULT_LLM_N_BATCH)
                and cached_threads == (os.cpu_count() or 4)
            ):
                return cached_llm

    log(f"[generate] Loading llama.cpp model: {model_path} (CPU-only) ...")
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=os.cpu_count() or 4,
            n_batch=DEFAULT_LLM_N_BATCH,
            n_gpu_layers=0,  # CPU-only
            verbose=False,
        )
        with _CACHED_LLM_LOCK:
            _CACHED_LLM = (
                model_path.resolve(),
                int(n_ctx),
                int(DEFAULT_LLM_N_BATCH),
                (os.cpu_count() or 4),
                llm,
            )
        return llm
    except Exception as e:
        log(f"[error] Failed to load llama model: {e}")
        sys.exit(13)


_CACHED_LLM_LOCK = threading.Lock()
_CACHED_LLM: Optional[tuple[Path, int, int, int, Any]] = None


def build_prompt(question: str, contexts: List[Passage]) -> str:
    context_blocks = []
    for p in contexts:
        context_blocks.append(f"[{p.source_id}]:\n{p.text}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no context)"

    instructions = (
        "You are a helpful assistant. Use ONLY the provided context passages to answer the question. "
        "Cite the sources inline using [source_id] immediately after the statements they support. "
        "If the answer is not in the context, say you don't know. Be concise.\n\n"
    )
    prompt = (
        f"{instructions}CONTEXT PASSAGES:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:"
    )
    return prompt


def generate_answer(
    question: str,
    contexts: List[Passage],
    model_path: Path,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    stream_to_stdout: bool = False,
) -> str:
    """Generate an answer from the LLM.

    max_tokens is kept modest by default so CPU-only inference
    returns in a reasonable time on typical machines.

    When stream_to_stdout is True, tokens are printed to stdout
    as they are generated (useful for CLI); the full text is still
    returned for callers that need it.
    """
    llm = _load_llm(model_path, n_ctx=DEFAULT_LLM_N_CTX)
    prompt = build_prompt(question, contexts)

    effective_max_tokens = max(16, int(max_tokens) if max_tokens else 256)

    if stream_to_stdout:
        log("[generate] Generating answer (streaming) ...")
        print("\n===== ANSWER =====\n", end="", flush=True)
        pieces: List[str] = []
        try:
            stream = llm.create_completion(
                prompt=prompt,
                max_tokens=effective_max_tokens,
                temperature=0.2,
                top_p=0.9,
                stop=["\n\n"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("text", "")
                if not delta:
                    continue
                pieces.append(delta)
                print(delta, end="", flush=True)
        except Exception as e:
            log(f"[error] Generation failed: {e}")
            sys.exit(14)
        text = "".join(pieces).strip()
    else:
        log("[generate] Generating answer ...")
        try:
            out = llm.create_completion(
                prompt=prompt,
                max_tokens=effective_max_tokens,
                temperature=0.2,
                top_p=0.9,
                stop=["\n\n"],  # stop at a blank line to keep it tight
            )
            text = out["choices"][0]["text"].strip()
        except Exception as e:
            log(f"[error] Generation failed: {e}")
            sys.exit(14)

    # Ensure citations are visible even if the model forgets inline
    unique_sources = []
    for p in contexts:
        if p.source_id not in unique_sources:
            unique_sources.append(p.source_id)
    if unique_sources:
        sources_line = "\n\nSources: " + " ".join(f"[{sid}]" for sid in unique_sources)
        text += sources_line
        if stream_to_stdout:
            print(sources_line, end="", flush=True)

    return text


# --- CLI ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG on Windows (Mongo + FAISS + SentenceTransformers + llama.cpp)")
    # Ingestion sources
    parser.add_argument("--ingest-files", action="store_true", help="Ingest supported files (txt,pdf,docx,csv,xlsx,json,xml,images) from data/docs")
    parser.add_argument("--ingest-oracle", action="store_true", help="Ingest rows from Oracle DB (requires --oracle-dsn --oracle-user --oracle-password)")
    parser.add_argument("--oracle-dsn", type=str, help="Oracle DSN e.g. host:port/service", default=None)
    parser.add_argument("--oracle-user", type=str, help="Oracle username", default=None)
    parser.add_argument("--oracle-password", type=str, help="Oracle password", default=None)
    parser.add_argument("--oracle-query", type=str, help="Oracle SELECT query returning id,title,body", default="SELECT ID, TITLE, BODY FROM DOCUMENTS")
    parser.add_argument("--ingest-mongo-source", action="store_true", help="Ingest documents from another MongoDB collection (requires --src-mongo-uri --src-mongo-db --src-mongo-coll)")
    parser.add_argument("--src-mongo-uri", type=str, default=None, help="Source Mongo URI")
    parser.add_argument("--src-mongo-db", type=str, default=None, help="Source Mongo DB name")
    parser.add_argument("--src-mongo-coll", type=str, default=None, help="Source Mongo collection name")
    parser.add_argument("--src-mongo-text-field", type=str, default="text", help="Field containing text in source docs")
    parser.add_argument("--src-mongo-id-field", type=str, default=None, help="Optional field used to build source_id")
    # Existing operations
    parser.add_argument("--build-index", action="store_true", help="Build or extend FAISS index from MongoDB passages (incremental by default)")
    parser.add_argument("--no-incremental", action="store_true", help="Force full rebuild of FAISS index instead of incremental append")
    parser.add_argument("--query", type=str, default=None, help="Ask a question (retrieval + generation)")
    parser.add_argument("--k", type=int, default=5, help="Top-K passages to retrieve")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to local .gguf model for llama.cpp")
    parser.add_argument("--embed-model-path", type=str, default=str(EMBED_MODEL_PATH), help="Path to local .gguf used for embeddings (llama.cpp)")
    parser.add_argument("--retrieve-only", type=str, default=None, help="Retrieve top-K passages for a question and print them (no generation)")
    parser.add_argument("--agent", type=str, default=None, help="Run a small agent over tools (search_docs, get_raw_from_db, write_note_to_db) for a high-level goal")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate for the answer (smaller is faster)")

    args = parser.parse_args()

    # Basic banner
    log("========================================")
    log(" Local RAG (Windows, CPU-only)")
    log(" - MongoDB + FAISS + SentenceTransformers + llama.cpp")
    log("========================================")

    if not any([
        args.ingest_files, args.ingest_oracle, args.ingest_mongo_source,
        args.build_index, args.query, args.retrieve_only, args.agent
    ]):
        parser.print_help()
        return
    # Ingest files
    if args.ingest_files:
        ingest_files()
    # Ingest from Oracle
    if args.ingest_oracle:
        if not all([args.oracle_dsn, args.oracle_user, args.oracle_password]):
            log("[oracle] Missing required credentials: --oracle-dsn --oracle-user --oracle-password")
        else:
            ingest_oracle(args.oracle_dsn, args.oracle_user, args.oracle_password, args.oracle_query)
    # Ingest from external Mongo source
    if args.ingest_mongo_source:
        if not all([args.src_mongo_uri, args.src_mongo_db, args.src_mongo_coll]):
            log("[mongo-src] Missing required: --src-mongo-uri --src-mongo-db --src-mongo-coll")
        else:
            ingest_mongo_source(
                uri=args.src_mongo_uri,
                db_name=args.src_mongo_db,
                collection=args.src_mongo_coll,
                text_field=args.src_mongo_text_field,
                id_field=args.src_mongo_id_field,
            )

    if args.build_index:
        # Update embedding model path if provided (avoid Python 'global' redeclare issue)
        new_embed_path = Path(args.embed_model_path).resolve()
        import importlib
        m = importlib.import_module(__name__)
        setattr(m, "EMBED_MODEL_PATH", new_embed_path)
        build_index(incremental=not args.no_incremental)

    if args.retrieve_only is not None:
        question = args.retrieve_only.strip()
        if not question:
            log("[retrieve-only] Empty question string.")
            return
        log(f"[retrieve-only] Question: {question}")
        contexts = retrieve(question, top_k=max(1, args.k))
        if not contexts:
            log("[retrieve-only] No context retrieved.")
            return
        log("\n===== RETRIEVED PASSAGES =====")
        for p in contexts:
            preview = (p.text[:400] + "...") if len(p.text) > 400 else p.text
            print(f"[{p.source_id}]\n{preview}\n")

    if args.agent is not None:
        goal = args.agent.strip()
        if not goal:
            log("[agent] Empty goal string.")
            return

        import agent as agent_module

        log(f"[agent] Running agent for goal: {goal}")
        model_path = Path(args.model_path).resolve()
        answer = agent_module.run_agent(goal=goal, model_path=model_path)
        log("\n===== AGENT ANSWER =====")
        print(answer)

    if args.query is not None:
        question = args.query.strip()
        if not question:
            log("[query] Empty question string.")
            return

        log(f"[query] Question: {question}")
        contexts = retrieve(question, top_k=max(1, args.k))
        if not contexts:
            log("[query] No context retrieved; cannot generate a helpful answer.")
            return

        model_path = Path(args.model_path).resolve()
        # Stream tokens directly to stdout for CLI usage
        generate_answer(question, contexts, model_path, max_tokens=args.max_tokens, stream_to_stdout=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[info] Interrupted by user.")
        sys.exit(130)
