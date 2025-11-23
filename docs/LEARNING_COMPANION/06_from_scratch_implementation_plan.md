# 06 – From-Scratch Implementation Plan

This is a structured roadmap for you to re-build the entire project solo. Follow stages; each stage ends with a “Checkpoint” verifying progress. Add your own notes as you go.

---
## Stage 0 – Prerequisites & Mindset
**Goal:** Prepare environment & learning objectives.
**Actions:**
- Install Python 3.11, Node.js LTS, MongoDB Community.
- Create empty Git repository.
- Define success: ingest → index → retrieve → generate.
**Checkpoint:** MongoDB `mongod` service running; empty repo initialized.

---
## Stage 1 – Basic Project Skeleton
**Goal:** Lay out directories.
**Actions:**
```
mkdir src models data\docs docs
copy NUL data\docs\.gitkeep
copy NUL models\.gitkeep
```
Create `README.md` with initial goals.
**Checkpoint:** Tree matches skeleton; README present.

---
## Stage 2 – Minimal Text Ingestion
**Goal:** Ingest `.txt` files into MongoDB.
**Actions:**
- Install: `pymongo`, `sentence-transformers`, `faiss-cpu`, `numpy`, `tqdm`.
- Write `src/rag_windows.py` with function: load files → chunk → insert.
- Chunker: 250 words, 50 overlap.
**Checkpoint:** After run, MongoDB `rag_db.passages` contains documents with `text`, `source_id`.

---
## Stage 3 – Embedding & Index Build
**Goal:** Create FAISS index.
**Actions:**
- Load all passages from Mongo.
- Embed with `SentenceTransformer('all-MiniLM-L6-v2')`.
- Normalize vectors.
- Build `IndexFlatIP` and save `faiss.index` + `id_map.json`.
**Checkpoint:** File `faiss.index` exists; search returns IDs.

---
## Stage 4 – Retrieval Function
**Goal:** Query top-K passages.
**Actions:**
- Add CLI flag `--retrieve-only --query "..." --k 3`.
- Embed query, normalize, search FAISS, map IDs to passages.
**Checkpoint:** Terminal prints top passages; texts match expectation.

---
## Stage 5 – Local LLM Generation
**Goal:** Add answer synthesis.
**Actions:**
- Install `llama-cpp-python==0.3.2`.
- Download TinyLlama GGUF model.
- Add `--query` path: retrieval + prompt assembly + llama generation.
**Checkpoint:** Answer displays + appended sources list.

---
## Stage 6 – Multi-Format Ingestion
**Goal:** Support PDF, DOCX, CSV, XLSX, JSON, XML, images (OCR).
**Actions:**
- Install optional libs: `PyPDF2 python-docx pandas pytesseract pdf2image pillow`.
- Implement extractor registry.
- Soft-fail missing dependencies.
**Checkpoint:** Each sample file ingests; passage count increases.

---
## Stage 7 – Database Source Ingestion
**Goal:** Oracle & external Mongo.
**Actions:**
- Install `oracledb`.
- Implement functions converting rows/docs to flat text chunks.
**Checkpoint:** Rows/documents appear as passages with `source_type` metadata.

---
## Stage 8 – API Layer
**Goal:** Wrap core functions in FastAPI.
**Actions:**
- Create `api.py` endpoints for ingest, build-index, retrieve, query, health.
- Use Pydantic models for validation.
**Checkpoint:** Swagger UI shows all endpoints; test health & retrieval.

---
## Stage 9 – React Web UI
**Goal:** Build presentational interface.
**Actions:**
- Create `web/` with CRA or manual config.
- Tabs: Ingest Files, Build Index, Query RAG, Retrieve Only.
- Status indicator uses `/health`.
**Checkpoint:** UI loads, interacts with API, displays results.

---
## Stage 10 – Scripts & Automation
**Goal:** Improve developer ergonomics.
**Actions:**
- Add `setup_web.ps1` (env checks + npm install).
- Add `start_web.ps1` (launch API + React concurrently).
**Checkpoint:** Single command launches both servers.

---
## Stage 11 – Documentation Expansion
**Goal:** Teach & explain internals.
**Actions:**
- Write guides: PROJECT, INGESTION, API, WEB UI.
- Add learning companion docs (this folder).
**Checkpoint:** All docs present; README links them.

---
## Stage 12 – Testing & Validation
**Goal:** Confidence in pipeline.
**Actions:**
- Create sample multi-format dataset.
- Run ingestion → build index → retrieve → query.
- Document results in `TEST_RESULTS.md`.
**Checkpoint:** Document shows successful end-to-end run.

---
## Stage 13 – Refactor & Hardening (Optional)
**Goal:** Improve structure.
**Actions:**
- Break `rag_windows.py` into modules.
- Add config file `config.yaml`.
- Introduce logging abstraction.
**Checkpoint:** Cleaner imports; easier testing.

---
## Stage 14 – Advanced Enhancements (Optional)
**Goal:** Push beyond baseline.
**Actions:**
- Add reranking (cross-encoder).
- Implement streaming token generation.
- Add authentication.
- Introduce metrics (request latency, generation time).
**Checkpoint:** Production-like observability & UX improvements.

---
## Time Allocation Guidance
| Stage | Estimated Time |
|-------|----------------|
| 0–4 (MVP Retrieval) | 3–5 hours |
| 5 (Generation) | 1–2 hours |
| 6–7 (Formats & DB) | 3–4 hours |
| 8–9 (API + UI) | 4–6 hours |
| 10–12 (DX + Docs + Testing) | 3–4 hours |
| 13–14 (Refactor & Advanced) | Variable |

---
## Self-Assessment Checklist
| Item | Status |
|------|--------|
| Can explain cosine similarity normalization |  |
| Can modify chunk size & test impact |  |
| Can add new extractor format |  |
| Can replace embedding model safely |  |
| Can rebuild index after ingestion |  |
| Can trace answer citations to source passages |  |
| Can run full system from scratch |  |
| Can debug port binding issue |  |

---
## Common Pitfalls & Avoidance
| Pitfall | Avoidance |
|---------|-----------|
| Rebuilding index without clearing old artifacts | Validate/backup before overwrite |
| Mixing embedding models (passages vs query) | Centralize model initialization |
| Oversized chunks reducing recall | Empirically test retrieval quality |
| Missing conditional dependency causing crash | Wrap imports in try/except |
| Letting scripts accumulate hard-coded values | Externalize to config |

---
## Reflect & Iterate
After completing all stages, write a short retrospective:
- What took longest?
- Which concepts were hardest?
- What would you refactor first?
- How would you scale to 1M passages?

---
Rebuilding with this roadmap will transform passive review into active mastery.
