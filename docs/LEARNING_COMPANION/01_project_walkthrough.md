# 01 – Project Walkthrough (Chronological)

This document reconstructs the entire journey so you can learn how each decision, problem, and solution emerged. It is divided into phases. Each phase lists: Goals, Actions, Problems, Diagnoses, Solutions, Lessons.

---
## Phase 1: Initial Requirements & Environment
**Goal:** Build a fully local RAG pipeline (MongoDB + SentenceTransformers + FAISS + llama.cpp) on Windows (CPU-only, no Docker).

**Actions:**
- Created Python virtual environment, installed base packages: `sentence-transformers`, `pymongo`, `faiss-cpu`, `llama-cpp-python`.
- Established folder structure: `src/`, `data/docs/`, `models/`, `docs/`.

**Problems Encountered:**
1. `llama-cpp-python` latest version attempted source build (requires a C++ toolchain on Windows).
2. Missing local GGUF model.

**Diagnosis:**
- Windows wheel availability limited for newer versions; building from source is slow.
- Model file required for generation step (LLM inference cannot run without weights).

**Solutions:**
- Pinned to a prebuilt wheel version (`0.3.2`) using `--only-binary` installation.
- Downloaded TinyLlama 1.1B Q4_K_M GGUF model into `models/model.gguf`.

**Lessons:**
- Prefer prebuilt wheels for Windows to avoid toolchain friction.
- Separate model acquisition from code to keep repo lightweight.

---
## Phase 2: Basic Ingestion & Retrieval
**Goal:** End-to-end text ingestion and retrieval.

**Actions:**
- Implemented `rag_windows.py` with CLI flags.
- Added simple TXT file chunking (word-based, overlap) + MongoDB storage.
- Built FAISS index with normalized embeddings (cosine similarity via inner product on normalized vectors).

**Problems:**
1. Deciding chunk size & overlap without over-engineering.
2. Ensuring embedding normalization matches FAISS metric.

**Diagnosis:**
- Chunk size too large reduces recall; too small inflates index size.
- FAISS IndexFlatIP expects inner product; for cosine similarity we must L2-normalize vectors.

**Solutions:**
- Picked 250-word chunks with 50-word overlap (balanced context vs index size for a demo).
- Normalized embeddings using `emb / np.linalg.norm(emb)` before adding to FAISS.

**Lessons:**
- Start with simple heuristics; refine later based on retrieval quality.
- Cosine similarity = inner product of normalized vectors.

---
## Phase 3: Generation Integration
**Goal:** Retrieve passages and generate answers with local LLM.

**Actions:**
- Added `generate_answer()` using llama.cpp Python bindings.
- Prompt template included retrieved passages + user question + simple citation formatting.

**Problems:**
1. Performance slow (CPU-only inference).
2. Need to ensure citations are traceable.

**Diagnosis:**
- TinyLlama is still sizable; token generation speed limited by CPU.
- Must preserve source identifiers through indexing pipeline.

**Solutions:**
- Accepted latency (demo context); suggested smaller models for speed.
- Stored mapping `faiss_id -> Mongo _id` in `id_map.json`.

**Lessons:**
- Optimize later (quantization, model choice). Correctness first.
- Always carry metadata for explainability.

---
## Phase 4: Multi-Format Ingestion Extension
**Goal:** Support PDF, DOCX, CSV, XLSX, JSON, XML, Images (OCR), Oracle DB, external MongoDB.

**Actions:**
- Built extractor registry `EXTRACTORS` keyed by extension.
- Added graceful fallback if optional libraries missing.
- Integrated OCR with auto-detect of Tesseract path.

**Problems:**
1. Handling optional dependencies without breaking base pipeline.
2. OCR reliability & path detection on Windows.
3. Diverse file semantics (tabular vs prose vs structured JSON/XML).

**Diagnosis:**
- Import failures should not crash ingestion of other formats.
- Windows typical path: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Need uniform output: plain text chunks.

**Solutions:**
- Wrapped imports in try/except; flagged unavailable formats.
- Auto-set `pytesseract.pytesseract.tesseract_cmd` if found.
- Flattened table data to joined text; serialized JSON/XML to textual summaries.

**Lessons:**
- Pluggable architecture isolates format logic.
- Robust ingestion means soft-failing non-critical features.

---
## Phase 5: Validation & Testing Files
**Goal:** Confirm ingestion correctness across formats.

**Actions:**
- Generated sample files: `.txt`, `.pdf`, `.docx`, `.csv`, `.xlsx`, `.json`, `.xml`, `.png`.
- Ran ingestion; confirmed passage counts.

**Problems:**
1. Ensuring chunk creation consistent across mixed sources.
2. Verifying OCR extracted text quality.

**Diagnosis:**
- Different formats supply variable text lengths → chunk boundary logic must remain stable.
- OCR output may be noisy; acceptable for demo.

**Solutions:**
- Reused universal chunker after per-format text extraction.
- Accepted minor OCR noise; documented improvement paths.

**Lessons:**
- Normalize inputs early (convert all to raw text before chunking).
- Provide upgrade notes instead of premature perfection.

---
## Phase 6: REST API Layer (FastAPI)
**Goal:** Expose ingestion, indexing, retrieval, generation via HTTP.

**Actions:**
- Created `api.py` with endpoints: `/ingest/files`, `/ingest/oracle`, `/ingest/mongo`, `/build-index`, `/retrieve`, `/query`, `/health`.
- Added Pydantic models for request validation.

**Problems:**
1. CORS issues with upcoming web UI.
2. Background server start & port conflicts.

**Diagnosis:**
- Browser-based app requires CORS for `http://localhost:3000`.
- Port 8000 previously occupied caused binding error (Win ephemeral process lingering).

**Solutions:**
- Added `CORSMiddleware` allowing origins list.
- Killed orphan PID before restart.

**Lessons:**
- Always configure CORS before front-end integration.
- Diagnose port conflicts via `netstat -ano | findstr :8000`.

---
## Phase 7: Web UI (React)
**Goal:** Provide demo-friendly graphical interface.

**Actions:**
- Created `web/` with React (Create-React-App style), `App.js` tabs for workflow.
- Added API proxy + `apiService.js` wrapper.
- Styled with modern CSS (gradient, animations, loading spinner).

**Problems:**
1. Ensuring file uploads align with FastAPI's multipart expectations.
2. Startup race (API not yet ready when UI loads).

**Diagnosis:**
- FastAPI expects `files` field with multiple file parts.
- UI should show offline state until health passes.

**Solutions:**
- Implemented FormData upload in `ingestFiles`.
- Added health polling & status indicator.

**Lessons:**
- Provide immediate feedback (loading, errors) for good UX.
- Keep API client minimal & transparent.

---
## Phase 8: Developer Experience Enhancements
**Goal:** Smooth onboarding and repeatable runs.

**Actions:**
- Added `setup_web.ps1` and `start_web.ps1` scripts.
- Added comprehensive docs: `WEB_UI_GUIDE.md`, `OVERVIEW.md`, `TEST_RESULTS.md`.

**Problems:**
1. Multi-terminal manual process tedious.
2. Users unsure of dependency state.

**Diagnosis:**
- Scripts reduce friction; checks catch missing dependencies early.

**Solutions:**
- Pre-flight checks: Node.js, MongoDB ping, Python imports.
- Parallel start of API + React with logging stream.

**Lessons:**
- Automate repetitive manual environment setup.
- Good docs accelerate self-teaching.

---
## Phase 9: Final Testing & Stabilization
**Goal:** Ensure entire stack works concurrently.

**Actions:**
- Verified MongoDB, API, React reachable.
- Ran retrieval + generation endpoints.
- Documented performance (~26s generation latency CPU-only).

**Problems:**
- Intermittent port binding & early termination (KeyboardInterrupt).

**Solutions:**
- Proper background start (Start-Process / Start-Job) & wait times.

**Lessons:**
- Distinguish transient startup issues from real code faults.

---
## Key Trade-Offs & Rationale
| Decision | Alternative | Reason |
|----------|------------|--------|
| Word-based chunking | Semantic splitting | Faster to implement; enough for demo |
| IndexFlatIP + cosine via normalization | HNSW / IVF | Simplicity; small data scale |
| TinyLlama Q4_K_M | Larger 7B model | CPU constraints; speed vs capacity |
| Optional deps soft-fail | Hard fail | Resilience; partial ingestion still useful |
| React separate dev server | Static serving via FastAPI | Faster iteration during learning phase |

---
## Lessons Summary
1. Start minimal → expand with confidence.
2. Normalize representations early (text + embeddings).
3. Metadata preservation critical for citations.
4. Scripts & docs multiply approachability.
5. Expect environment-specific friction on Windows (ports, builds, paths).

---
## Suggested Next Improvements (For Learning)
- Implement semantic chunking (e.g., via sentence tokenization + dynamic grouping).
- Add cross-encoder reranking for improved passage ordering.
- Introduce streaming generation tokens in UI.
- Add auth layer (JWT) to protect endpoints.
- Implement passage metadata filters (source type, date).

---
Rebuild this path step-by-step to reinforce learning using `06_from_scratch_implementation_plan.md`.
