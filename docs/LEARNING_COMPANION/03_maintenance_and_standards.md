# 03 – Maintenance, Standards & Repository Hygiene

This document explains non-functional artifacts that keep the project maintainable: ignore rules, attributes, scripts, documentation structure, dependency decisions, naming conventions.

---
## 1. Directory Structure Rationale
```
project_root/
  src/                 # Executable Python source (RAG logic + API)
  data/docs/           # Raw ingestion source documents
  models/              # Local GGUF model(s)
  web/                 # React front-end
  docs/                # Formal documentation & guides
  docs/LEARNING_COMPANION/ # Deep learning-focused docs (you requested)
  faiss.index          # Serialized FAISS index
  id_map.json          # Mapping FAISS IDs -> Mongo ObjectIds
  setup_web.ps1        # One-time environment/setup tasks
  start_web.ps1        # Convenience multi-server launcher
  TEST_RESULTS.md      # Snapshot of validation run
```
**Principles:**
- *Separation of concerns*: backend vs frontend vs knowledge artifacts.
- *Predictable locations*: indexes & maps at root for quick existence checks.
- *Docs parity*: both high-level and deep educational resources.

---
## 2. `.gitignore`
**Purpose:** Prevent committing transient, large, OS-specific, cache or environment files.
**Key Entries:**
- Python caches: `__pycache__/`, `*.py[cod]`
- Virtual env: `.venv/`
- Model/data: `models/` except `.gitkeep`
- Generated index: `faiss.index`, `id_map.json`
- Node artifacts: `web/node_modules/`, `web/build/`
- Logs: `*.log`, `logs/`
**Why it matters:** Keeps repository lean; avoids accidentally committing sensitive or large binaries.
**Extend:** Add `coverage/`, `*.sqlite3`, `.env` for secrets.

---
## 3. `.gitattributes`
**Purpose:** Normalize line endings & declare binary types.
**Typical Entries:**
- `* text=auto` ensures consistent line endings across platforms.
- Binary patterns: `*.gguf`, `*.bin`, `*.index` marked to prevent diff heuristics.
**Why:** Avoid merge conflicts from CRLF/LF; treat large binary models as opaque.
**Extend:** Add linguist overrides for documentation vs code metrics if using GitHub linguist.

---
## 4. Requirements Handling
**Strategy:** Minimal core dependencies installed manually; optional extras (PDF, DOCX, OCR) imported conditionally.
**Reasoning:** New learners can start with small footprint; advanced formats unlock progressively.
**Suggested `requirements.txt` (if you generate):**
```
sentence-transformers
pymongo
faiss-cpu
llama-cpp-python==0.3.2
numpy
tqdm
fastapi
uvicorn
python-multipart
PyPDF2
python-docx
pandas
pytesseract
pdf2image
oracledb
pillow
```
**Improvement:** Pin versions to ensure reproducibility; add `hash=` lines (pip hash-checking) for supply-chain hardening.

---
## 5. Scripts (`setup_web.ps1`, `start_web.ps1`)
| Script | Role | Idempotency | Key Checks |
|--------|------|-------------|------------|
| setup_web.ps1 | Install Node deps & verify environment | Yes | Node present, Mongo ping, Python imports |
| start_web.ps1 | Launch API + React concurrently | Yes | Ensures prerequisites installed |
**Maintenance Tip:** Use comments for each block; keep user messages concise & actionable.
**Extend:** Parameterize ports: `-ApiPort 8000 -WebPort 3000`.

---
## 6. Documentation Strategy
| Doc | Audience | Purpose |
|-----|----------|---------|
| README.md | New developer | Quick start overview |
| PROJECT_GUIDE.md | Learner / engineer | Deep architecture & pipeline internals |
| INGESTION_MODULE.md | Data engineer | Format-specific ingestion details |
| API_GUIDE.md | Integrator | REST endpoints usage |
| WEB_UI_GUIDE.md | Demo user | Operating front-end |
| OVERVIEW.md | Manager / overview | High-level system summary |
| TEST_RESULTS.md | QA / engineering | Evidence of working state |
| LEARNING_COMPANION/* | You (student) | Education-focused deep dives |
**Maintenance Tip:** Each doc should have a clear update date; include change logs as complexity grows.

---
## 7. Code Organization & Naming
**Python:**
- `rag_windows.py`: intentionally monolithic early; could refactor into modules (`ingestion.py`, `indexing.py`, `retrieval.py`, `generation.py`).
- Function names verb-focused: `ingest_files`, `build_index`, `retrieve`, `generate_answer` for clarity.
**React:**
- Single `App.js` for MVP; scale to component folders (`components/Upload`, `components/QueryPanel`).
**Tip:** Introduce type hints everywhere for maintainability (already partially present).

---
## 8. Error Handling Practices
- Soft-fail on missing optional libs (logs message, skips format).
- HTTP errors via `HTTPException(status_code=...)` in API.
- User feedback in UI (colored result boxes). 
**Extend:** Centralize exception handling middleware; map internal errors to structured JSON.

---
## 9. Configuration Management
Currently implicit (hard-coded model name, chunk sizes). For maturity:
- Create `config/settings.yaml` with keys: `chunk_size`, `chunk_overlap`, `embedding_model`, `default_k`, `model_path`.
- Load at startup with fallback defaults.
**Benefit:** Experimentation easier; fosters reproducibility.

---
## 10. Testing Approach (Current & Future)
**Current:** Manual integration tests (API + UI + sample files).
**Future:**
- Pytest fixtures for Mongo ephemeral test DB.
- Snapshot tests for retrieval ordering given fixed embeddings.
- Mock llama.cpp for faster pipeline tests.
- Front-end Cypress tests for UI flows.

---
## 11. Versioning & Metadata (Future Capability)
Add metadata fields to passages:
```json
{
  "text": "...",
  "source_id": "file.pdf#3",
  "ingested_at": "2025-11-24T10:15:00Z",
  "chunk_strategy": "words_250_overlap_50",
  "embedding_model": "all-MiniLM-L6-v2"
}
```
**Reason:** Track provenance & reproducibility.

---
## 12. Security Considerations (Roadmap)
| Concern | Risk | Mitigation |
|---------|------|------------|
| Open endpoints | Unauthorized abuse | Add auth (JWT / API keys) |
| Arbitrary file upload | Injection / large files | Restrict extensions & size checks |
| DB exposure | Sensitive data leak | Isolate DB network & enable auth |
| Model path input | Path traversal | Validate & sanitize user-supplied paths |

---
## 13. Performance & Scaling Paths
| Layer | Option | Trade-Off |
|-------|--------|-----------|
| Embeddings | Larger model (e.g., `multi-qa-MiniLM-L6-cos-v1`) | Better recall vs slower build |
| Index | IVF/HNSW | Faster retrieval vs complexity |
| Generation | GPU quantized model | Faster tokens vs hardware requirement |
| Caching | Redis for passage results | Memory usage vs latency reduction |

---
## 14. Continuous Improvement Checklist
- [ ] Introduce configuration file.
- [ ] Add automated tests (pytest). 
- [ ] Add logging abstraction.
- [ ] Implement incremental indexing.
- [ ] Add reranking.
- [ ] Add authentication.
- [ ] Add monitoring metrics.

---
## 15. Why These Standards Matter
| Standard | Impact |
|----------|--------|
| Ignore rules | Keeps repo clean & small |
| Scripts | Lowers barrier for new collaborators |
| Documentation segmentation | Tailored learning & onboarding |
| Soft dependency model | Progressive enhancement |
| Persistent artifacts | Avoid rework & reduce startup latency |

---
Use this as a maintenance lens while rebuilding to reinforce professional best practices.
