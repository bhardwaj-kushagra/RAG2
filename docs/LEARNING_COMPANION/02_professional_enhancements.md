# 02 – Professional Enhancements & Operational Practices

This document explains the features added beyond a bare-minimum RAG prototype—why they matter, how to use them, and what problems they solve. Each item includes: Purpose, Implementation, Usage, Extension Ideas.

---
## 1. Health Check Endpoint (`GET /health`)
**Purpose:** Quick operational confidence. Confirms API reachable and MongoDB responsive.
**Implementation:** In `api.py` returns a JSON object with `passages_count` (collection size).
**Usage:**
```bash
curl http://localhost:8000/health
```
**Extension Ideas:** Add FAISS index status, embedding model name, model file existence, uptime timestamp.

---
## 2. Retrieval vs Generation Separation
**Purpose:** Allows fast validation of semantic search without waiting for LLM generation.
**Implementation:** Distinct `/retrieve` and `/query` endpoints; UI tab “Retrieve Only”.
**Usage:**
```bash
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "top_k": 5}'
```
**Extension:** Include similarity scores; filter by source type.

---
## 3. File Upload (Multipart Ingestion)
**Purpose:** Real-world ingestion from user interface.
**Implementation:** FastAPI `UploadFile` list; React `FormData` with field name `files`.
**Usage:**
```bash
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@local/path/file1.pdf" -F "files=@local/path/file2.txt"
```
**Extension:** Progress indication, server-side validation, duplicate detection.

---
## 4. Pluggable Extractor Registry
**Purpose:** Clean separation of per-format logic; easy to add new types.
**Implementation:** `EXTRACTORS = {'.pdf': extract_pdf, ...}`; ingestion loops through selected files.
**Usage:** Add new extractor function; register key in dict.
**Extension:** Auto-detect filetype by magic bytes; asynchronous ingestion pipeline.

---
## 5. OCR Integration (Tesseract)
**Purpose:** Unlock text from images/screenshots for RAG.
**Implementation:** Auto-detect Windows install path; fallback if missing.
**Usage:** Provide `.png` / `.jpg` images; ingestion attempts OCR.
**Extension:** Confidence threshold filtering; bounding-box metadata for region linking.

---
## 6. Oracle & External MongoDB Ingestion
**Purpose:** Demonstrate multi-source ingestion (structured data → unstructured knowledge base).
**Implementation:** Dedicated functions `ingest_oracle()`, `ingest_mongo_source()` converting rows/documents to text chunks.
**Usage:** CLI flags or future API/React expansion.
**Extension:** Field-level filtering; schema introspection; incremental updates.

---
## 7. Index Persistence (FAISS + ID Map)
**Purpose:** Avoid rebuilding embeddings every run; map FAISS vector IDs to MongoDB passage IDs.
**Implementation:** `faiss.index` + `id_map.json`; loaded before retrieval if available.
**Usage:**
```bash
python src/rag_windows.py --build-index
```
**Extension:** Store embedding model version; support incremental indexing (add-only updates).

---
## 8. Normalized Embeddings for Cosine Similarity
**Purpose:** Correct semantic ranking with FAISS IndexFlatIP which uses inner product.
**Implementation:** `emb /= np.linalg.norm(emb)` per vector.
**Usage:** Transparent to user.
**Extension:** Switch to `IndexFlatL2` + raw vectors; evaluate performance trade-offs.

---
## 9. CLI Script Flags
**Purpose:** Reproducibility & automation.
**Implementation:** Argparse flags: `--ingest-files`, `--build-index`, `--query`, `--retrieve-only`, `--k`.
**Usage:**
```bash
python src/rag_windows.py --query "What is ML?" --k 3
```
**Extension:** Add `--chunk-size`, `--chunk-overlap`, `--embedding-model`.

---
## 10. React UI Professional Styling
**Purpose:** Presentation quality for demos; communicates state clearly.
**Implementation:** Modular CSS: spinners, status dots, tabs, gradient backgrounds.
**Usage:** Run `npm start` inside `web/`.
**Extension:** Dark mode toggle; accessibility review (ARIA labels).

---
## 11. Health & Status Indicators in UI
**Purpose:** Transparent system readiness for non-technical stakeholders.
**Implementation:** Poll `/health` on mount; update color-coded dot.
**Usage:** Automatic; manual refresh button.
**Extension:** Show index size & last update timestamp.

---
## 12. Setup & Start Scripts (`setup_web.ps1`, `start_web.ps1`)
**Purpose:** Reduce onboarding friction; automate multi-process startup.
**Implementation:** PowerShell checks for Node, MongoDB, Python packages; starts API + React in background jobs.
**Usage:**
```powershell
./setup_web.ps1
./start_web.ps1
```
**Extension:** Add cleanup, environment variable toggles, port overrides.

---
## 13. Testing & Debugging Techniques
| Technique | Purpose | Example |
|-----------|---------|---------|
| Health ping | Check basic availability | `curl /health` |
| Retrieval-only | Validate semantic search | `/retrieve` endpoint |
| Port inspection | Resolve binding conflicts | `netstat -ano | findstr :8000` |
| Process kill | Clear stale servers | `taskkill /PID <pid> /F` |
| Sample datasets | Controlled ingestion targets | Synthetic PDF/DOCX/CSV |
| Logging (stdout) | Monitor startup phases | Uvicorn boot messages |

**Extension:** Structured logging (JSON), tracing request latency, integration tests.

---
## 14. Error Resilience Patterns
| Pattern | Description | Benefit |
|---------|-------------|---------|
| Soft-fail optional imports | Try/except around format libs | Ingest unaffected by missing extras |
| CORS middleware early | Config before endpoints | Avoid preflight failures |
| Explicit ID mapping | Maintain source traceability | Reliable citation rendering |
| Separate retrieval/generation | Decouple latency-critical steps | Faster validation cycles |

---
## 15. curl Usage Cheatsheet
```bash
# Health
curl http://localhost:8000/health

# Retrieve only
curl -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"query":"machine learning","top_k":3}'

# Query with generation
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"What is deep learning?","top_k":2}'

# Build index
curl -X POST http://localhost:8000/build-index -H 'Content-Type: application/json' \
  -d '{"embedding_model":"all-MiniLM-L6-v2"}'

# Upload files
curl -X POST http://localhost:8000/ingest/files \
  -F 'files=@docs/sample.txt' -F 'files=@docs/test_ai.pdf'
```

---
## 16. Professional Practices Summary
| Practice | Value |
|----------|-------|
| Structured endpoints | Predictable integration surface |
| Validation models (Pydantic) | Early error detection |
| Environment scripts | Lower setup cognitive load |
| Separation of concerns | Easier extension, better testability |
| Persistent index artifacts | Faster warm restarts |
| CORS configuration | Front-end compatibility |

---
## 17. Future Enhancements (Professional Grade)
- Observability: metrics (Prometheus), structured logs.
- Authentication: JWT on protected endpoints.
- Reranking: cross-encoder for improved relevance.
- Streaming responses: chunked transfer for token streaming.
- Versioning: track embedding model & chunk params in metadata.

---
Use these practices as a checklist when you rebuild the project manually.
