# 07 – Additional Crucial Topics & Missing Pieces

This document fills remaining gaps so your understanding is truly end-to-end.

---
## 1. Configuration & Parameterization (Not Yet Implemented)
**Current State:** Hard-coded values (chunk size, overlap, embedding model, k).
**Why Important:** Enables experimentation without code edits.
**Suggested File:** `config.yaml`
```yaml
chunk_size: 250
chunk_overlap: 50
embedding_model: all-MiniLM-L6-v2
default_k: 3
model_path: models/model.gguf
index_path: faiss.index
id_map_path: id_map.json
```
**Loading Pattern:**
```python
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
```

---
## 2. Logging & Observability
**Current:** Print statements / default server logs.
**Goal:** Structured logs + metrics.
**Example:**
```python
import logging, time
logger = logging.getLogger("rag")
logger.info({"event": "index_build_start", "count": passage_count})
```
**Metrics:** Generation latency, retrieval size, ingestion throughput.

---
## 3. Error Taxonomy
Categorize errors for easier debugging:
| Category | Example | Mitigation |
|----------|---------|-----------|
| Dependency Missing | OCR library not installed | Soft-fail skip |
| Resource Access | Mongo connection refused | Retry/backoff; check service |
| Model Loading | File not found | Validate path early; user instructions |
| Index Missing | Query before build | Return instructive message |
| Port Conflict | 8000 in use | Detect & suggest PID kill |

---
## 4. Security Foundations (Even for Local)
| Vector | Consideration |
|--------|---------------|
| File Upload | Validate extension, size limit |
| Query Injection | Not severe for pure retrieval; sanitize DB queries |
| Environment Secrets | For DB creds (future) use `.env` ignored by Git |
| Denial-of-Service | Rate limit generation endpoint |

---
## 5. Performance Profiling Starter
Use `time.perf_counter()` wrappers:
```python
start = time.perf_counter()
# embed passages
elapsed = time.perf_counter() - start
print(f"Embedding phase took {elapsed:.2f}s")
```
Expand with `cProfile` for deeper CPU hotspots.

---
## 6. Memory Considerations
| Artifact | Approx Size (Demo) | Notes |
|----------|--------------------|-------|
| TinyLlama Q4_K_M | ~0.5–0.7 GB | In RAM while generating |
| FAISS Flat Index | ~(#vectors * dim * 4 bytes) | MiniLM dim=384 |
| Embedding Model | ~90–100 MB | HF cache outside repo |
**Tip:** Monitor with Task Manager during generation.

---
## 7. Prompt Engineering Improvements
Current prompt: simple concatenation of contexts + question.
Enhance with:
```
You are a domain assistant. Use ONLY the provided sources.
If unsure, say "I don't know".

Sources:
[1] {passage_1}
[2] {passage_2}
...
Question: {user_question}
Answer (cite sources as [n]):
```
**Benefit:** Reduced hallucination; clearer citation mapping.

---
## 8. Evaluation & Quality Metrics
| Metric | Method |
|--------|-------|
| Retrieval Recall@K | Manual or relevance judgments |
| Passage Diversity | Unique source count per answer |
| Answer Citation Coverage | % of cited sources actually used |
| Generation Latency | Timed measurement |

---
## 9. Incremental Indexing (Future)
**Current:** Full rebuild required.
**Strategy:** Track last ingested `_id`; only embed new passages -> `index.add(new_vectors)`.
**Edge Case:** Removing passages -> need rebuild or maintain deletion tombstones.

---
## 10. Hybrid Retrieval Expansion
Combine keyword + semantic:
1. Text search (e.g., using MongoDB text index or BM25 library).
2. Dense retrieval via FAISS.
3. Merge + rerank results.
**Benefit:** Better recall on rare entity names + semantic coverage.

---
## 11. Reranking Implementation Sketch
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# pairs: (query, passage_text)
scores = model.predict(pairs)
# sort passages by score descending
```
**When:** After initial FAISS top-K (e.g., take 20 → rerank → choose final 5).

---
## 12. Streaming Generation (UX Improvement)
Utilize llama.cpp token callback to push partial text to client via WebSocket.
**Benefit:** Perceived latency reduction; interactive experience.

---
## 13. Authentication Basics
Add dependency:
```
pip install python-jose passlib[bcrypt]
```
Implement OAuth2 password flow or simple API key header check.

---
## 14. Deployment Considerations
| Aspect | Local | Production |
|--------|-------|------------|
| API Server | Uvicorn single worker | Gunicorn/Uvicorn multiple workers |
| Static UI | CRA dev server | Build + reverse proxy (Nginx) |
| DB | Local Mongo | Managed cluster (MongoDB Atlas) |
| Model Storage | Local file | Object store (S3) |
| Logs | Stdout | Centralized collector (ELK/EFK) |

---
## 15. Resilience Enhancements
- Add retry logic to DB connections.
- Wrap generation in timeout; return partial answer gracefully.
- Use circuit breaker pattern if model repeatedly fails.

---
## 16. Git Workflow Suggestions
| Practice | Benefit |
|----------|---------|
| Feature branches | Isolate changes |
| Conventional commits | Readable history |
| Pre-commit hooks | Auto-format & lint |
| Pull request templates | Consistent review context |

---
## 17. Linting & Formatting
Add tools:
```
pip install black isort flake8 mypy
```
Run:
```
black src/
flake8 src/
mypy src/
```
**Benefit:** Early error detection & consistent style.

---
## 18. Dependency Update Policy
- Review monthly.
- Pin major versions; allow minor updates.
- Test retrieval correctness after embedding model changes.

---
## 19. Data Privacy Considerations
If using sensitive docs:
- Strip PII before ingestion.
- Encrypt MongoDB backups.
- Restrict model output logging.

---
## 20. Checklist Before Calling It "Production"
| Item | Completed? |
|------|------------|
| Auth & rate limiting |  |
| Structured logging |  |
| Monitoring & metrics |  |
| CI pipeline (tests) |  |
| Automated backups |  |
| Config externalization |  |
| Security scan (deps) |  |
| Load test retrieval |  |

---
These topics round out the foundation so you can reason about reliability, scalability, and professional deployment standards.
