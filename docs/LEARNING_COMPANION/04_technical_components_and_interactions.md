# 04 – Technical Components & Interactions

This document explains every core component, its role, internal data flows, and how they collaborate inside the RAG architecture.

---
## 1. High-Level Architecture
```
┌───────────────┐      Upload / Query      ┌───────────────┐
│   React UI     │  ─────────────────────>  │    FastAPI     │
│ (Presentation) │                          │  (Service Layer)│
└───────┬────────┘                          └───────┬─────────┘
        │                                         │
        │ REST / JSON                             │ Function calls
        ▼                                         ▼
┌───────────────┐      Text + Metadata      ┌──────────────┐
│   Ingestion    │  ─────────────────────>  │   MongoDB     │
│  (Extractors)  │                          │ (Passages Coll)│
└────────┬──────┘                          └──────┬─────────┘
         │                                        │
         │ Embeddings build                       │ Passage texts
         ▼                                        │
┌───────────────┐  Add vectors / IDs  ┌───────────────┐
│ SentenceTransf │ ─────────────────> │     FAISS      │
│ (Embeddings)   │ <───────────────── │ (Vector Index) │
└───────────────┘  Retrieve top-K     └───────┬────────┘
                                              │ IDs
                                              ▼
                                      ┌──────────────┐
                                      │  llama.cpp    │
                                      │ (LLM answer)  │
                                      └──────────────┘
```

---
## 2. Component Roles
| Component | Role | Why Chosen |
|-----------|------|------------|
| React UI | Interaction layer & visualization | Improves demo usability |
| FastAPI | HTTP API surface | Async capable, easy docs (Swagger) |
| Ingestion Extractors | Convert heterogeneous sources to plain text chunks | Unifies retrieval corpus |
| MongoDB | Persistent storage for passages & metadata | Flexible schema, easy local setup |
| SentenceTransformers | Dense semantic embedding generation | High-quality sentence-level representations |
| FAISS | Efficient similarity search over vectors | Industry-standard library for ANN search |
| llama.cpp | Local LLM inference | CPU-friendly, runs quantized models |
| Tesseract OCR | Extract text from images | Unlock image-based docs |

---
## 3. Data Model (Passage Document)
Minimal base shape:
```json
{
  "_id": ObjectId("..."),
  "source_id": "test_ai.pdf#2",
  "text": "Neural networks are ...",
  "chunk_index": 2,
  "source_type": "pdf",
  "created_at": ISODate("2025-11-24T10:12:50Z")
}
```
**Rationale:**
- `source_id`: stable for citation.
- `chunk_index`: debugging chunk splits.
- Extendable for additional metadata (embedding version, ingestion strategy).

---
## 4. Ingestion Pipeline Flow
1. Enumerate input sources (files, DB rows, images).
2. For each source: detect extension, call matching extractor.
3. Extract plain text; clean & normalize whitespace.
4. Chunk into overlapping segments → list of strings.
5. Insert each chunk into MongoDB with metadata.

**Key Principles:**
- Extraction isolated from chunking (single responsibility).
- Chunking uniform across formats → consistent embedding shape.

---
## 5. Embedding & Indexing Flow
1. Query MongoDB for all passages.
2. Embed each passage text using `SentenceTransformer`.
3. Normalize vector (L2 divide) for cosine using inner product.
4. Add vector to FAISS `IndexFlatIP`.
5. Record mapping: FAISS internal row -> MongoDB `_id`.
6. Persist index + mapping files.

**Why Normalize:** Enables using inner product to approximate cosine similarity.

---
## 6. Retrieval Flow
1. Embed incoming query (same model, normalization).
2. `index.search(query_vec, k)` returns top-K FAISS IDs.
3. Map FAISS IDs back to MongoDB `_id` via `id_map.json`.
4. Fetch passage texts & metadata.
5. Return structured list to user (or generation pipeline).

**Potential Enhancements:**
- Add scoring to responses.
- Rerank using cross-encoder.
- Filter by metadata before retrieval.

---
## 7. Generation Flow (RAG)
1. Retrieve top-K passages.
2. Construct prompt: instructions + concatenated context + user question.
3. llama.cpp model generates tokens until stop criteria.
4. Post-process answer: attach citations + sources list.
5. Return answer + retrieved passages.

**Prompt Considerations:**
- Keep context concise; too long reduces model quality.
- Include explicit instruction to cite sources.

---
## 8. Interaction Patterns
| Interaction | Source | Destination | Protocol |
|-------------|--------|-------------|----------|
| File Upload | React | FastAPI `/ingest/files` | HTTP multipart |
| Health Check | React | FastAPI `/health` | HTTP GET |
| Query | React | FastAPI `/query` | HTTP JSON POST |
| Retrieve | React | FastAPI `/retrieve` | HTTP JSON POST |
| Index Build | React/CLI | FastAPI or local script | HTTP / direct call |
| Generation | FastAPI | llama.cpp binding | Python API |

---
## 9. Performance Characteristics
| Step | Complexity | Dominant Cost |
|------|------------|---------------|
| Ingestion extraction | O(n sources) | Disk I/O, parsing libs |
| Chunking | O(n text length) | String splitting |
| Embedding build | O(n passages * model cost) | Transformer forward passes |
| Index search | O(k log n) (approx) | FAISS linear scan (Flat) |
| Generation | O(tokens) | LLM token loop (CPU) |

**Observation:** Generation is slowest on CPU. Retrieval comparatively instant for small corpora.

---
## 10. Scaling Path
| Concern | Current | Scaled Approach |
|---------|---------|----------------|
| Corpus size | Small (hundreds) | Shard DB, advanced FAISS (IVF/HNSW) |
| Latency | CPU token gen | GPU inference / distillation |
| Relevance | Basic cosine | Reranking, hybrid (BM25 + dense) |
| Freshness | Manual rebuild | Incremental embeddings & delta index update |
| Observability | Manual logging | Structured logs + metrics + tracing |

---
## 11. Alternative Choices & Trade-Offs
| Layer | Chosen | Alternative | Trade-off |
|-------|--------|------------|-----------|
| Storage | MongoDB | PostgreSQL / SQLite | BSON flexibility vs relational consistency |
| Index | FAISS Flat | Milvus / Pinecone | Simplicity vs managed scalability |
| Embeddings | all-MiniLM-L6-v2 | Larger (e5-large) | Speed vs semantic richness |
| LLM | TinyLlama 1.1B | 7B model | Latency vs expressiveness |
| Front-end | React CRA | Next.js / Svelte | Familiarity vs SSR/modern features |

---
## 12. Failure Modes & Mitigations
| Failure | Cause | Mitigation |
|---------|-------|-----------|
| Port binding error | Process already running | Kill PID, retry |
| Missing optional lib | Dependency not installed | Soft-fail; log warning |
| Slow generation | Large model / CPU-only | Quantized smaller model |
| Retrieval mismatch | Chunk size too coarse | Adjust chunk parameters |
| Empty index | Forgot to build | Validate existence before query; fallback message |

---
## 13. Data Flow Example (Concrete)
**User Query:** "What is deep learning?"
1. React sends POST `/query` with JSON body.
2. FastAPI validates payload (Pydantic model).
3. Query embedding produced (MiniLM).
4. FAISS search returns vector IDs (e.g., `[12, 5]`).
5. Map IDs → Mongo `_id`s via `id_map.json`.
6. Fetch texts from Mongo: `["Deep learning is ...", "Neural networks..."]`.
7. Prompt = Template + contexts + user question.
8. llama.cpp generates answer tokens.
9. Answer + sources returned to React.
10. UI displays answer block + passage cards.

---
## 14. Concurrency & State
**FAISS Index:** Loaded into memory; read-only during queries. Rebuild invalidates previous mapping.
**MongoDB:** Safe for concurrent reads/writes; ingestion adds documents seamlessly.
**LLM:** Single-process generation; concurrent requests could queue (future improvement: async worker pool).

---
## 15. Suggested Enhancements Per Component
| Component | Enhancement |
|-----------|-------------|
| Ingestion | Semantic splitting; duplication detection |
| MongoDB | TTL for outdated passages; indexing on `source_id` |
| Embeddings | Model selection UI; caching layer |
| FAISS | Incremental add + on-disk merging |
| LLM | Streaming tokens; multi-model selection |
| API | Rate limiting; validation error detail |
| UI | History panel; filtering sources |

---
Mastering these components equips you to design, evaluate, and improve any future RAG pipeline.
