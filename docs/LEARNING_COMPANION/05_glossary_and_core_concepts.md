# 05 – Glossary & Core Concepts

This glossary decodes terminology used throughout the project. Each entry includes: Definition, Why It Matters Here, Practical Tip.

---
## Retrieval-Augmented Generation (RAG)
**Definition:** Pattern combining external knowledge retrieval with a generative model to produce informed answers.
**Why It Matters:** Keeps model responses grounded in current, local data.
**Tip:** Always cite sources for trust & debugging.

## Passage / Chunk
**Definition:** A segment of the original document, sized to fit embedding & prompt constraints.
**Why:** Smaller chunks improve recall; too small harms context coherence.
**Tip:** Tune chunk size/overlap based on domain (e.g., legal vs tweets).

## Embedding
**Definition:** Numeric vector representing semantic meaning of text.
**Why:** Enables similarity comparison in continuous space.
**Tip:** Keep consistent model for both passages & queries.

## Cosine Similarity
**Definition:** Measure of angle between vectors; values near 1 imply high similarity.
**Why:** Semantic ranking used in FAISS search.
**Tip:** Ensure vectors normalized when using inner product index.

## FAISS
**Definition:** Library for efficient similarity search over vectors.
**Why:** Powers fast top-K retrieval even as corpus grows.
**Tip:** Start with Flat index; evolve to approximate strategies when scaling.

## MongoDB
**Definition:** Document-oriented NoSQL database.
**Why:** Flexible schema for varied ingestion formats (PDF, JSON, etc.).
**Tip:** Index fields like `source_id` for faster lookups.

## llama.cpp
**Definition:** Lightweight C/C++ inference backend supporting quantized LLMs.
**Why:** Enables running LLM locally on CPU with acceptable performance.
**Tip:** Use quantized GGUF models (Q4_K_M etc.) for memory efficiency.

## Quantization
**Definition:** Reducing numeric precision of model weights (e.g., 16-bit → 4-bit) to shrink size & improve speed.
**Why:** Makes local CPU inference feasible.
**Tip:** Slight quality trade-off; choose level appropriate for demo vs production.

## OCR (Optical Character Recognition)
**Definition:** Converting image-based text into machine-readable strings.
**Why:** Unlocks screenshots & scanned documents for retrieval.
**Tip:** Post-process output (strip irregular spacing) to improve embedding quality.

## Prompt Engineering
**Definition:** Crafting model input to elicit useful, structured responses.
**Why:** Ensures citations & reduces hallucinations.
**Tip:** Include explicit instruction: “Use ONLY provided sources and cite them.”

## Citation Mapping
**Definition:** Linking generated answer references back to original source chunks.
**Why:** Transparency & trust; enables verification.
**Tip:** Maintain ID map (`faiss_id -> mongo_id`).

## Normalization (Vector)
**Definition:** Scaling vector so its L2 norm equals 1.
**Why:** Converts inner product into cosine similarity.
**Tip:** Always re-check norm: `np.linalg.norm(vec) ≈ 1.0`.

## ANN (Approximate Nearest Neighbor)
**Definition:** Algorithms for fast similarity search with some accuracy trade-off.
**Why:** Required for large-scale corpora beyond brute force.
**Tip:** Not necessary for small demo sets (<100k vectors).

## CRUD (Create, Read, Update, Delete)
**Definition:** Basic data operations across persistence layers.
**Why:** In RAG ingestion = Create, retrieval = Read; rarely update/delete in prototypes.
**Tip:** Add update/delete when handling dynamic knowledge bases.

## Soft-Fail Strategy
**Definition:** Failing gracefully (skip feature) instead of crashing entire pipeline.
**Why:** Maintains ingestion progress even if, e.g., OCR library missing.
**Tip:** Log actionable warning messages.

## Reranking
**Definition:** Second-stage scoring of retrieved candidates using more precise models.
**Why:** Improves relevance beyond initial quick ANN results.
**Tip:** Integrate when quality matters over raw speed.

## Chunk Overlap
**Definition:** Number of words reused between consecutive chunks.
**Why:** Preserves context that spans boundary.
**Tip:** Overlap 10–20% typical; too large wastes storage.

## Latency Budget
**Definition:** Acceptable time for user-visible operations.
**Why:** Guides model size & retrieval complexity choices.
**Tip:** Separate retrieval (<1s) from generation (longer) for UX clarity.

## Index Rebuild
**Definition:** Full recreation of vector index after large ingestion changes.
**Why:** Ensures new passages searchable.
**Tip:** Consider incremental append for scalability.

## Front-End Proxy
**Definition:** Dev server feature forwarding API calls to backend.
**Why:** Avoid manual CORS config complexity in dev.
**Tip:** Keep proxy in `package.json`; adjust port as needed.

## CORS (Cross-Origin Resource Sharing)
**Definition:** Browser security policy controlling resource access across origins.
**Why:** Must allow React UI (localhost:3000) to call API (localhost:8000).
**Tip:** Restrict origins in production.

## Idempotent Operation
**Definition:** Running the same action multiple times yields same result (no additional side-effects).
**Why:** E.g., setup scripts safe to re-run.
**Tip:** Make build / start scripts idempotent for reliability.

## Structured Logging (Future)
**Definition:** Machine-parseable log format (JSON lines) for observability.
**Why:** Enables metrics, dashboards, tracing.
**Tip:** Use `logging` module with custom formatters.

## Vector Store
**Definition:** System that stores & retrieves embeddings (FAISS in this project).
**Why:** Core to semantic retrieval.
**Tip:** Compare alternatives (Weaviate, Pinecone, Milvus) when scaling.

## Semantic Drift
**Definition:** Divergence between retrieved context and intended user query meaning.
**Why:** Causes poor answers.
**Tip:** Improve with better chunking + reranking + filter criteria.

---
## Concept Relationships Graph (Simplified)
```
[Documents] --(Extraction+OCR)--> [Text] --(Chunking)--> [Passages]
[Passages] --(Embedding)--> [Vectors] --(Indexing)--> [FAISS Index]
[Query] --(Embedding)--> [Query Vector] --(Similarity)--> [Top-K IDs]
[Top-K IDs] --(Mapping)--> [Passage Texts] --(Prompt Assembly)--> [LLM Generation]
[LLM Generation] --(Citation Formatting)--> [Answer + Sources]
```

---
## Learning Path Recommendations
1. Reimplement chunking with sentence tokenization.
2. Swap embedding model; compare retrieval quality.
3. Add reranking stage; measure improvement.
4. Replace Flat index with IVF to learn ANN trade-offs.
5. Implement streaming token output in UI.

---
Use this glossary as a quick reference while rebuilding the system.
