# RAG System – Conversation Notes (Condensed Technical Reference)

These notes summarize the full set of discussion points about the local Windows RAG system: ingestion, embeddings, FAISS, LLM use, security, alternatives to Hugging Face hosted models, and architectural decisions.

---
## 1. High-Level Purpose
- Build a **fully local** Retrieval-Augmented Generation stack (no external APIs) on Windows, CPU-only.
- Components: MongoDB (passages), Embeddings (semantic vectors), FAISS (similarity search), Llama.cpp (local generation), React UI + FastAPI.
- Goal: Private, reproducible, educational demonstration + extensible foundation.

---
## 2. End-to-End Pipeline Flow
1. Ingest files in `data/docs/` → extract text (PDF, DOCX, CSV, XLSX, JSON, XML, images via OCR, Oracle/Mongo sources).
2. Chunk text (≈250 words + 50-word overlap) → store passages in MongoDB.
3. Build index: embed each passage → normalized vectors → store in `faiss.index` + `id_map.json`.
4. Query: embed user question → similarity search (top-K) in FAISS.
5. Retrieve: map FAISS IDs back to passages (source_id + text).
6. Generate: construct prompt with retrieved context → llama.cpp (`model.gguf`) → answer + citations.

---
## 3. Chunking Mechanics
- Simple word-based splitter: fixed size + overlap to preserve sentence continuity.
- Overlap avoids context fracture (e.g., splitting a critical sentence).
- Could be upgraded (sentence boundary detection, adaptive chunk sizing).

---
## 4. Embeddings (Neural) – Original Approach
- Model: `all-MiniLM-L6-v2` via `sentence-transformers`.
- Steps: tokenization → transformer forward → mean pooling → L2 normalization → 384-dim float32 vector.
- Query and documents share **same embedding space** → semantic retrieval via cosine (inner product on normalized vectors).
- Vectors stored in exact FAISS `IndexFlatIP` (fast for small/moderate corpora).

---
## 5. Embedding Lifecycle (Local)
- First run: model weights downloaded (if using HF identifier) → cached under user profile.
- Subsequent runs: loaded from disk; no network use.
- Weights are inert data; execution logic is in Python libraries.

---
## 6. Generation (LLM)
- Runtime: `llama-cpp-python` (CPU threads, quantized model `Q4_K_M`).
- Model file: `models/model.gguf` (TinyLlama variant or replacement).
- Prompt template: system instructions + merged context blocks + user question.
- Inference loop: next-token prediction until stop token or max tokens.

---
## 7. Security & Privacy Model
- All operations (embedding, retrieval, generation) run locally; no external API calls for data.
- Risks center on third-party code execution (libraries), not weights.
- Mitigations: pin versions, offline/air-gapped execution, firewall egress, internal artifact hosting.
- Model weights (.bin / .gguf) are passive; only loaders execute code.

---
## 8. Alternatives to Hugging Face Hosted Models
- Keep library, replace remote identifier with **local directory** (org-supplied weights).
- Replace neural embeddings entirely with **classical IR**: TF-IDF (scikit-learn), BM25 (rank-bm25).
- Use **internal embedding service** (REST) returning vectors (no local model files). 
- Full replacement (advanced): train/fine-tune custom model; serve via private registry.

---
## 9. Trade-offs: Neural vs Classical Retrieval
| Aspect | Neural (MiniLM) | TF-IDF/BM25 |
|--------|-----------------|------------|
| Semantic match | Strong | Lexical only |
| Setup complexity | Moderate | Low |
| Dependency risk | Higher (ML libs) | Lower |
| Offline viability | High | High |
| Performance on synonyms | Good | Weak |

---
## 10. Model & Library Separation
- **Library**: `sentence-transformers` (code scaffolding: tokenizer, pooling, batching).
- **Weights**: embedding model folder (config + tensors). Can be provided internally.
- Using local folder = avoiding HF Hub while leveraging library utilities.
- Not equivalent to “inventing embedding math” (unless you train your own).

---
## 11. Replacing Embedding Implementation (Interface Pattern)
Define a swappable function:
```python
def get_embedder(kind="local"):
    if kind == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("models/embeddings/org_model")
        return lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    if kind == "bm25":
        from rank_bm25 import BM25Okapi
        corpus_tokens = [t.split() for t in GLOBAL_CORPUS]
        bm25 = BM25Okapi(corpus_tokens)
        return lambda texts: [bm25.get_scores(q.split()) for q in texts]  # custom adapter
    raise ValueError("Unknown embedder kind")
```

---
## 12. FAISS Index Details
- Type: `IndexFlatIP` (exact inner product search).
- Requires normalized vectors → IP ≈ cosine similarity.
- Saves binary file (`faiss.index`) + position → document mapping (`id_map.json`).
- For large scale: switch to IVF/HNSW or a vector DB (Milvus, Qdrant, Weaviate).

---
## 13. Prompt Construction Principles
- Provide explicit context blocks with labeled source IDs.
- Instruction: “If answer not in context, say ‘I don’t know’.”
- Encourage citations for traceability; optionally enforce a `Sources:` line.

---
## 14. Potential Enhancements
- Reranking (cross-encoder) after initial top-K.
- Hybrid retrieval (BM25 + dense fusion).
- Metadata filtering (tags, file types, recency).
- Embedding cache (frequent queries).
- Streaming generation output in UI.

---
## 15. Key Q&A Bullet Recap
- What converts query to embeddings? Same embedding model as passages.
- Are weights executing code? No—libraries do; weights are data.
- Can libraries exfiltrate data? Technically yes; mitigate via isolation & trust chain.
- How to avoid HF? Use local model folders or classical IR methods.
- Is using local folder “making my own embeddings”? Partially—control over distribution, not training.
- OCR path? Fallback using Tesseract for scanned PDFs.
- Why overlap in chunking? Preserves cross-boundary semantic continuity.
- Why normalize embeddings? Enables cosine via inner product; stabilizes similarity scale.

---
## 16. Minimal No-HF Embedding Swap Example (TF-IDF)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
passage_matrix = vectorizer.fit_transform(passage_texts)  # sparse

def retrieve_tfidf(query: str, k: int = 5):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, passage_matrix).ravel()
    top_idx = scores.argsort()[::-1][:k]
    return [(passage_texts[i], float(scores[i])) for i in top_idx]
```

---
## 17. Security Hardening Checklist
- Pin dependencies (`requirements.txt`).
- Air-gap for sensitive runs.
- Firewall: block outbound except explicitly allowed.
- Integrity scan of model folders (hash compare).
- Log only metadata, not raw passage text.

---
## 18. When to Move Beyond This Stack
- Corpus size >> 1M passages → approximate indexing.
- Need filtering by attributes → vector DB or custom layer.
- Need multi-lingual semantic parity → stronger multilingual embedding model.
- Need faster generation → GPU or distilled quantization + prompt truncation.

---
## 19. Mental Model (One-Line Summary)
“Documents become vectors (meaning), queries become vectors (intent), similarity retrieves context, the LLM writes grounded answers — all locally, under your control.”

---
## 20. Fast Demo Script (Talking Points)
- “We ingest locally, chunk for tractable context windows.”
- “Embeddings turn language into geometry (384-D).”
- “FAISS performs semantic nearest-neighbor search.”
- “Prompt injects only retrieved context — reduces hallucination.”
- “LLM runs on CPU via quantized GGUF; private and portable.”
- “Everything can be swapped: embedding strategy, index type, model.”

---
## 21. Glossary (Quick)
- Passage: A chunked text segment stored in Mongo.
- Embedding: Numeric vector encoding semantic meaning.
- FAISS: Library for efficient similarity search.
- GGUF: Optimized on-disk format for Llama.cpp models.
- Retrieval: Selecting top-K most similar passages.
- Generation: Producing answer using retrieved context + LLM.

---
Use these notes as a rapid reference for explanations, presentations, or audits.
