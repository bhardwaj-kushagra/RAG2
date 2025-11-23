# Local RAG on Windows (CPU-only) — Project Guide

This guide explains how the project works end-to-end and how to run, extend, and troubleshoot it. It’s aimed at a CS professional familiar with general ML terms but new to Retrieval-Augmented Generation (RAG).

## What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with a language model: it retrieves relevant passages from a knowledge base and uses them as grounded context for a generative model. Benefits include:
- Better factual grounding versus pure LLM prompting
- Integrating private/domain knowledge without retraining
- Easy updates: add/modify docs and rebuild the index (no model finetuning)

## Architecture overview

Components used in this project:
- MongoDB (local): stores text passages
- SentenceTransformers (all-MiniLM-L6-v2): computes dense vector embeddings
- FAISS (CPU): similarity search over embeddings
- llama-cpp-python: runs a local `.gguf` LLM for generation

Data flow:

```
TXT files (data/docs/) --ingest--> MongoDB (rag_db.passages)
MongoDB --build-index--> Embeddings (SentenceTransformers)
                      --> FAISS index (faiss.index)
                      --> ID map (id_map.json)

Query --embed--> FAISS --top-K--> MongoDB texts --prompt--> llama.cpp --> Answer (+ [source_id] citations)
```

Key files:
- `src/rag_windows.py`: main script with CLI
- `data/docs/`: folder for your `.txt` documents
- `models/model.gguf`: local llama model
- `faiss.index`: FAISS vector index (created by `--build-index`)
- `id_map.json`: FAISS index position -> Mongo document mapping (created by `--build-index`)

## Why these choices?

- CPU-only: keeps setup lightweight and avoids GPU drivers/CUDA
- all-MiniLM-L6-v2: small, fast, decent semantic retrieval quality
- FAISS IndexFlatIP + normalized embeddings: cosine search with simple, reliable behavior
- llama.cpp: runs a local `.gguf` model on CPU; easy to swap models

## Setup (Windows, PowerShell)

Prerequisites:
- MongoDB Community Server installed and running locally (default URI: `mongodb://localhost:27017`)
- Python 3.10–3.11 recommended
- VS Code (optional but recommended)

One-time steps in the project root:

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install sentence-transformers pymongo numpy tqdm faiss-cpu

# llama-cpp-python: install a wheel that works on Windows CPU
# If the default pip install tries to build from source, use this pinned wheel:
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu `
  llama-cpp-python==0.3.2
```

Place a local `.gguf` model at `models\model.gguf`. For testing on CPU, a small model like TinyLlama works:
- Example (TinyLlama 1.1B Q4_K_M): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

## Project structure

```
project_root/
  data/docs/          # text files to ingest (you add files here)
  models/model.gguf   # local llama model
  src/rag_windows.py  # main RAG script (CLI)
  faiss.index         # vector index (generated)
  id_map.json         # index position -> Mongo mapping (generated)
```

## CLI commands

From project root (venv active):

```powershell
# 1) Ingest .txt files into MongoDB
python src/rag_windows.py --ingest

# 2) Build FAISS index over all passages
python src/rag_windows.py --build-index

# 3) Retrieval-only (no model needed):
python src/rag_windows.py --retrieve-only "Your question" --k 5

# 4) Full answer generation (needs models\model.gguf):
python src/rag_windows.py --query "Your question" --k 5

# Options:
# --k <int>        : number of passages to retrieve (default 5)
# --model-path <p> : path to your .gguf (default models/model.gguf)
```

## Pipeline internals

### 1) Ingest: chunk and store
- Reads `.txt` from `data/docs/`
- Splits into 250-word chunks with 50-word overlap (simple word-based splitter)
- Stores each chunk in MongoDB `rag_db.passages`
- Each passage has a `source_id` like `filename.txt#chunk_index`

Design notes:
- Word-based chunking is simple and robust for plain text
- Overlap helps preserve context across chunk boundaries
- MongoDB is convenient for simple CRUD and scales locally

### 2) Build-index: embed and index
- Loads all passages from MongoDB (`text`, `source_id`)
- Encodes passages using `all-MiniLM-L6-v2`, normalized embeddings
- Builds FAISS `IndexFlatIP` (inner product). With normalized vectors, IP ≈ cosine similarity
- Saves index to `faiss.index`
- Saves `id_map.json` mapping FAISS index row -> `{ mongo_id, source_id }`

Design notes:
- IndexFlatIP is exact search; simple and good for small projects
- Normalization enables cosine similarity without extra code
- `id_map.json` ensures we can recover the associated Mongo doc quickly

### 3) Retrieve: top-K from FAISS
- Encodes the query with the same embedder (normalized)
- Searches FAISS for top-K neighbors
- Uses `id_map.json` to fetch `text` + `source_id` from MongoDB
- Prints the retrieved passages for transparency

Design notes:
- Using the same embedding model for queries and passages is essential
- K is configurable; typical values 3–10

### 4) Generate: llama.cpp with citations
- Builds a prompt with instructions to cite inline using `[source_id]`
- Appends all contexts in blocks like:
  - `[filename.txt#i]:\n<chunk text>`
- Calls `llama_cpp.Llama.create_completion` with moderate constraints:
  - `max_tokens=512`, `temperature=0.2`, `top_p=0.9`
- Appends a `Sources: [id] [id] ...` line to guarantee citations are shown

Design notes:
- The prompt strongly asks for inline citations, which helps but is model-dependent
- A trailing Sources line ensures citations are never missing
- CPU-only: `n_gpu_layers=0`, `n_threads` from CPU core count

## Error handling and logging

The script prints clear logs at each stage:
- Ingest: file reads, chunk counts, Mongo inserts
- Index: model download, embedding, index write, id map write
- Retrieve: top-K hits, previews
- Generate: model load, generation start
- Missing Mongo/FAISS/model files produce readable error messages and exit codes

Common messages you might see:
- `Could not connect to MongoDB`: ensure service is running locally
- `Missing LLM model file`: place a `.gguf` at `models/model.gguf` or use `--model-path`
- `Missing FAISS index`: run `--build-index` after `--ingest`
- Hugging Face symlink warning on Windows: safe to ignore; cache will still work

## Performance and tuning

- Retrieval quality
  - Increase `--k` for more passages, at the cost of longer prompts
  - Consider better chunking (sentence-aware splitters) if your docs are complex
- Embedding performance
  - `batch_size` in code defaults to 64; tune if you have more CPU/memory
- Index type
  - `IndexFlatIP` is exact. For larger corpora, consider IVF or HNSW (not included here to keep things simple)
- Generation
  - Smaller `.gguf` models run faster on CPU (e.g., TinyLlama)
  - Adjust `n_ctx` (context window) and `max_tokens` to balance speed vs. completeness
  - If you see `n_ctx_pre_seq > n_ctx_train` warnings, reduce the context or use a model with a larger training context

## Security and privacy

- Everything runs locally: MongoDB, FAISS, embeddings, and the LLM
- Your documents never leave your machine
- Be mindful of where your HF cache is stored (user home by default)

## How to extend

- Better chunking
  - Sentence/paragraph-aware splitters; dynamic chunk sizes
- Metadata
  - Store file-level metadata (title, date, tags) and use it during retrieval and prompting
- Reranking
  - Apply a lightweight cross-encoder on the top 20 FAISS hits to rerank down to K
- Caching
  - Cache query embeddings if you issue repeated queries
- Alternative index types
  - FAISS IVF/HNSW for large-scale; or use a vector DB if you want built-in persistence and filters
- Prompt templates
  - Add role/system prompts or formatting to better match your model’s chat style

## Troubleshooting

- llama-cpp-python install tries to build from source
  - Use a prebuilt CPU wheel: `llama-cpp-python==0.3.2` via the extra index (see Setup section)
  - Otherwise install C++ build tools (MSVC + CMake) and try again
- Model is slow or runs out of memory
  - Use a smaller quantization (Q4_K_M or even Q3)
  - Reduce `--k` and/or the chunk size to shorten the prompt
- FAISS import fails
  - Ensure `faiss-cpu` (not `faiss`) is installed in your venv
- MongoDB connection fails
  - Check the Windows service: `Get-Service MongoDB`; start if needed

## Minimal “try it” run

```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install sentence-transformers pymongo numpy tqdm faiss-cpu
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu `
  llama-cpp-python==0.3.2

# Put some .txt files into data\docs\
python src/rag_windows.py --ingest
python src/rag_windows.py --build-index

# Place a local .gguf at models\model.gguf (or pass --model-path)
python src/rag_windows.py --query "What is RAG?" --k 3
```

## Source references

- Script: `src/rag_windows.py`
- Generated: `faiss.index`, `id_map.json`
- Data: `data/docs/` (your inputs)
- Model: `models/model.gguf`

## License

Use third-party models and data according to their licenses. This project is provided for educational/demo purposes.
