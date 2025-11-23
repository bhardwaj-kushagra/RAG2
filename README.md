# Local RAG on Windows (CPU-only)

A minimal Retrieval-Augmented Generation pipeline that runs fully locally on Windows using:

- MongoDB Community Server (local)
- SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS (`faiss-cpu`) for similarity search
- `llama-cpp-python` with a local `.gguf` model for generation

No Docker, CPU-only, simple and lightweight.

## Project layout

```
project_root/
  data/docs/          # .txt files to ingest
  models/model.gguf   # local llama model (place your model here)
  src/rag_windows.py  # main script
  faiss.index         # saved FAISS index (created by --build-index)
  id_map.json         # FAISS -> Mongo mapping (created by --build-index)
```

## Quick start (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install sentence-transformers pymongo numpy tqdm faiss-cpu llama-cpp-python

# 1) Put your .txt files into data\docs\
python src/rag_windows.py --ingest

# 2) Build the FAISS index
python src/rag_windows.py --build-index

# 3) Place a local .gguf model at models\model.gguf
#    (or pass --model-path to point to your file)

# 4) Ask a question
python src/rag_windows.py --query "example question"

# (Optional) Retrieval-only test (no llama model needed)
python src/rag_windows.py --retrieve-only "example question" --k 5

## Full guide

For a detailed, end-to-end explanation (architecture, setup, pipeline internals, tuning, troubleshooting), see:

- docs/PROJECT_GUIDE.md

## Ingestion module reference

Extended ingestion supports txt, pdf (with OCR fallback), docx, csv, xlsx, json, xml, images (png/jpg/jpeg/tiff), Oracle DB rows, and external Mongo collections.
See `docs/INGESTION_MODULE.md` for setup, optional dependencies, and examples.

## REST API

FastAPI server exposing all RAG capabilities via HTTP endpoints. See `docs/API_GUIDE.md` for setup and usage.

Quick start:
```powershell
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

## Web UI

React-based web interface for demos and presentations. See `docs/WEB_UI_GUIDE.md` for setup.

Quick start:
```powershell
# Terminal 1: Start API server
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start React dev server
cd web
npm install
npm start
# Web UI at http://localhost:3000
```
```

Notes:
- MongoDB must be installed and running locally at `mongodb://localhost:27017`.
- The first time you run embedding it will download the model into the HF cache.

## How it works

- `--ingest`: reads `.txt` files in `data/docs/`, splits them into 250-word chunks with 50-word overlap, stores chunks in MongoDB (`rag_db.passages`).
- `--build-index`: embeds all passages with `all-MiniLM-L6-v2`, normalizes embeddings, builds a FAISS IndexFlatIP (cosine similarity) and saves `faiss.index` and `id_map.json`.
- `--query`: embeds the question, retrieves top-K passages from FAISS, fetches their text from MongoDB, and prompts the local llama.cpp model. The answer includes inline citations when possible and always appends a `Sources: [id] ...` line.

## Options

```powershell
# Change top-K passages
python src/rag_windows.py --query "..." --k 8

# Use a different local model path
python src/rag_windows.py --query "..." --model-path "c:\\path\\to\\your.gguf"
```

## Troubleshooting

- If PowerShell blocks activation, run once as Administrator:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
- If `llama-cpp-python` fails to install: ensure you have a recent version of Python (3.9+ recommended). Wheels are provided for Windows CPU; if a build is attempted, install Microsoft C++ Build Tools.
- If FAISS import fails, make sure you installed `faiss-cpu` (not `faiss`).
- If MongoDB connection fails, make sure the service is running and reachable at `mongodb://localhost:27017`.

## License

This project is for educational/demo purposes. Use models and data in accordance with their respective licenses.
