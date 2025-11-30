# Local RAG on Windows (CPU-only)

A minimal Retrieval-Augmented Generation pipeline that runs fully locally on Windows using:

- MongoDB Community Server (local)
- SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS (`faiss-cpu`) for similarity search
- `llama-cpp-python` with a local `.gguf` model for generation
- **Cloud Agent Delegation** (optional): delegate LLM generation to OpenAI or compatible cloud APIs

No Docker, CPU-only, simple and lightweight.

## Project layout

```
project_root/
  data/docs/          # .txt files to ingest
  models/model.gguf   # local llama model (place your model here)
  src/rag_windows.py  # main script
  src/cloud_agent.py  # cloud agent delegation module
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

## Cloud Agent Delegation

Instead of using a local llama.cpp model, you can delegate LLM generation to cloud-based services (OpenAI or compatible APIs). This is useful when:

- You don't have a local model file
- You want faster generation
- You need access to more capable models (GPT-4, etc.)

### Setup

1. Install the `openai` package (or `httpx` as a fallback):
   ```powershell
   pip install openai
   # or
   pip install httpx
   ```

2. Set your API key as an environment variable:
   ```powershell
   $env:OPENAI_API_KEY = "your-api-key-here"
   ```

### Usage

```powershell
# Delegate to cloud agent (uses OPENAI_API_KEY from environment)
python src/rag_windows.py --query "example question" --delegate-to-cloud

# Specify a different model
python src/rag_windows.py --query "..." --delegate-to-cloud --cloud-model "gpt-4"

# Use a custom API endpoint (OpenAI-compatible)
python src/rag_windows.py --query "..." --delegate-to-cloud --cloud-base-url "https://your-api.example.com/v1"

# Pass API key directly (not recommended for production)
python src/rag_windows.py --query "..." --delegate-to-cloud --cloud-api-key "your-key"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI or compatible service |
| `CLOUD_AGENT_MODEL` | Model to use (default: `gpt-3.5-turbo`) |
| `CLOUD_AGENT_BASE_URL` | Custom API base URL |

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
