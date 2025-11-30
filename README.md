# Local RAG on Windows (CPU-only)

A minimal Retrieval-Augmented Generation pipeline that runs fully locally on Windows using:

- MongoDB Community Server (local)
- SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS (`faiss-cpu`) for similarity search
- `llama-cpp-python` with a local `.gguf` model for generation
- **RAGAS** for RAG pipeline evaluation

No Docker, CPU-only, simple and lightweight.

## Project layout

```
project_root/
  data/docs/                    # .txt files to ingest
  data/sample_eval_data.json    # sample data for RAGAS evaluation
  models/model.gguf             # local llama model (place your model here)
  src/rag_windows.py            # main RAG script
  src/ragas_evaluator.py        # RAGAS evaluation module
  tests/                        # test suite
  faiss.index                   # saved FAISS index (created by --build-index)
  id_map.json                   # FAISS -> Mongo mapping (created by --build-index)
  requirements.txt              # Python dependencies
```

## Quick start (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

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

## RAGAS Evaluation

This project includes [RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) for evaluating RAG pipeline quality.

### What is RAGAS?

RAGAS is a framework that evaluates RAG pipelines using the following metrics:

- **Faithfulness**: How well the generated answer is grounded in the retrieved context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: How relevant the retrieved contexts are to the question
- **Context Recall**: How well the contexts cover the ground truth answer

### Running RAGAS Evaluation

```powershell
# Run evaluation with built-in sample data
python src/ragas_evaluator.py --evaluate

# Run evaluation with custom test data
python src/ragas_evaluator.py --evaluate --test-data data/sample_eval_data.json

# Specify output file for results
python src/ragas_evaluator.py --evaluate --output my_results.json

# Run specific metrics only
python src/ragas_evaluator.py --evaluate --metrics faithfulness answer_relevancy
```

### Full RAGAS vs Mock Evaluation

- **Full RAGAS** (requires OpenAI API key): Uses LLM-based evaluation for accurate semantic similarity
  ```powershell
  $env:OPENAI_API_KEY = "your-api-key"
  python src/ragas_evaluator.py --evaluate
  ```

- **Mock Evaluation** (no API key): Falls back to word overlap metrics for basic evaluation
  - Useful for testing and development
  - Results are less accurate but provide a baseline

### Test Data Format

Create a JSON file with the following structure:

```json
[
  {
    "question": "Your question here?",
    "answer": "The generated answer from RAG pipeline",
    "contexts": ["Retrieved context 1", "Retrieved context 2"],
    "ground_truth": "The expected correct answer"
  }
]
```

See `data/sample_eval_data.json` for a complete example.

### Sample Evaluation Results

Running the RAGAS evaluator produces results like:

```
==================================================
 RAGAS Evaluation Results
==================================================

Evaluation Mode: mock (word overlap)
Number of Samples: 3
Metrics Used: faithfulness, answer_relevancy, context_precision, context_recall

--- Average Scores ---
  faithfulness: 0.7222
  answer_relevancy: 0.6300
  context_precision: 0.2826
  context_recall: 0.5970
  overall_score: 0.5579
```

### Running Tests

```powershell
# Run all tests
pytest tests/ -v

# Run RAGAS evaluator tests only
pytest tests/test_ragas_evaluator.py -v
```
