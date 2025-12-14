# RAGAS Evaluation Guide

This guide explains how to use the RAGAS (Retrieval Augmented Generation Assessment) evaluation module to assess the quality of your RAG pipeline.

## Overview

RAGAS is a framework specifically designed to evaluate RAG systems. It provides metrics that measure different aspects of RAG performance:

| Metric | Description | Range |
|--------|-------------|-------|
| **Faithfulness** | Measures if the generated answer is factually grounded in the retrieved context | 0-1 |
| **Answer Relevancy** | Measures if the answer actually addresses the question asked | 0-1 |
| **Context Precision** | Measures if the retrieved contexts are relevant to the question | 0-1 |
| **Context Recall** | Measures if the contexts contain all information needed to answer | 0-1 |

## Installation

The evaluation module requires the following dependencies:

```bash
pip install ragas datasets sentence-transformers numpy
```

For full RAGAS evaluation with LLM-based metrics, you'll also need:

```bash
pip install langchain-openai
```

## Quick Start

### Run Simplified Evaluation (No API Key Required)

```bash
python src/evaluation.py --evaluate
```

This runs an embedding-based evaluation that doesn't require external API calls.

### Run Full RAGAS Evaluation (Requires OpenAI API Key)

```bash
OPENAI_API_KEY=sk-your-key python src/evaluation.py --evaluate --full
```

### Save Results to Custom Path

```bash
python src/evaluation.py --evaluate --output my_results.json
```

## Understanding the Results

### Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| > 0.8 | Excellent - Your RAG system is performing very well |
| 0.7 - 0.8 | Good - Solid performance with room for improvement |
| 0.5 - 0.7 | Moderate - Consider tuning your pipeline |
| < 0.5 | Needs Improvement - Review retrieval and generation |

### Example Output

```text
============================================================
           RAGAS EVALUATION RESULTS
============================================================

Evaluation Type: simplified
Number of Samples: 5

--- Aggregate Metrics ---
  Faithfulness:       0.6500
  Answer Relevancy:   0.7200
  Context Precision:  0.5800
  Context Recall:     0.6100

--- Individual Sample Scores ---

  Sample 1: What is machine learning?...
    - Faithfulness:       0.7102
    - Answer Relevancy:   0.8339
    - Context Precision:  0.6055
    - Context Recall:     0.7347
```

## Evaluation Modes

### 1. Simplified Mode (Default)

The simplified mode uses embedding-based similarity to approximate RAGAS metrics:

- **Pros**: Fast, no API costs, works offline
- **Cons**: Less accurate than LLM-based evaluation

```python
from evaluation import run_evaluation_simple
results = run_evaluation_simple()
```

### 2. Full RAGAS Mode

The full mode uses LLM-based evaluation for more accurate metrics:

- **Pros**: More accurate, follows official RAGAS methodology
- **Cons**: Requires OpenAI API key, has API costs

```python
from evaluation import run_evaluation_with_ragas
results = run_evaluation_with_ragas()
```

## Creating Custom Evaluation Data

You can create your own evaluation samples:

```python
from evaluation import EvaluationSample, run_evaluation_simple

samples = [
    EvaluationSample(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=[
            "France is a country in Western Europe. Its capital city is Paris.",
            "Paris is known for the Eiffel Tower and the Louvre Museum."
        ],
        ground_truth="The capital of France is Paris."
    ),
    # Add more samples...
]

results = run_evaluation_simple(samples)
```

## Improving Your Scores

### Low Faithfulness Score

- Your answers may contain information not in the retrieved context
- Solution: Tune your prompt to emphasize using only the provided context

### Low Answer Relevancy Score

- Answers don't directly address the questions
- Solution: Improve your prompt to focus on answering the specific question

### Low Context Precision Score

- Retrieved documents aren't relevant to queries
- Solution: Improve your embedding model or retrieval parameters (e.g., top-K)

### Low Context Recall Score

- Retrieved documents don't contain needed information
- Solution: Add more documents to your knowledge base, improve chunking strategy

## Integration with the RAG Pipeline

The evaluation module is designed to work with the existing RAG pipeline. You can evaluate actual RAG responses:

```python
from rag_windows import retrieve, generate_answer
from evaluation import EvaluationSample, run_evaluation_simple

# Get actual RAG responses
question = "What is machine learning?"
contexts = retrieve(question, top_k=3)
answer = generate_answer(question, contexts, model_path)

# Create evaluation sample
sample = EvaluationSample(
    question=question,
    answer=answer,
    contexts=[p.text for p in contexts],
    ground_truth="Your expected answer here"
)

# Evaluate
results = run_evaluation_simple([sample])
```

## Incremental Indexing Note

Your FAISS index build step now supports incremental updates by appending only new MongoDB passages. For best evaluation fidelity:

- Re-run `--build-index` after ingesting new documents so retrieval reflects the latest corpus.
- Use `--no-incremental` occasionally (e.g., weekly) to fully rebuild and defragment ordering.
- Improved recall from fresh index updates should raise Context Recall and Faithfulness scores over time.

Example:

```bash
python src/rag_windows.py --ingest-files
python src/rag_windows.py --build-index          # incremental append
python src/evaluation.py --evaluate              # simplified metrics
OPENAI_API_KEY=sk-... python src/evaluation.py --evaluate --full  # full RAGAS
```

## API Reference

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--evaluate` | Run RAGAS evaluation on sample data |
| `--full` | Use full RAGAS evaluation with LLM (requires OPENAI_API_KEY) |
| `--output` | Output path for results JSON (default: evaluation_results.json) |

### Python API

```python
# Core functions
from evaluation import (
    run_evaluation_simple,      # Simplified embedding-based evaluation
    run_evaluation_with_ragas,  # Full RAGAS evaluation (needs API key)
    get_sample_evaluation_data, # Get default sample data
    EvaluationSample,           # Data class for samples
    save_results,               # Save results to JSON
    print_results,              # Pretty print results
)
```

## Offline Mode

When network access is unavailable, the module automatically falls back to a simple text embedding approach. You can force offline mode:

```bash
HF_HUB_OFFLINE=1 python src/evaluation.py --evaluate
```

## Troubleshooting

### "Network unavailable" message

### "OPENAI_API_KEY not found"

```bash
export OPENAI_API_KEY=sk-your-key
```

### Import errors

```bash
pip install ragas datasets sentence-transformers numpy
```

## References

- [RAGAS Documentation](https://docs.ragas.io/)
