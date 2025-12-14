#!/usr/bin/env python3
"""
RAGAS Evaluation Module for RAG System

This module provides functionality to evaluate the RAG pipeline using the RAGAS
(Retrieval Augmented Generation Assessment) framework.

RAGAS metrics evaluate:
- Faithfulness: How factually accurate is the generated answer given the context
- Answer Relevancy: How relevant is the answer to the question
- Context Precision: Are retrieved contexts relevant and precise
- Context Recall: Does the context contain all information needed to answer

Usage:
    python src/evaluation.py --evaluate           # Run evaluation with sample data
    python src/evaluation.py --evaluate --output results.json  # Save results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np

# --- Third-party deps (optional for full RAGAS) ---
RAGAS_AVAILABLE = False
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    pass  # Will use simplified evaluation

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Try to load sentence-transformers, but fall back to synthetic embeddings if unavailable
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None

# Optional local embeddings via llama.cpp
LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False

# --- Constants & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results.json"
HUGGINGFACE_HOST = "huggingface.co"
HUGGINGFACE_PORT = 443
# Default local embedding model for evaluation: nomic-embed-text-v1.5 Q4_K_M
DEFAULT_EMBED_MODEL_PATH = PROJECT_ROOT / "models" / "nomic-embed-text-v1.5.Q4_K_M.gguf"


def log(msg: str) -> None:
    """Log message to stdout."""
    print(msg, flush=True)


# --- Simple Text Embedding (fallback when models unavailable) ---
def simple_text_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Generate a simple deterministic embedding from text using hashing.
    
    This is a fallback for when ML models are unavailable (e.g., no network access).
    It provides consistent embeddings that can be used for relative similarity comparisons.
    
    Args:
        text: Input text to embed
        dim: Dimension of the embedding vector
        
    Returns:
        Normalized embedding vector
    """
    # Normalize text
    text = text.lower().strip()
    
    # Create word-level features
    words = text.split()
    
    # Use hashing to create a deterministic vector
    # Note: Using SHA-256 for better practices, though this is non-cryptographic use
    embedding = np.zeros(dim, dtype=np.float32)
    
    for i, word in enumerate(words):
        # Hash the word with position information
        h = hashlib.sha256(f"{word}_{i % 10}".encode()).hexdigest()
        # Use hash to set values in embedding
        for j in range(min(dim, len(h))):
            idx = j % dim
            val = int(h[j], 16) / 15.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding[idx] += val * (1.0 / (1 + i * 0.1))  # Decay with position
    
    # Also incorporate character n-grams for better coverage
    for i in range(len(text) - 2):
        ngram = text[i:i+3]
        h = hashlib.sha256(ngram.encode()).hexdigest()[:4]
        idx = int(h, 16) % dim
        embedding[idx] += 0.1
    
    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def batch_simple_embedding(texts: List[str], dim: int = 128) -> np.ndarray:
    """Generate embeddings for a batch of texts."""
    return np.array([simple_text_embedding(t, dim) for t in texts])


class SimpleEmbedder:
    """
    Simple text embedder that works without external dependencies.
    
    Used as a fallback when SentenceTransformers is unavailable or
    when there's no network access to download models.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        
    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        embeddings = batch_simple_embedding(texts, self.dim)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        return embeddings


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Handles both 1D and 2D arrays properly.
    """
    # Flatten to 1D if needed
    e1 = emb1.flatten()
    e2 = emb2.flatten()
    
    dot_product = np.dot(e1, e2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


@dataclass
class EvaluationSample:
    """A single evaluation sample for RAGAS."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_sample_evaluation_data() -> List[EvaluationSample]:
    """
    Returns sample evaluation data for testing the RAG pipeline.
    
    These samples represent typical question-context-answer triplets
    that would be generated by a RAG system.
    """
    samples = [
        EvaluationSample(
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed. It uses algorithms to identify patterns in data.",
            contexts=[
                "Machine learning is a subset of artificial intelligence (AI) that focuses on building systems that can learn from data. These systems improve their performance over time without being explicitly programmed for specific tasks.",
                "AI encompasses various technologies including machine learning, deep learning, and natural language processing. Machine learning algorithms identify patterns in data to make predictions.",
            ],
            ground_truth="Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve without explicit programming."
        ),
        EvaluationSample(
            question="What are neural networks?",
            answer="Neural networks are computing systems inspired by biological neural networks in the brain. They consist of layers of interconnected nodes (neurons) that process information and can learn to recognize patterns in data.",
            contexts=[
                "Neural networks are computational models inspired by the human brain's structure. They consist of interconnected nodes organized in layers: input, hidden, and output layers.",
                "Deep learning uses neural networks with multiple hidden layers. These deep neural networks can learn hierarchical representations of data, enabling them to solve complex problems.",
            ],
            ground_truth="Neural networks are computing systems inspired by the brain, consisting of interconnected layers of nodes that can learn patterns from data."
        ),
        EvaluationSample(
            question="What is retrieval augmented generation?",
            answer="Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context for a language model to generate more accurate and grounded responses.",
            contexts=[
                "Retrieval Augmented Generation (RAG) enhances large language models by retrieving relevant information from external knowledge sources before generating responses.",
                "RAG systems typically use vector databases to store document embeddings, enabling semantic search to find relevant context for user queries.",
            ],
            ground_truth="RAG is an AI technique that retrieves relevant documents and uses them as context for language models to generate informed responses."
        ),
        EvaluationSample(
            question="What is FAISS?",
            answer="FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It enables fast nearest neighbor search in large-scale datasets.",
            contexts=[
                "FAISS is an open-source library developed by Facebook AI Research for efficient similarity search. It supports various index types optimized for different use cases.",
                "For semantic search, FAISS indexes embedding vectors and performs approximate nearest neighbor search to find similar documents quickly.",
            ],
            ground_truth="FAISS is a library by Facebook AI for efficient similarity search and clustering of dense vectors."
        ),
        EvaluationSample(
            question="How do embeddings work in RAG systems?",
            answer="In RAG systems, embeddings convert text into numerical vectors that capture semantic meaning. Documents and queries are embedded into the same vector space, allowing similarity search to find relevant context based on semantic similarity rather than keyword matching.",
            contexts=[
                "Embeddings are numerical representations of text that capture semantic meaning. Similar concepts have similar vector representations in embedding space.",
                "RAG systems embed both documents and user queries. Similarity between query and document embeddings determines relevance for retrieval.",
                "SentenceTransformers is a popular library for generating text embeddings using pre-trained models like all-MiniLM-L6-v2.",
            ],
            ground_truth="Embeddings convert text to numerical vectors capturing meaning. RAG systems compare query and document embeddings to find relevant context."
        ),
    ]
    return samples


def create_evaluation_dataset(samples: List[EvaluationSample]):
    """
    Convert evaluation samples to a Hugging Face Dataset format required by RAGAS.
    
    Args:
        samples: List of EvaluationSample objects
        
    Returns:
        Dataset in RAGAS-compatible format or None if datasets unavailable
    """
    if not DATASETS_AVAILABLE:
        log("[evaluation] datasets library not available, skipping Dataset creation")
        return None
        
    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    }
    return Dataset.from_dict(data)


def get_embedder(embed_model_path: Optional[Path] = None):
    """
    Get the best available embedder.
    
    Returns SentenceTransformer if available and network accessible,
    otherwise falls back to SimpleEmbedder.
    
    Set HF_HUB_OFFLINE=1 environment variable to skip network attempts.
    """
    # Check if offline mode is forced
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    # Try local llama.cpp embeddings first if available
    if LLAMA_CPP_AVAILABLE:
        model_path = embed_model_path or DEFAULT_EMBED_MODEL_PATH
        if model_path.exists():
            try:
                log(f"[evaluation] Using local llama.cpp embeddings: {model_path}")
                llm = Llama(model_path=str(model_path), n_ctx=2048, embedding=True)

                class LlamaEmbedder:
                    def encode(self, texts: List[str], normalize_embeddings: bool = True):
                        vecs: List[np.ndarray] = []
                        for t in texts:
                            out = llm.create_embedding(t)
                            emb = np.array(out["data"][0]["embedding"], dtype="float32")
                            if normalize_embeddings:
                                n = np.linalg.norm(emb)
                                if n > 0:
                                    emb = emb / n
                            vecs.append(emb)
                        return np.vstack(vecs)

                return LlamaEmbedder()
            except Exception as e:
                log(f"[evaluation] Failed to init llama.cpp embeddings, falling back: {e}")

    if not offline_mode and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Set a short timeout for network requests
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(5)  # 5 second timeout
            
            # Check network availability first
            try:
                socket.create_connection((HUGGINGFACE_HOST, HUGGINGFACE_PORT), timeout=2)
                network_available = True
            except (socket.timeout, socket.error, OSError):
                network_available = False
            finally:
                socket.setdefaulttimeout(old_timeout)
            
            if network_available:
                log("[evaluation] Attempting to load SentenceTransformer model...")
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                log("[evaluation] SentenceTransformer loaded successfully")
                return embedder
            else:
                log("[evaluation] Network unavailable, skipping SentenceTransformer")
        except Exception as e:
            log(f"[evaluation] SentenceTransformer unavailable: {str(e)[:100]}...")
            log("[evaluation] Falling back to simple embedder")
    
    log("[evaluation] Using simple text embedder (offline mode)")
    return SimpleEmbedder(dim=128)


def run_evaluation_simple(samples: Optional[List[EvaluationSample]] = None, embed_model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a simplified RAGAS-style evaluation without requiring an LLM API key.
    
    This function computes metrics using embedding-based similarity measures
    that don't require external LLM API calls.
    
    Args:
        samples: List of evaluation samples. If None, uses default samples.
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if samples is None:
        samples = get_sample_evaluation_data()
    
    log("[evaluation] Running simplified RAGAS-style evaluation...")
    log(f"[evaluation] Processing {len(samples)} samples...")
    
    # Load embedding model (with fallback)
    log("[evaluation] Loading embedding model...")
    embedder = get_embedder(Path(embed_model_path) if embed_model_path else None)
    
    # Compute metrics based on embedding similarity
    results = {
        "faithfulness_scores": [],
        "answer_relevancy_scores": [],
        "context_precision_scores": [],
        "context_recall_scores": [],
    }
    
    for i, sample in enumerate(samples):
        log(f"[evaluation] Processing sample {i+1}/{len(samples)}: {sample.question[:50]}...")
        
        # Embed all texts
        question_emb = embedder.encode([sample.question], normalize_embeddings=True)[0]
        answer_emb = embedder.encode([sample.answer], normalize_embeddings=True)[0]
        context_embs = embedder.encode(sample.contexts, normalize_embeddings=True)
        ground_truth_emb = embedder.encode([sample.ground_truth], normalize_embeddings=True)[0]
        
        # Faithfulness: How well does the answer match the context
        # (answer-context similarity)
        answer_context_sims = [
            cosine_similarity(answer_emb, ctx_emb)
            for ctx_emb in context_embs
        ]
        faithfulness_score = max(answer_context_sims) if answer_context_sims else 0.0
        results["faithfulness_scores"].append(faithfulness_score)
        
        # Answer Relevancy: How relevant is the answer to the question
        answer_relevancy_score = cosine_similarity(answer_emb, question_emb)
        results["answer_relevancy_scores"].append(answer_relevancy_score)
        
        # Context Precision: How relevant are the contexts to the question
        context_question_sims = [
            cosine_similarity(question_emb, ctx_emb)
            for ctx_emb in context_embs
        ]
        context_precision_score = sum(context_question_sims) / len(context_question_sims) if context_question_sims else 0.0
        results["context_precision_scores"].append(context_precision_score)
        
        # Context Recall: How well do contexts cover the ground truth
        context_ground_truth_sims = [
            cosine_similarity(ground_truth_emb, ctx_emb)
            for ctx_emb in context_embs
        ]
        context_recall_score = max(context_ground_truth_sims) if context_ground_truth_sims else 0.0
        results["context_recall_scores"].append(context_recall_score)
    
    # Compute average scores
    avg_results = {
        "faithfulness": sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]),
        "answer_relevancy": sum(results["answer_relevancy_scores"]) / len(results["answer_relevancy_scores"]),
        "context_precision": sum(results["context_precision_scores"]) / len(results["context_precision_scores"]),
        "context_recall": sum(results["context_recall_scores"]) / len(results["context_recall_scores"]),
        "num_samples": len(samples),
        "individual_scores": {
            "questions": [s.question for s in samples],
            "faithfulness": results["faithfulness_scores"],
            "answer_relevancy": results["answer_relevancy_scores"],
            "context_precision": results["context_precision_scores"],
            "context_recall": results["context_recall_scores"],
        }
    }
    
    return avg_results


def run_evaluation_with_ragas(
    samples: Optional[List[EvaluationSample]] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full RAGAS evaluation using LLM-based metrics.
    
    This requires an OpenAI API key for the LLM-based evaluation metrics.
    
    Args:
        samples: List of evaluation samples. If None, uses default samples.
        openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        
    Returns:
        Dictionary containing RAGAS evaluation results
    """
    if samples is None:
        samples = get_sample_evaluation_data()
    
    # Check for API key
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        log("[evaluation] No OpenAI API key found. Falling back to simplified evaluation.")
        log("[evaluation] Set OPENAI_API_KEY environment variable for full RAGAS evaluation.")
        return run_evaluation_simple(samples)
    
    if ChatOpenAI is None or OpenAIEmbeddings is None:
        log("[evaluation] langchain_openai not installed. Falling back to simplified evaluation.")
        return run_evaluation_simple(samples)
    
    log("[evaluation] Running full RAGAS evaluation with LLM-based metrics...")
    log(f"[evaluation] Processing {len(samples)} samples...")
    
    # Create dataset
    dataset = create_evaluation_dataset(samples)
    
    # Initialize LLM and embeddings
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Wrap for RAGAS
    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # Create RAGAS dataset
    eval_dataset = EvaluationDataset.from_hf_dataset(dataset)
    
    # Define metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    # Run evaluation
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
        )
        
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
            "num_samples": len(samples),
            "evaluation_type": "ragas_full",
        }
    except Exception as e:
        log(f"[evaluation] RAGAS evaluation failed: {e}")
        log("[evaluation] Falling back to simplified evaluation...")
        return run_evaluation_simple(samples)


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"[evaluation] Results saved to {output_path}")


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results in a formatted way."""
    log("\n" + "=" * 60)
    log("           RAGAS EVALUATION RESULTS")
    log("=" * 60)
    
    eval_type = results.get("evaluation_type", "simplified")
    log(f"\nEvaluation Type: {eval_type}")
    log(f"Number of Samples: {results.get('num_samples', 'N/A')}")
    
    log("\n--- Aggregate Metrics ---")
    log(f"  Faithfulness:       {results.get('faithfulness', 0):.4f}")
    log(f"  Answer Relevancy:   {results.get('answer_relevancy', 0):.4f}")
    log(f"  Context Precision:  {results.get('context_precision', 0):.4f}")
    log(f"  Context Recall:     {results.get('context_recall', 0):.4f}")
    
    # Print individual scores if available
    if "individual_scores" in results:
        log("\n--- Individual Sample Scores ---")
        ind = results["individual_scores"]
        for i, question in enumerate(ind.get("questions", [])):
            log(f"\n  Sample {i+1}: {question[:50]}...")
            log(f"    - Faithfulness:       {ind['faithfulness'][i]:.4f}")
            log(f"    - Answer Relevancy:   {ind['answer_relevancy'][i]:.4f}")
            log(f"    - Context Precision:  {ind['context_precision'][i]:.4f}")
            log(f"    - Context Recall:     {ind['context_recall'][i]:.4f}")
    
    log("\n" + "=" * 60)
    log("           INTERPRETATION GUIDE")
    log("=" * 60)
    log("""
  Faithfulness (0-1):
    Measures if the answer is grounded in the given context.
    Higher = more factually consistent with retrieved documents.
    
  Answer Relevancy (0-1):
    Measures if the answer addresses the question asked.
    Higher = more relevant and on-topic responses.
    
  Context Precision (0-1):
    Measures if retrieved contexts are relevant to the question.
    Higher = better retrieval of relevant information.
    
  Context Recall (0-1):
    Measures if contexts contain information needed for the answer.
    Higher = more complete information retrieval.
    
  Target Scores:
    > 0.7: Good performance
    > 0.8: Excellent performance
    < 0.5: Needs improvement
""")
    log("=" * 60)


def main() -> None:
    """CLI entry point for RAGAS evaluation."""
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/evaluation.py --evaluate
  python src/evaluation.py --evaluate --output results.json
  OPENAI_API_KEY=sk-... python src/evaluation.py --evaluate --full
        """
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAGAS evaluation on sample data"
    )
    parser.add_argument(
        "--embed-model-path",
        type=str,
        default=str(DEFAULT_EMBED_MODEL_PATH),
        help="Path to local .gguf used for embeddings (llama.cpp)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full RAGAS evaluation with LLM (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON (default: evaluation_results.json)"
    )
    
    args = parser.parse_args()
    
    log("=" * 60)
    log("       RAGAS - RAG Assessment Framework")
    log("       Evaluating RAG Pipeline Quality")
    log("=" * 60)
    
    if not args.evaluate:
        parser.print_help()
        return
    
    # Run evaluation
    if args.full:
        results = run_evaluation_with_ragas()
    else:
        results = run_evaluation_simple(embed_model_path=args.embed_model_path)
        results["evaluation_type"] = "simplified"
    
    # Print results
    print_results(results)
    
    # Save results if output path specified
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH
    save_results(results, output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[info] Interrupted by user.")
        sys.exit(130)
