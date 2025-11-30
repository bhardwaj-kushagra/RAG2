#!/usr/bin/env python3
"""
RAGAS Evaluation Module for the Local RAG Pipeline.

This module integrates RAGAS (Retrieval Augmented Generation Assessment) framework
to evaluate the quality of the RAG pipeline. It can evaluate:
- Faithfulness: How well the generated answer is grounded in the context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: How relevant the retrieved contexts are
- Context Recall: How well the contexts cover the ground truth

Usage:
    python src/ragas_evaluator.py --evaluate
    python src/ragas_evaluator.py --evaluate --test-data path/to/data.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
except ImportError as e:
    print(f"[error] Missing RAGAS dependencies: {e}")
    print("        Please run: pip install ragas datasets langchain langchain-openai")
    sys.exit(1)

# Project root for paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvaluationSample:
    """A single sample for RAGAS evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


@dataclass
class EvaluationResult:
    """Results from RAGAS evaluation."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None


def create_sample_test_data() -> List[Dict[str, Any]]:
    """
    Create sample test data for RAGAS evaluation.
    In a real scenario, this would load from a curated test dataset.
    """
    samples = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            "contexts": [
                "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
                "The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide."
            ],
            "ground_truth": "Machine learning is a branch of artificial intelligence that allows computer systems to learn and improve from experience without explicit programming, using data and algorithms to identify patterns and make decisions."
        },
        {
            "question": "What are neural networks?",
            "answer": "Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers that process information using connectionist approaches to computation.",
            "contexts": [
                "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. An artificial neural network is a large number of simple units, the neurons, organized into layers.",
                "A neural network consists of interconnected nodes (neurons) organized in layers. Information flows through the network, with each node processing inputs and passing outputs to the next layer."
            ],
            "ground_truth": "Neural networks are computational systems modeled after the brain's biological neural networks, consisting of interconnected artificial neurons organized in layers that process information."
        },
        {
            "question": "What is RAG in NLP?",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative AI models. It first retrieves relevant documents from a knowledge base and then uses those documents as context to generate more accurate and grounded responses.",
            "contexts": [
                "Retrieval-Augmented Generation (RAG) is a technique for augmenting LLM knowledge with additional data. LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on.",
                "RAG combines two components: a retrieval system that fetches relevant documents from a knowledge base, and a generation model that uses these documents as context to produce responses. This helps ground the model's outputs in specific, retrieved information."
            ],
            "ground_truth": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation, where relevant documents are first retrieved from a knowledge base and then used as context for a language model to generate accurate, grounded responses."
        }
    ]
    return samples


def load_test_data(file_path: Optional[Path] = None) -> List[EvaluationSample]:
    """
    Load test data for RAGAS evaluation.
    
    Args:
        file_path: Optional path to a JSON file containing test data.
                  If None, uses built-in sample data.
    
    Returns:
        List of EvaluationSample objects
    """
    if file_path and file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = create_sample_test_data()
    
    samples = []
    for item in data:
        samples.append(EvaluationSample(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
            ground_truth=item["ground_truth"]
        ))
    return samples


def prepare_dataset_for_ragas(samples: List[EvaluationSample]) -> Dataset:
    """
    Convert evaluation samples to RAGAS-compatible dataset format.
    
    Args:
        samples: List of EvaluationSample objects
    
    Returns:
        HuggingFace Dataset formatted for RAGAS
    """
    data = {
        "user_input": [s.question for s in samples],
        "response": [s.answer for s in samples],
        "retrieved_contexts": [s.contexts for s in samples],
        "reference": [s.ground_truth for s in samples]
    }
    return Dataset.from_dict(data)


def run_ragas_evaluation(
    samples: List[EvaluationSample],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on the provided samples.
    
    Args:
        samples: List of EvaluationSample objects
        metrics: Optional list of metric names to evaluate.
                Defaults to all metrics if None.
    
    Returns:
        Dictionary containing evaluation results
    """
    # Check for OpenAI API key (required for default RAGAS metrics)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("[info] OPENAI_API_KEY not set. Running in mock evaluation mode.")
        print("[info] For full RAGAS evaluation, set OPENAI_API_KEY environment variable.")
        return run_mock_evaluation(samples)
    
    # Prepare dataset
    dataset = prepare_dataset_for_ragas(samples)
    
    # Select metrics
    available_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    
    if metrics:
        selected_metrics = [available_metrics[m] for m in metrics if m in available_metrics]
    else:
        selected_metrics = list(available_metrics.values())
    
    print(f"[ragas] Evaluating {len(samples)} samples with {len(selected_metrics)} metrics...")
    
    try:
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
        )
        
        return {
            "scores": results.scores,
            "individual_results": results.to_pandas().to_dict('records'),
            "num_samples": len(samples),
            "metrics_used": [m.__class__.__name__ for m in selected_metrics]
        }
    except Exception as e:
        print(f"[error] RAGAS evaluation failed: {e}")
        print("[info] Falling back to mock evaluation mode.")
        return run_mock_evaluation(samples)


def run_mock_evaluation(samples: List[EvaluationSample]) -> Dict[str, Any]:
    """
    Run a mock evaluation when OpenAI API key is not available.
    This provides basic text similarity metrics as a fallback.
    
    Args:
        samples: List of EvaluationSample objects
    
    Returns:
        Dictionary containing mock evaluation results
    """
    print("[mock] Running mock evaluation (no LLM API available)...")
    
    results = []
    for sample in samples:
        # Simple word overlap metrics
        answer_words = set(sample.answer.lower().split())
        context_words = set(" ".join(sample.contexts).lower().split())
        ground_truth_words = set(sample.ground_truth.lower().split())
        question_words = set(sample.question.lower().split())
        
        # Calculate basic overlap ratios
        if context_words:
            faithfulness_score = len(answer_words & context_words) / max(len(answer_words), 1)
        else:
            faithfulness_score = 0.0
            
        if ground_truth_words:
            answer_relevancy_score = len(answer_words & ground_truth_words) / max(len(ground_truth_words), 1)
        else:
            answer_relevancy_score = 0.0
            
        if context_words:
            context_precision_score = len(context_words & ground_truth_words) / max(len(context_words), 1)
        else:
            context_precision_score = 0.0
            
        if ground_truth_words:
            context_recall_score = len(context_words & ground_truth_words) / max(len(ground_truth_words), 1)
        else:
            context_recall_score = 0.0
        
        results.append({
            "question": sample.question,
            "faithfulness": round(faithfulness_score, 4),
            "answer_relevancy": round(answer_relevancy_score, 4),
            "context_precision": round(context_precision_score, 4),
            "context_recall": round(context_recall_score, 4)
        })
    
    # Calculate averages
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results) if results else 0
    avg_answer_relevancy = sum(r["answer_relevancy"] for r in results) / len(results) if results else 0
    avg_context_precision = sum(r["context_precision"] for r in results) / len(results) if results else 0
    avg_context_recall = sum(r["context_recall"] for r in results) / len(results) if results else 0
    overall_score = (avg_faithfulness + avg_answer_relevancy + avg_context_precision + avg_context_recall) / 4
    
    return {
        "scores": {
            "faithfulness": round(avg_faithfulness, 4),
            "answer_relevancy": round(avg_answer_relevancy, 4),
            "context_precision": round(avg_context_precision, 4),
            "context_recall": round(avg_context_recall, 4),
            "overall_score": round(overall_score, 4)
        },
        "individual_results": results,
        "num_samples": len(samples),
        "metrics_used": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "mode": "mock (word overlap)"
    }


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[ragas] Results saved to {output_path}")


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 50)
    print(" RAGAS Evaluation Results")
    print("=" * 50)
    
    mode = results.get("mode", "full RAGAS")
    print(f"\nEvaluation Mode: {mode}")
    print(f"Number of Samples: {results['num_samples']}")
    print(f"Metrics Used: {', '.join(results['metrics_used'])}")
    
    print("\n--- Average Scores ---")
    scores = results.get("scores", {})
    for metric, score in scores.items():
        if score is not None:
            print(f"  {metric}: {score:.4f}")
    
    print("\n--- Individual Results ---")
    individual = results.get("individual_results", [])
    for i, result in enumerate(individual, 1):
        print(f"\n  Sample {i}:")
        question = result.get("question", result.get("user_input", "N/A"))
        if len(question) > 60:
            question = question[:60] + "..."
        print(f"    Question: {question}")
        for key, value in result.items():
            if key not in ["question", "user_input", "response", "reference", "retrieved_contexts"]:
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
    
    print("\n" + "=" * 50)


def main() -> None:
    """Main entry point for RAGAS evaluation."""
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for Local RAG Pipeline"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAGAS evaluation"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to JSON file with test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        default=None,
        help="Specific metrics to evaluate"
    )
    
    args = parser.parse_args()
    
    if not args.evaluate:
        parser.print_help()
        return
    
    print("=" * 50)
    print(" RAGAS Evaluator - RAG Pipeline Quality Assessment")
    print("=" * 50)
    
    # Load test data
    test_data_path = Path(args.test_data) if args.test_data else None
    samples = load_test_data(test_data_path)
    print(f"[ragas] Loaded {len(samples)} test samples")
    
    # Run evaluation
    results = run_ragas_evaluation(samples, args.metrics)
    
    # Print results
    print_results(results)
    
    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        save_results(results, output_path)
    else:
        # Default output path
        output_path = PROJECT_ROOT / "evaluation_results.json"
        save_results(results, output_path)


if __name__ == "__main__":
    main()
