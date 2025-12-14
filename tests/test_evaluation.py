#!/usr/bin/env python3
"""
Tests for the RAGAS Evaluation Module

Run with: python -m pytest tests/test_evaluation.py -v
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np

from evaluation import (
    simple_text_embedding,
    batch_simple_embedding,
    SimpleEmbedder,
    cosine_similarity,
    EvaluationSample,
    get_sample_evaluation_data,
    run_evaluation_simple,
)


class TestSimpleTextEmbedding:
    """Tests for the simple text embedding function."""
    
    def test_returns_correct_dimension(self):
        """Test that embedding has correct dimension."""
        embedding = simple_text_embedding("hello world", dim=128)
        assert embedding.shape == (128,)
        
    def test_different_dimensions(self):
        """Test different embedding dimensions."""
        for dim in [64, 128, 256]:
            embedding = simple_text_embedding("test text", dim=dim)
            assert embedding.shape == (dim,)
    
    def test_deterministic(self):
        """Test that same input produces same output."""
        emb1 = simple_text_embedding("hello world", dim=128)
        emb2 = simple_text_embedding("hello world", dim=128)
        np.testing.assert_array_equal(emb1, emb2)
    
    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        emb1 = simple_text_embedding("hello world", dim=128)
        emb2 = simple_text_embedding("goodbye world", dim=128)
        assert not np.allclose(emb1, emb2)
    
    def test_normalized(self):
        """Test that embeddings are normalized."""
        embedding = simple_text_embedding("test text", dim=128)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=0.01)


class TestBatchSimpleEmbedding:
    """Tests for batch embedding function."""
    
    def test_returns_correct_shape(self):
        """Test that batch embedding returns correct shape."""
        texts = ["hello", "world", "test"]
        embeddings = batch_simple_embedding(texts, dim=128)
        assert embeddings.shape == (3, 128)
    
    def test_consistent_with_single(self):
        """Test that batch embedding is consistent with single embedding."""
        text = "hello world"
        single = simple_text_embedding(text, dim=128)
        batch = batch_simple_embedding([text], dim=128)[0]
        np.testing.assert_array_equal(single, batch)


class TestSimpleEmbedder:
    """Tests for the SimpleEmbedder class."""
    
    def test_encode_returns_correct_shape(self):
        """Test that encode returns correct shape."""
        embedder = SimpleEmbedder(dim=128)
        texts = ["hello", "world"]
        embeddings = embedder.encode(texts)
        assert embeddings.shape == (2, 128)
    
    def test_normalized_by_default(self):
        """Test that embeddings are normalized by default."""
        embedder = SimpleEmbedder(dim=128)
        embeddings = embedder.encode(["test text"])
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, atol=0.01)


class TestCosineSimilarity:
    """Tests for cosine similarity function."""
    
    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1."""
        vec = np.array([1.0, 0.0, 0.0])
        assert np.isclose(cosine_similarity(vec, vec), 1.0)
    
    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert np.isclose(cosine_similarity(vec1, vec2), 0.0)
    
    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity -1."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        assert np.isclose(cosine_similarity(vec1, vec2), -1.0)
    
    def test_handles_2d_arrays(self):
        """Test that it handles 2D arrays correctly."""
        vec1 = np.array([[1.0, 0.0, 0.0]])
        vec2 = np.array([[1.0, 0.0, 0.0]])
        assert np.isclose(cosine_similarity(vec1, vec2), 1.0)


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""
    
    def test_creation(self):
        """Test that EvaluationSample can be created."""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            ground_truth="AI is artificial intelligence."
        )
        assert sample.question == "What is AI?"
        assert sample.answer == "AI is artificial intelligence."
        assert len(sample.contexts) == 1
        assert sample.ground_truth == "AI is artificial intelligence."


class TestGetSampleEvaluationData:
    """Tests for sample data function."""
    
    def test_returns_list(self):
        """Test that it returns a list."""
        samples = get_sample_evaluation_data()
        assert isinstance(samples, list)
    
    def test_returns_evaluation_samples(self):
        """Test that all items are EvaluationSample."""
        samples = get_sample_evaluation_data()
        for sample in samples:
            assert isinstance(sample, EvaluationSample)
    
    def test_samples_have_all_fields(self):
        """Test that samples have all required fields."""
        samples = get_sample_evaluation_data()
        for sample in samples:
            assert sample.question
            assert sample.answer
            assert sample.contexts
            assert sample.ground_truth


class TestRunEvaluationSimple:
    """Tests for the simplified evaluation function."""
    
    def test_returns_dict(self):
        """Test that evaluation returns a dictionary."""
        results = run_evaluation_simple()
        assert isinstance(results, dict)
    
    def test_has_required_metrics(self):
        """Test that results contain all required metrics."""
        results = run_evaluation_simple()
        required_keys = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "num_samples",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_metrics_are_float(self):
        """Test that metrics are floats."""
        results = run_evaluation_simple()
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            assert isinstance(results[key], float), f"{key} should be float"
    
    def test_with_custom_samples(self):
        """Test evaluation with custom samples."""
        samples = [
            EvaluationSample(
                question="What is 2+2?",
                answer="2+2 equals 4.",
                contexts=["Basic arithmetic: 2+2=4"],
                ground_truth="The answer is 4."
            )
        ]
        results = run_evaluation_simple(samples)
        assert results["num_samples"] == 1
    
    def test_individual_scores_present(self):
        """Test that individual scores are present."""
        results = run_evaluation_simple()
        assert "individual_scores" in results
        ind = results["individual_scores"]
        assert "questions" in ind
        assert "faithfulness" in ind
        assert "answer_relevancy" in ind
        assert "context_precision" in ind
        assert "context_recall" in ind


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
