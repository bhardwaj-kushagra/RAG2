#!/usr/bin/env python3
"""
Tests for RAGAS Evaluation Module.

This module contains tests to verify the RAGAS integration works correctly
with the RAG pipeline.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragas_evaluator import (
    EvaluationSample,
    create_sample_test_data,
    load_test_data,
    prepare_dataset_for_ragas,
    run_mock_evaluation,
)


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""
    
    def test_creation(self):
        """Test creating an EvaluationSample."""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["Context 1", "Context 2"],
            ground_truth="AI stands for artificial intelligence."
        )
        
        assert sample.question == "What is AI?"
        assert sample.answer == "AI is artificial intelligence."
        assert len(sample.contexts) == 2
        assert sample.ground_truth == "AI stands for artificial intelligence."


class TestSampleTestData:
    """Tests for sample test data generation."""
    
    def test_create_sample_test_data_not_empty(self):
        """Test that sample test data is not empty."""
        data = create_sample_test_data()
        assert len(data) > 0
    
    def test_create_sample_test_data_structure(self):
        """Test that sample test data has correct structure."""
        data = create_sample_test_data()
        
        for item in data:
            assert "question" in item
            assert "answer" in item
            assert "contexts" in item
            assert "ground_truth" in item
            assert isinstance(item["contexts"], list)
            assert len(item["contexts"]) > 0
    
    def test_create_sample_test_data_content(self):
        """Test that sample test data has meaningful content."""
        data = create_sample_test_data()
        
        for item in data:
            assert len(item["question"]) > 5
            assert len(item["answer"]) > 10
            assert len(item["ground_truth"]) > 10
            for context in item["contexts"]:
                assert len(context) > 10


class TestLoadTestData:
    """Tests for loading test data."""
    
    def test_load_test_data_default(self):
        """Test loading default test data without file."""
        samples = load_test_data()
        
        assert len(samples) > 0
        assert all(isinstance(s, EvaluationSample) for s in samples)
    
    def test_load_test_data_from_file(self):
        """Test loading test data from a JSON file."""
        test_data = [
            {
                "question": "Test question?",
                "answer": "Test answer.",
                "contexts": ["Test context 1", "Test context 2"],
                "ground_truth": "Test ground truth."
            }
        ]
        
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                json.dump(test_data, f)
                temp_path = Path(f.name)
            
            samples = load_test_data(temp_path)
            assert len(samples) == 1
            assert samples[0].question == "Test question?"
            assert samples[0].answer == "Test answer."
            assert samples[0].contexts == ["Test context 1", "Test context 2"]
            assert samples[0].ground_truth == "Test ground truth."
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()
    
    def test_load_test_data_nonexistent_file(self):
        """Test loading with non-existent file falls back to default."""
        samples = load_test_data(Path("/nonexistent/path/file.json"))
        
        # Should fall back to default sample data
        assert len(samples) > 0


class TestPrepareDataset:
    """Tests for preparing RAGAS-compatible dataset."""
    
    def test_prepare_dataset_for_ragas(self):
        """Test preparing dataset for RAGAS evaluation."""
        samples = [
            EvaluationSample(
                question="Q1",
                answer="A1",
                contexts=["C1"],
                ground_truth="GT1"
            ),
            EvaluationSample(
                question="Q2",
                answer="A2",
                contexts=["C2", "C3"],
                ground_truth="GT2"
            )
        ]
        
        dataset = prepare_dataset_for_ragas(samples)
        
        assert len(dataset) == 2
        assert "user_input" in dataset.column_names
        assert "response" in dataset.column_names
        assert "retrieved_contexts" in dataset.column_names
        assert "reference" in dataset.column_names
    
    def test_prepare_dataset_content(self):
        """Test that dataset content matches input samples."""
        samples = [
            EvaluationSample(
                question="What is Python?",
                answer="Python is a programming language.",
                contexts=["Python is versatile.", "Python is popular."],
                ground_truth="Python is a high-level programming language."
            )
        ]
        
        dataset = prepare_dataset_for_ragas(samples)
        
        assert dataset["user_input"][0] == "What is Python?"
        assert dataset["response"][0] == "Python is a programming language."
        assert dataset["retrieved_contexts"][0] == ["Python is versatile.", "Python is popular."]
        assert dataset["reference"][0] == "Python is a high-level programming language."


class TestMockEvaluation:
    """Tests for mock evaluation functionality."""
    
    def test_run_mock_evaluation_returns_results(self):
        """Test that mock evaluation returns results."""
        samples = [
            EvaluationSample(
                question="What is AI?",
                answer="AI is artificial intelligence that enables machines to learn.",
                contexts=["Artificial intelligence (AI) is the simulation of human intelligence by machines."],
                ground_truth="AI is artificial intelligence."
            )
        ]
        
        results = run_mock_evaluation(samples)
        
        assert "scores" in results
        assert "individual_results" in results
        assert "num_samples" in results
        assert "metrics_used" in results
        assert "mode" in results
    
    def test_run_mock_evaluation_scores_structure(self):
        """Test mock evaluation scores structure."""
        samples = load_test_data()
        results = run_mock_evaluation(samples)
        
        scores = results["scores"]
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores
        assert "context_recall" in scores
        assert "overall_score" in scores
    
    def test_run_mock_evaluation_scores_range(self):
        """Test that mock evaluation scores are in valid range [0, 1]."""
        samples = load_test_data()
        results = run_mock_evaluation(samples)
        
        scores = results["scores"]
        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{metric} score {score} out of range"
    
    def test_run_mock_evaluation_individual_results(self):
        """Test individual results from mock evaluation."""
        samples = load_test_data()
        results = run_mock_evaluation(samples)
        
        individual = results["individual_results"]
        assert len(individual) == len(samples)
        
        for result in individual:
            assert "question" in result
            assert "faithfulness" in result
            assert "answer_relevancy" in result
            assert "context_precision" in result
            assert "context_recall" in result
    
    def test_run_mock_evaluation_num_samples(self):
        """Test that num_samples matches input."""
        samples = load_test_data()
        results = run_mock_evaluation(samples)
        
        assert results["num_samples"] == len(samples)
    
    def test_run_mock_evaluation_empty_contexts(self):
        """Test mock evaluation with empty contexts."""
        samples = [
            EvaluationSample(
                question="Test?",
                answer="Answer",
                contexts=[],
                ground_truth="Ground truth"
            )
        ]
        
        results = run_mock_evaluation(samples)
        
        # Should not crash, scores may be 0
        assert results["num_samples"] == 1
        assert results["scores"]["faithfulness"] == 0.0


class TestIntegration:
    """Integration tests for the full evaluation flow."""
    
    def test_full_evaluation_flow(self):
        """Test the complete evaluation flow."""
        # Create samples
        samples = load_test_data()
        
        # Prepare dataset
        dataset = prepare_dataset_for_ragas(samples)
        
        # Run mock evaluation (since we likely don't have API key in tests)
        results = run_mock_evaluation(samples)
        
        # Verify results
        assert results is not None
        assert results["num_samples"] == len(samples)
        assert all(
            metric in results["scores"]
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        )
    
    def test_results_serialization(self):
        """Test that results can be serialized to JSON."""
        samples = load_test_data()
        results = run_mock_evaluation(samples)
        
        # Should be JSON serializable
        json_str = json.dumps(results)
        parsed = json.loads(json_str)
        
        assert parsed["num_samples"] == results["num_samples"]
        assert parsed["scores"] == results["scores"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
