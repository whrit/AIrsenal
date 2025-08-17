#!/usr/bin/env python3
"""
Standalone test script for weighted performance metrics.

This script tests the weighted performance system without relying on 
the full AIrsenal test infrastructure, focusing on the core functionality.
"""

import time
from unittest.mock import Mock
import numpy as np

# Import the weighted performance modules
from airsenal.framework.weighted_performance import (
    PerformanceWeightMatrix,
    WeightedPerformanceCalculator,
    calculate_weighted_performance,
    calculate_batch_weighted_performance,
)


def test_performance_weight_matrix():
    """Test PerformanceWeightMatrix functionality."""
    print("Testing PerformanceWeightMatrix...")
    
    matrix = PerformanceWeightMatrix()
    
    # Test position weight retrieval
    gk_weights = matrix.get_weights_for_position('GK')
    assert 'clean_sheets' in gk_weights
    assert 'saves' in gk_weights
    
    def_weights = matrix.get_weights_for_position('DEF')
    assert 'clean_sheets' in def_weights
    assert 'goals' in def_weights
    
    # Test metric normalization
    assert matrix.normalize_metric('goals', 0) == 0.0
    assert matrix.normalize_metric('goals', 2) == 0.5
    assert matrix.normalize_metric('goals', 4) == 1.0
    assert matrix.normalize_metric('clean_sheets', 1) == 1.0
    
    print("✓ PerformanceWeightMatrix tests passed")


def test_weighted_performance_calculator():
    """Test WeightedPerformanceCalculator functionality."""
    print("Testing WeightedPerformanceCalculator...")
    
    calculator = WeightedPerformanceCalculator()
    
    # Create mock PlayerScore
    score = Mock()
    score.goals = 2
    score.assists = 1
    score.bonus = 3
    score.clean_sheets = 1
    score.saves = 5
    score.own_goals = 0
    score.penalties_saved = 0
    score.minutes = 90
    score.player_id = 1
    score.fixture_id = None
    
    # Test metric extraction for different positions
    gk_metrics = calculator.extract_metrics_from_score(score, 'GK')
    assert 'goals' in gk_metrics
    assert 'clean_sheets' in gk_metrics
    assert 'saves' in gk_metrics
    
    def_metrics = calculator.extract_metrics_from_score(score, 'DEF')
    assert 'goals' in def_metrics
    assert 'clean_sheets' in def_metrics
    assert 'own_goals' in def_metrics
    
    # Test single score calculation
    weighted_score = calculator.calculate_weighted_score(score, 'DEF', apply_difficulty_adjustment=False)
    assert 0.0 <= weighted_score <= 1.0
    
    print("✓ WeightedPerformanceCalculator tests passed")


def test_batch_vs_individual_consistency():
    """Test that batch and individual calculations produce identical results."""
    print("Testing batch vs individual consistency...")
    
    calculator = WeightedPerformanceCalculator()
    
    # Create multiple test scores
    scores_and_positions = []
    for i in range(10):
        score = Mock()
        score.goals = i % 3
        score.assists = (i + 1) % 2
        score.bonus = i % 4
        score.clean_sheets = i % 2
        score.saves = (i * 2) % 6
        score.own_goals = 0
        score.penalties_saved = 0
        score.minutes = 90
        score.player_id = i
        score.fixture_id = None
        
        position = ['GK', 'DEF', 'MID', 'FWD'][i % 4]
        scores_and_positions.append((score, position))
    
    # Calculate using batch method
    batch_results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
    
    # Calculate using individual method
    individual_results = []
    for score, position in scores_and_positions:
        result = calculator.calculate_weighted_score(score, position, apply_difficulty_adjustment=False)
        individual_results.append(result)
    
    # Check consistency (allow small floating point differences)
    for i, (batch_result, individual_result) in enumerate(zip(batch_results, individual_results)):
        diff = abs(batch_result - individual_result)
        assert diff < 1e-10, f"Inconsistency at index {i}: batch={batch_result}, individual={individual_result}, diff={diff}"
    
    print("✓ Batch vs individual consistency tests passed")


def test_performance_benchmarks():
    """Test that performance requirements are met."""
    print("Testing performance benchmarks...")
    
    calculator = WeightedPerformanceCalculator()
    
    # Create large batch for performance testing
    num_scores = 650
    scores_and_positions = []
    
    for i in range(num_scores):
        score = Mock()
        score.goals = i % 4
        score.assists = i % 3
        score.bonus = i % 4
        score.clean_sheets = i % 2
        score.saves = i % 8
        score.own_goals = 0
        score.penalties_saved = 0
        score.minutes = 90
        score.player_id = i
        score.fixture_id = None
        
        position = ['GK', 'DEF', 'MID', 'FWD'][i % 4]
        scores_and_positions.append((score, position))
    
    # Time batch calculation
    start_time = time.perf_counter()
    results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
    end_time = time.perf_counter()
    
    batch_time_ms = (end_time - start_time) * 1000
    
    # Verify results
    assert len(results) == num_scores
    assert all(0.0 <= score <= 1.0 for score in results)
    
    # Check performance requirement
    assert batch_time_ms < 50.0, f"Batch calculation took {batch_time_ms:.2f}ms, exceeds 50ms target"
    
    # Test individual calculation performance
    score = scores_and_positions[0][0]
    start_time = time.perf_counter()
    for _ in range(1000):
        calculator.calculate_weighted_score(score, 'DEF', apply_difficulty_adjustment=False)
    end_time = time.perf_counter()
    
    avg_individual_time_ms = ((end_time - start_time) / 1000) * 1000
    assert avg_individual_time_ms < 1.0, f"Individual calculation averaging {avg_individual_time_ms:.4f}ms is too slow"
    
    print(f"✓ Performance benchmarks passed (batch: {batch_time_ms:.2f}ms, individual avg: {avg_individual_time_ms:.4f}ms)")


def test_convenience_functions():
    """Test convenience functions."""
    print("Testing convenience functions...")
    
    score = Mock()
    score.goals = 1
    score.assists = 1
    score.bonus = 2
    score.clean_sheets = 1
    score.saves = 3
    score.own_goals = 0
    score.penalties_saved = 0
    score.minutes = 90
    score.player_id = 1
    score.fixture_id = None
    
    # Test single calculation convenience function
    result = calculate_weighted_performance(score, 'DEF')
    assert 0.0 <= result <= 1.0
    
    # Test batch calculation convenience function
    scores_and_positions = [(score, 'DEF'), (score, 'MID')]
    results = calculate_batch_weighted_performance(scores_and_positions)
    assert len(results) == 2
    assert all(0.0 <= score <= 1.0 for score in results)
    
    print("✓ Convenience function tests passed")


def test_config_system():
    """Test configuration system."""
    print("Testing configuration system...")
    
    try:
        from airsenal.framework.weighted_performance_config import (
            validate_weight_matrix,
            PerformanceWeightMatrix
        )
        
        matrix = PerformanceWeightMatrix()
        validation = validate_weight_matrix(matrix)
        
        assert validation['valid'] is True
        assert 'position_weight_sums' in validation
        assert len(validation['position_weight_sums']) == 4
        
        print("✓ Configuration system tests passed")
        
    except ImportError as e:
        print(f"⚠ Configuration system tests skipped (PyYAML not available): {e}")


def main():
    """Run all standalone tests."""
    print("🏈 Running Weighted Performance Metrics Standalone Tests")
    print("=" * 60)
    
    tests = [
        test_performance_weight_matrix,
        test_weighted_performance_calculator,
        test_batch_vs_individual_consistency,
        test_performance_benchmarks,
        test_convenience_functions,
        test_config_system,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Weighted Performance System is working correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review implementation")
        return 1


if __name__ == "__main__":
    exit(main())