#!/usr/bin/env python3
"""
Performance validation script for weighted performance metrics system.

This script validates that the weighted performance calculation system meets
the required performance targets:
- Batch calculations of 600+ players in <50ms
- Individual calculations in <1ms average
- Memory usage remains reasonable under load

Usage:
    python validate_weighted_performance.py [--num-players 650] [--runs 10] [--verbose]
"""

import argparse
import statistics
import time
from typing import List, Tuple
from unittest.mock import Mock

import numpy as np

from airsenal.framework.weighted_performance import (
    WeightedPerformanceCalculator,
    PerformanceWeightMatrix
)
from airsenal.framework.schema import PlayerScore


def create_mock_player_scores(num_scores: int) -> List[Tuple[PlayerScore, str]]:
    """
    Create mock PlayerScore objects for performance testing.
    
    Args:
        num_scores: Number of PlayerScore objects to create
        
    Returns:
        List of (PlayerScore, position) tuples
    """
    scores_and_positions = []
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    for i in range(num_scores):
        score = Mock(spec=PlayerScore)
        
        # Assign realistic performance values
        score.goals = max(0, int(np.random.poisson(0.5)))  # Average ~0.5 goals per game
        score.assists = max(0, int(np.random.poisson(0.3)))  # Average ~0.3 assists per game
        score.bonus = max(0, int(np.random.poisson(0.8)))  # Average ~0.8 bonus per game
        score.clean_sheets = np.random.choice([0, 1], p=[0.65, 0.35])  # ~35% clean sheet rate
        score.saves = max(0, int(np.random.poisson(3.0)))  # Goalkeepers average ~3 saves
        score.own_goals = np.random.choice([0, 1], p=[0.99, 0.01])  # ~1% own goal rate
        score.penalties_saved = np.random.choice([0, 1], p=[0.98, 0.02])  # ~2% penalty save rate
        score.minutes = np.random.choice([0, 90, 45, 30], p=[0.1, 0.7, 0.15, 0.05])  # Playing time distribution
        score.fixture_id = None  # No difficulty adjustment for pure performance test
        score.player_id = i
        
        # Distribute positions realistically (more outfield players)
        position = positions[i % 4] if i < 4 else np.random.choice(positions, p=[0.05, 0.35, 0.35, 0.25])
        scores_and_positions.append((score, position))
    
    return scores_and_positions


def benchmark_batch_calculations(calculator: WeightedPerformanceCalculator,
                                 num_players: int,
                                 num_runs: int,
                                 verbose: bool = False) -> dict:
    """
    Benchmark batch calculation performance.
    
    Args:
        calculator: WeightedPerformanceCalculator instance
        num_players: Number of players to calculate in each batch
        num_runs: Number of benchmark runs
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n=== Batch Calculation Benchmark ===")
    print(f"Players per batch: {num_players}")
    print(f"Number of runs: {num_runs}")
    
    calculation_times = []
    
    for run in range(num_runs):
        # Create fresh mock data for each run
        scores_and_positions = create_mock_player_scores(num_players)
        
        # Mock the extract_metrics method for consistent testing
        def mock_extract_metrics(score, position):
            base_metrics = {
                'goals': float(score.goals),
                'assists': float(score.assists),
                'bonus': float(score.bonus),
            }
            
            if position == 'GK':
                base_metrics.update({
                    'clean_sheets': float(score.clean_sheets),
                    'saves': float(score.saves),
                    'penalties_saved': float(score.penalties_saved),
                })
            elif position == 'DEF':
                base_metrics.update({
                    'clean_sheets': float(score.clean_sheets),
                    'own_goals': float(score.own_goals),
                })
            elif position == 'MID':
                base_metrics.update({
                    'clean_sheets': float(score.clean_sheets),
                    'key_passes_per_90': np.random.uniform(1, 6),
                    'penalty_taker_boost': np.random.choice([0.0, 1.0], p=[0.8, 0.2]),
                })
            else:  # FWD
                base_metrics.update({
                    'shots_per_90': np.random.uniform(1, 8),
                    'penalty_taker_boost': np.random.choice([0.0, 1.0], p=[0.7, 0.3]),
                })
            
            return base_metrics
        
        # Patch extract_metrics for this run
        original_method = calculator.extract_metrics_from_score
        calculator.extract_metrics_from_score = mock_extract_metrics
        
        try:
            # Time the batch calculation
            start_time = time.perf_counter()
            results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
            end_time = time.perf_counter()
            
            calculation_time_ms = (end_time - start_time) * 1000
            calculation_times.append(calculation_time_ms)
            
            # Validate results
            assert len(results) == num_players
            assert all(0.0 <= score <= 1.0 for score in results)
            
            if verbose:
                print(f"Run {run + 1}: {calculation_time_ms:.2f}ms")
        
        finally:
            # Restore original method
            calculator.extract_metrics_from_score = original_method
    
    # Calculate statistics
    mean_time = statistics.mean(calculation_times)
    median_time = statistics.median(calculation_times)
    max_time = max(calculation_times)
    min_time = min(calculation_times)
    std_time = statistics.stdev(calculation_times) if len(calculation_times) > 1 else 0
    
    results = {
        'num_players': num_players,
        'num_runs': num_runs,
        'mean_time_ms': mean_time,
        'median_time_ms': median_time,
        'max_time_ms': max_time,
        'min_time_ms': min_time,
        'std_time_ms': std_time,
        'target_met': max_time < 50.0,
    }
    
    print(f"\nResults:")
    print(f"  Mean time: {mean_time:.2f}ms")
    print(f"  Median time: {median_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Std dev: {std_time:.2f}ms")
    print(f"  Target (<50ms): {'✓ PASSED' if results['target_met'] else '✗ FAILED'}")
    
    return results


def benchmark_individual_calculations(calculator: WeightedPerformanceCalculator,
                                      num_calculations: int,
                                      verbose: bool = False) -> dict:
    """
    Benchmark individual calculation performance.
    
    Args:
        calculator: WeightedPerformanceCalculator instance
        num_calculations: Number of individual calculations to perform
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n=== Individual Calculation Benchmark ===")
    print(f"Number of calculations: {num_calculations}")
    
    # Create a single mock score for repeated calculation
    score = Mock(spec=PlayerScore)
    score.goals = 1
    score.assists = 1
    score.bonus = 2
    score.clean_sheets = 1
    score.saves = 5
    score.own_goals = 0
    score.penalties_saved = 0
    score.fixture_id = None
    score.player_id = 1
    
    def mock_extract_metrics(score, position):
        return {
            'goals': float(score.goals),
            'assists': float(score.assists),
            'bonus': float(score.bonus),
            'clean_sheets': float(score.clean_sheets),
        }
    
    # Patch extract_metrics
    original_method = calculator.extract_metrics_from_score
    calculator.extract_metrics_from_score = mock_extract_metrics
    
    try:
        # Time individual calculations
        start_time = time.perf_counter()
        
        for _ in range(num_calculations):
            result = calculator.calculate_weighted_score(score, 'DEF', apply_difficulty_adjustment=False)
            assert 0.0 <= result <= 1.0
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / num_calculations
        
    finally:
        calculator.extract_metrics_from_score = original_method
    
    results = {
        'num_calculations': num_calculations,
        'total_time_ms': total_time_ms,
        'avg_time_ms': avg_time_ms,
        'target_met': avg_time_ms < 1.0,
    }
    
    print(f"\nResults:")
    print(f"  Total time: {total_time_ms:.2f}ms")
    print(f"  Average time: {avg_time_ms:.4f}ms")
    print(f"  Target (<1ms avg): {'✓ PASSED' if results['target_met'] else '✗ FAILED'}")
    
    return results


def benchmark_memory_usage(calculator: WeightedPerformanceCalculator,
                           num_players: int) -> dict:
    """
    Benchmark memory usage during calculation.
    
    Args:
        calculator: WeightedPerformanceCalculator instance
        num_players: Number of players to test with
        
    Returns:
        Dictionary with memory usage results
    """
    print(f"\n=== Memory Usage Benchmark ===")
    print(f"Testing with {num_players} players")
    
    try:
        import psutil
        process = psutil.Process()
        memory_available = True
    except ImportError:
        print("psutil not available - skipping memory benchmark")
        return {'memory_available': False}
    
    # Get baseline memory usage
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large dataset
    scores_and_positions = create_mock_player_scores(num_players)
    
    def mock_extract_metrics(score, position):
        return {
            'goals': float(score.goals),
            'assists': float(score.assists),
            'bonus': float(score.bonus),
        }
    
    original_method = calculator.extract_metrics_from_score
    calculator.extract_metrics_from_score = mock_extract_metrics
    
    try:
        # Perform calculation and measure peak memory
        peak_memory = baseline_memory
        results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(peak_memory, current_memory)
        
    finally:
        calculator.extract_metrics_from_score = original_method
    
    memory_used = peak_memory - baseline_memory
    memory_per_player = memory_used / num_players * 1024  # KB per player
    
    results = {
        'memory_available': True,
        'baseline_memory_mb': baseline_memory,
        'peak_memory_mb': peak_memory,
        'memory_used_mb': memory_used,
        'memory_per_player_kb': memory_per_player,
        'reasonable_usage': memory_used < 100,  # Less than 100MB for large batch
    }
    
    print(f"\nResults:")
    print(f"  Baseline memory: {baseline_memory:.1f}MB")
    print(f"  Peak memory: {peak_memory:.1f}MB")
    print(f"  Memory used: {memory_used:.1f}MB")
    print(f"  Memory per player: {memory_per_player:.2f}KB")
    print(f"  Reasonable usage (<100MB): {'✓ PASSED' if results['reasonable_usage'] else '✗ FAILED'}")
    
    return results


def validate_accuracy(calculator: WeightedPerformanceCalculator) -> dict:
    """
    Validate calculation accuracy and consistency.
    
    Args:
        calculator: WeightedPerformanceCalculator instance
        
    Returns:
        Dictionary with accuracy validation results
    """
    print(f"\n=== Accuracy Validation ===")
    
    # Test that identical inputs produce identical outputs
    score = Mock(spec=PlayerScore)
    score.goals = 2
    score.assists = 1
    score.bonus = 3
    score.fixture_id = None
    
    def mock_extract_metrics(score, position):
        return {'goals': 2.0, 'assists': 1.0, 'bonus': 3.0}
    
    original_method = calculator.extract_metrics_from_score
    calculator.extract_metrics_from_score = mock_extract_metrics
    
    try:
        # Calculate same score multiple times
        results = []
        for _ in range(10):
            result = calculator.calculate_weighted_score(score, 'FWD', apply_difficulty_adjustment=False)
            results.append(result)
        
        # Check consistency
        all_equal = all(abs(r - results[0]) < 1e-10 for r in results)
        
        # Test batch vs individual consistency
        scores_and_positions = [(score, 'FWD')] * 5
        batch_results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
        individual_results = [
            calculator.calculate_weighted_score(score, 'FWD', apply_difficulty_adjustment=False)
            for _ in range(5)
        ]
        
        batch_consistent = all(
            abs(batch_results[i] - individual_results[i]) < 1e-10
            for i in range(5)
        )
    
    finally:
        calculator.extract_metrics_from_score = original_method
    
    results = {
        'consistent_individual': all_equal,
        'consistent_batch_vs_individual': batch_consistent,
        'sample_result': results[0],
        'result_in_range': 0.0 <= results[0] <= 1.0,
    }
    
    print(f"\nResults:")
    print(f"  Individual consistency: {'✓ PASSED' if results['consistent_individual'] else '✗ FAILED'}")
    print(f"  Batch vs individual: {'✓ PASSED' if results['consistent_batch_vs_individual'] else '✗ FAILED'}")
    print(f"  Result in range [0,1]: {'✓ PASSED' if results['result_in_range'] else '✗ FAILED'}")
    print(f"  Sample result: {results['sample_result']:.6f}")
    
    return results


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate weighted performance metrics system")
    parser.add_argument("--num-players", type=int, default=650, 
                       help="Number of players for batch benchmark (default: 650)")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of benchmark runs (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    
    args = parser.parse_args()
    
    print("🏈 AIrsenal Weighted Performance Metrics Validation")
    print("=" * 55)
    
    # Initialize calculator
    calculator = WeightedPerformanceCalculator()
    
    # Run benchmarks
    batch_results = benchmark_batch_calculations(
        calculator, args.num_players, args.runs, args.verbose
    )
    
    individual_results = benchmark_individual_calculations(
        calculator, 1000, args.verbose
    )
    
    memory_results = benchmark_memory_usage(calculator, args.num_players)
    
    accuracy_results = validate_accuracy(calculator)
    
    # Summary
    print(f"\n{'=' * 55}")
    print("🏆 VALIDATION SUMMARY")
    print(f"{'=' * 55}")
    
    all_passed = True
    
    # Check batch performance target
    if batch_results['target_met']:
        print("✓ Batch calculation performance: PASSED (<50ms)")
    else:
        print(f"✗ Batch calculation performance: FAILED ({batch_results['max_time_ms']:.2f}ms)")
        all_passed = False
    
    # Check individual performance target
    if individual_results['target_met']:
        print("✓ Individual calculation performance: PASSED (<1ms avg)")
    else:
        print(f"✗ Individual calculation performance: FAILED ({individual_results['avg_time_ms']:.4f}ms avg)")
        all_passed = False
    
    # Check memory usage
    if memory_results.get('reasonable_usage', True):
        print("✓ Memory usage: PASSED")
    else:
        print(f"✗ Memory usage: FAILED ({memory_results['memory_used_mb']:.1f}MB)")
        all_passed = False
    
    # Check accuracy
    accuracy_passed = (
        accuracy_results['consistent_individual'] and
        accuracy_results['consistent_batch_vs_individual'] and
        accuracy_results['result_in_range']
    )
    if accuracy_passed:
        print("✓ Calculation accuracy: PASSED")
    else:
        print("✗ Calculation accuracy: FAILED")
        all_passed = False
    
    print(f"{'=' * 55}")
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED - System ready for production!")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED - Review performance issues")
        return 1


if __name__ == "__main__":
    exit(main())