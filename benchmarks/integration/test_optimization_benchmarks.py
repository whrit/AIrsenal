"""
Optimization Performance Benchmarks

Tests performance of optimization algorithms including:
- Squad optimization latency
- Transfer optimization performance
- Genetic algorithm convergence speed
- Memory usage during optimization
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any

from benchmarks.utils import BenchmarkRunner


class OptimizationBenchmarks:
    """Optimization performance benchmark suite."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.config = runner.config
        
        # Test data
        self.test_players = self._generate_test_players()
        self.test_squad = self._generate_test_squad()
    
    def _generate_test_players(self) -> List[Dict[str, Any]]:
        """Generate test player data for optimization."""
        positions = ["GK", "DEF", "MID", "FWD"]
        return [
            {
                "id": i,
                "name": f"Player {i}",
                "position": positions[i % 4],
                "price": np.random.uniform(4.0, 13.0),
                "predicted_points": np.random.uniform(0, 15),
                "team": np.random.randint(1, 21)
            }
            for i in range(500)  # Large pool for optimization
        ]
    
    def _generate_test_squad(self) -> Dict[str, Any]:
        """Generate test squad for transfer optimization."""
        return {
            "players": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "budget": 100.0,
            "free_transfers": 1
        }
    
    def run_all(self):
        """Run all optimization benchmarks."""
        self.benchmark_squad_optimization()
        self.benchmark_transfer_optimization()
        self.benchmark_genetic_algorithm()
        self.benchmark_optimization_memory()
    
    @pytest.mark.benchmark(group="optimization")
    def benchmark_squad_optimization(self):
        """Benchmark squad optimization performance."""
        with self.runner.benchmark_context("optimization_squad_build"):
            # Simulate squad building optimization
            best_squad = self._simulate_squad_optimization(
                players=self.test_players,
                budget=100.0,
                iterations=100
            )
    
    @pytest.mark.benchmark(group="optimization")
    def benchmark_transfer_optimization(self):
        """Benchmark transfer optimization performance."""
        for num_transfers in [1, 2, 3]:
            with self.runner.benchmark_context(f"optimization_transfers_{num_transfers}"):
                # Simulate transfer optimization
                transfers = self._simulate_transfer_optimization(
                    current_squad=self.test_squad,
                    available_players=self.test_players,
                    max_transfers=num_transfers,
                    iterations=50
                )
    
    @pytest.mark.benchmark(group="optimization")
    def benchmark_genetic_algorithm(self):
        """Benchmark genetic algorithm performance."""
        population_sizes = [50, 100, 200]
        
        for pop_size in population_sizes:
            with self.runner.benchmark_context(f"optimization_ga_pop_{pop_size}"):
                # Simulate genetic algorithm
                result = self._simulate_genetic_algorithm(
                    population_size=pop_size,
                    generations=20,
                    mutation_rate=0.1
                )
    
    @pytest.mark.benchmark(group="optimization")
    def benchmark_optimization_memory(self):
        """Benchmark memory usage during optimization."""
        import psutil
        process = psutil.Process()
        
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with self.runner.benchmark_context("optimization_memory_usage"):
            # Run memory-intensive optimization
            for _ in range(5):
                squad = self._simulate_squad_optimization(
                    players=self.test_players,
                    budget=100.0,
                    iterations=200
                )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - start_memory
        
        # Log memory usage
        self.runner.results[-1]["memory_used_mb"] = memory_used
        self.runner.results[-1]["peak_memory_mb"] = peak_memory
    
    def _simulate_squad_optimization(self, players: List[Dict], budget: float, iterations: int) -> Dict[str, Any]:
        """Simulate squad optimization algorithm."""
        # Simulate optimization time based on problem complexity
        time.sleep(iterations * 0.001)  # 1ms per iteration
        
        # Mock optimization result
        return {
            "total_predicted_points": np.random.uniform(500, 600),
            "total_cost": budget,
            "iterations": iterations,
            "convergence_time": iterations * 0.001
        }
    
    def _simulate_transfer_optimization(self, current_squad: Dict, available_players: List[Dict], 
                                      max_transfers: int, iterations: int) -> List[Dict]:
        """Simulate transfer optimization algorithm."""
        # Simulate optimization time
        time.sleep(max_transfers * iterations * 0.0005)  # 0.5ms per transfer per iteration
        
        # Mock transfer results
        return [
            {
                "player_out": np.random.randint(1, 16),
                "player_in": np.random.randint(1, 501),
                "cost": np.random.uniform(-2.0, 2.0),
                "point_gain": np.random.uniform(-5, 15)
            }
            for _ in range(max_transfers)
        ]
    
    def _simulate_genetic_algorithm(self, population_size: int, generations: int, mutation_rate: float) -> Dict[str, Any]:
        """Simulate genetic algorithm optimization."""
        # Simulate GA time complexity
        time.sleep(population_size * generations * 0.00001)  # 0.01ms per individual per generation
        
        return {
            "best_fitness": np.random.uniform(500, 600),
            "generations": generations,
            "population_size": population_size,
            "convergence_generation": np.random.randint(5, generations)
        }


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="optimization_single")
def test_squad_optimization(benchmark):
    """Pytest-benchmark test for squad optimization."""
    
    # Generate test data
    players = [
        {
            "id": i, "position": ["GK", "DEF", "MID", "FWD"][i % 4],
            "price": np.random.uniform(4.0, 13.0),
            "predicted_points": np.random.uniform(0, 15)
        }
        for i in range(200)
    ]
    
    def optimize_squad():
        # Simulate optimization
        time.sleep(0.1)  # 100ms
        return {"total_points": np.random.uniform(500, 600)}
    
    result = benchmark(optimize_squad)