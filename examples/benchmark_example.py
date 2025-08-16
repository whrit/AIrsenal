#!/usr/bin/env python3
"""
AIrsenal Performance Benchmarking Example

This script demonstrates how to use the AIrsenal performance benchmarking framework
to measure and monitor system performance.

Usage:
    python examples/benchmark_example.py
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.utils import BenchmarkRunner
from benchmarks.configs.benchmark_config import get_config


def run_example_benchmarks():
    """Run a subset of benchmarks as an example."""
    
    print("=== AIrsenal Performance Benchmarking Example ===\n")
    
    # Initialize benchmark runner
    config = get_config("local")
    runner = BenchmarkRunner(config)
    
    try:
        print("Setting up benchmark environment...")
        runner.setup()
        
        # Example 1: Simple Feature Store benchmark
        print("\n1. Feature Store Performance Test")
        print("-" * 40)
        
        if runner.feature_store:
            with runner.benchmark_context("example_feature_retrieval"):
                # Simulate feature retrieval
                features = runner.feature_store.get_features(
                    entity_type="player",
                    entity_ids=[1, 2, 3, 4, 5],
                    feature_names=["rolling_goals_5"],
                    season="2425",
                    gameweek=10
                )
            print("✅ Feature store benchmark completed")
        else:
            print("⚠️ Feature store not available")
        
        # Example 2: Redis Cache benchmark
        print("\n2. Redis Cache Performance Test")
        print("-" * 40)
        
        if runner.redis_cache:
            test_data = {"test": "data", "timestamp": time.time()}
            
            with runner.benchmark_context("example_cache_operations"):
                # Test SET operation
                runner.redis_cache.set("benchmark:example", test_data, ttl=60)
                
                # Test GET operation
                retrieved = runner.redis_cache.get("benchmark:example")
            
            print("✅ Redis cache benchmark completed")
        else:
            print("⚠️ Redis cache not available")
        
        # Example 3: Simple computation benchmark
        print("\n3. Computation Performance Test")
        print("-" * 40)
        
        with runner.benchmark_context("example_computation"):
            # Simulate computation
            import numpy as np
            
            # Generate test data
            data = np.random.uniform(0, 100, (1000, 10))
            
            # Perform computations
            rolling_mean = np.mean(data, axis=1)
            rolling_std = np.std(data, axis=1)
            
            # Simulate processing time
            result = np.column_stack([rolling_mean, rolling_std])
        
        print("✅ Computation benchmark completed")
        
        # Example 4: Database query simulation
        print("\n4. Database Query Performance Test")
        print("-" * 40)
        
        if runner.session:
            try:
                with runner.benchmark_context("example_database_query"):
                    # Simple database query
                    from airsenal.framework.schema import Player
                    players = runner.session.query(Player).limit(10).all()
                
                print("✅ Database benchmark completed")
            except Exception as e:
                print(f"⚠️ Database benchmark failed: {e}")
        else:
            print("⚠️ Database session not available")
        
        # Generate summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        
        if runner.results:
            for result in runner.results:
                test_name = result["test_name"]
                duration = result["duration"]
                memory_mb = result.get("memory_delta_mb", 0)
                
                print(f"{test_name:30} | {duration:8.4f}s | {memory_mb:6.1f}MB")
            
            # Calculate totals
            total_time = sum(r["duration"] for r in runner.results)
            total_memory = sum(r.get("memory_delta_mb", 0) for r in runner.results)
            
            print("-" * 50)
            print(f"{'TOTAL':30} | {total_time:8.4f}s | {total_memory:6.1f}MB")
            
            # Save results
            results_file = runner.save_results("example_benchmark_results.json")
            print(f"\nResults saved to: {results_file}")
            
            # Check performance thresholds
            from benchmarks.utils import check_thresholds
            threshold_check = check_thresholds(runner.results, config)
            
            if threshold_check["pass"]:
                print("✅ All benchmarks passed performance thresholds")
            else:
                print(f"⚠️ {threshold_check['total_violations']} threshold violations detected")
                for violation in threshold_check["violations"]:
                    print(f"   - {violation['test_name']}: {violation['violation_percent']:.1f}% over threshold")
        else:
            print("No benchmark results to display")
    
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        runner.teardown()


def demonstrate_dashboard():
    """Demonstrate performance dashboard generation."""
    print("\n=== Performance Dashboard Example ===\n")
    
    try:
        from airsenal.scripts.performance_dashboard import PerformanceDashboard
        
        # Create dashboard
        dashboard = PerformanceDashboard(
            data_dir="benchmarks/reports",
            port=8080
        )
        
        # Generate dashboard
        dashboard_file = dashboard.save_dashboard()
        print(f"Performance dashboard generated: {dashboard_file}")
        print("Open this file in a web browser to view performance metrics")
        
    except Exception as e:
        print(f"Dashboard generation failed: {e}")


def demonstrate_load_testing():
    """Demonstrate load testing setup."""
    print("\n=== Load Testing Example ===\n")
    
    print("Load testing requires a running AIrsenal API server.")
    print("To run load tests:")
    print()
    print("1. Start the AIrsenal API:")
    print("   cd airsenal/api && python app.py")
    print()
    print("2. Run prediction load test:")
    print("   locust -f benchmarks/load_testing/prediction_load_test.py --host=http://localhost:5000")
    print()
    print("3. Or run programmatically:")
    print("   from benchmarks.load_testing.prediction_load_test import run_prediction_load_test")
    print("   run_prediction_load_test()")


def main():
    """Main example function."""
    print("AIrsenal Performance Benchmarking Framework")
    print("==========================================\n")
    
    print("This example demonstrates the key features of the benchmarking framework:\n")
    print("1. Component-level performance testing")
    print("2. System resource monitoring")
    print("3. Performance threshold checking")
    print("4. Results reporting and analysis")
    print("5. Dashboard generation")
    print()
    
    # Run the example benchmarks
    run_example_benchmarks()
    
    # Demonstrate dashboard
    demonstrate_dashboard()
    
    # Show load testing info
    demonstrate_load_testing()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("To run the full benchmark suite:")
    print("  airsenal_run_benchmarks --suite framework")
    print()
    print("To start performance monitoring:")
    print("  airsenal_performance_dashboard --monitor")
    print()
    print("To run pytest benchmarks:")
    print("  pytest benchmarks/ --benchmark-only")
    print()
    print("For more information, see: benchmarks/README.md")


if __name__ == "__main__":
    main()