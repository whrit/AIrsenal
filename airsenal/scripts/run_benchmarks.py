"""
AIrsenal Performance Benchmark Runner

This script runs comprehensive performance benchmarks for all AIrsenal components
including the new Sprint 00 framework components.

Usage:
    airsenal_run_benchmarks --suite framework
    airsenal_run_benchmarks --suite integration --env ci
    airsenal_run_benchmarks --suite all --compare-baseline
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.configs.benchmark_config import get_config
from benchmarks.utils import BenchmarkRunner, check_thresholds, compare_to_baseline


def run_framework_benchmarks(runner: BenchmarkRunner) -> None:
    """Run benchmarks for framework components."""
    print("=== Framework Component Benchmarks ===")

    # Import benchmark modules
    try:
        from benchmarks.framework.test_base_models_benchmarks import (
            BaseModelsBenchmarks,
        )
        from benchmarks.framework.test_database_benchmarks import DatabaseBenchmarks
        from benchmarks.framework.test_feature_store_benchmarks import (
            FeatureStoreBenchmarks,
        )
        from benchmarks.framework.test_model_versioning_benchmarks import (
            ModelVersioningBenchmarks,
        )
        from benchmarks.framework.test_redis_cache_benchmarks import (
            RedisCacheBenchmarks,
        )

        # Run feature store benchmarks
        if runner.feature_store:
            print("\n--- Feature Store Benchmarks ---")
            fs_bench = FeatureStoreBenchmarks(runner)
            fs_bench.run_all()

        # Run Redis cache benchmarks
        if runner.redis_cache:
            print("\n--- Redis Cache Benchmarks ---")
            redis_bench = RedisCacheBenchmarks(runner)
            redis_bench.run_all()

        # Run model versioning benchmarks
        if runner.model_manager:
            print("\n--- Model Versioning Benchmarks ---")
            mv_bench = ModelVersioningBenchmarks(runner)
            mv_bench.run_all()

        # Run base models benchmarks
        print("\n--- Base Models Benchmarks ---")
        bm_bench = BaseModelsBenchmarks(runner)
        bm_bench.run_all()

        # Run database benchmarks
        print("\n--- Database Benchmarks ---")
        db_bench = DatabaseBenchmarks(runner)
        db_bench.run_all()

    except ImportError as e:
        print(f"Benchmark modules not found: {e}")
        print("Run: pytest benchmarks/framework/ --benchmark-only")


def run_integration_benchmarks(runner: BenchmarkRunner) -> None:
    """Run integration/end-to-end benchmarks."""
    print("=== Integration Benchmarks ===")

    try:
        from benchmarks.integration.test_optimization_benchmarks import (
            OptimizationBenchmarks,
        )
        from benchmarks.integration.test_prediction_pipeline_benchmarks import (
            PredictionPipelineBenchmarks,
        )

        # Run prediction pipeline benchmarks
        print("\n--- Prediction Pipeline Benchmarks ---")
        pp_bench = PredictionPipelineBenchmarks(runner)
        pp_bench.run_all()

        # Run optimization benchmarks
        print("\n--- Optimization Benchmarks ---")
        opt_bench = OptimizationBenchmarks(runner)
        opt_bench.run_all()

    except ImportError as e:
        print(f"Integration benchmark modules not found: {e}")
        print("Run: pytest benchmarks/integration/ --benchmark-only")


def run_load_tests(runner: BenchmarkRunner) -> None:
    """Run load tests using locust."""
    print("=== Load Testing ===")

    try:
        from benchmarks.load_testing.api_load_test import run_api_load_test
        from benchmarks.load_testing.prediction_load_test import (
            run_prediction_load_test,
        )

        # Run prediction service load test
        print("\n--- Prediction Service Load Test ---")
        run_prediction_load_test(runner.config)

        # Run API load test if API is available
        print("\n--- API Load Test ---")
        run_api_load_test(runner.config)

    except ImportError as e:
        print(f"Load testing modules not found: {e}")
        print("Install locust: pip install locust")


def main():
    """Main benchmark runner entry point."""
    parser = argparse.ArgumentParser(description="Run AIrsenal performance benchmarks")

    parser.add_argument(
        "--suite",
        choices=["framework", "integration", "load", "all"],
        default="framework",
        help="Benchmark suite to run",
    )

    parser.add_argument(
        "--env",
        choices=["local", "ci", "stress"],
        default="local",
        help="Environment configuration to use",
    )

    parser.add_argument(
        "--compare-baseline", action="store_true", help="Compare results to baseline"
    )

    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as new baseline",
    )

    parser.add_argument("--output", type=str, help="Output file for results")

    parser.add_argument(
        "--threshold-check",
        action="store_true",
        help="Check results against performance thresholds",
    )

    parser.add_argument(
        "--profile", action="store_true", help="Enable CPU profiling (slower)"
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.env)
    if args.profile:
        config.profile_cpu = True

    # Initialize benchmark runner
    runner = BenchmarkRunner(config)

    try:
        print(f"Setting up benchmark environment (env: {args.env})...")
        runner.setup()

        start_time = time.time()

        # Run requested benchmark suites
        if args.suite in ("framework", "all"):
            run_framework_benchmarks(runner)

        if args.suite in ("integration", "all"):
            run_integration_benchmarks(runner)

        if args.suite in ("load", "all"):
            run_load_tests(runner)

        # Calculate total time
        total_time = time.time() - start_time
        print(f"\nAll benchmarks completed in {total_time:.2f} seconds")

        # Save results
        results_file = runner.save_results(args.output)

        # Generate summary
        summary = runner.get_summary()
        print("\nBenchmark Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Average duration: {summary['avg_duration']:.4f}s")
        print(f"  Max memory usage: {summary['max_memory_mb']:.2f}MB")
        print(f"  Failed tests: {summary['failed_tests']}")

        # Compare to baseline if requested
        if args.compare_baseline:
            print("\nComparing to baseline...")
            comparison = compare_to_baseline(runner.results)

            if comparison["status"] == "compared":
                print(f"  Regressions: {comparison['summary']['total_regressions']}")
                print(f"  Improvements: {comparison['summary']['total_improvements']}")

                if comparison["regressions"]:
                    print("\nPerformance Regressions:")
                    for reg in comparison["regressions"]:
                        print(
                            f"  {reg['test_name']}: {reg['change_percent']:.1f}% slower"
                        )

        # Check thresholds if requested
        if args.threshold_check:
            print("\nChecking performance thresholds...")
            threshold_check = check_thresholds(runner.results, config)

            if threshold_check["pass"]:
                print("  All tests passed threshold checks ✓")
            else:
                print(f"  {threshold_check['total_violations']} threshold violations:")
                for violation in threshold_check["violations"]:
                    print(
                        f"    {violation['test_name']}: {violation['violation_percent']:.1f}% over threshold"
                    )

                # Exit with error code if thresholds violated
                sys.exit(1)

        # Save as baseline if requested
        if args.save_baseline:
            baseline_path = Path("benchmarks/configs/baselines/baseline.json")
            baseline_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            shutil.copy(results_file, baseline_path)
            print(f"Saved new baseline: {baseline_path}")

        print(f"\nResults saved to: {results_file}")

    except KeyboardInterrupt:
        print("\nBenchmarks interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        runner.teardown()


if __name__ == "__main__":
    main()
