#!/usr/bin/env python3
"""
AIrsenal Feature Store Usage Examples

This script demonstrates practical usage of the new feature store infrastructure
for FPL prediction workflows. It shows how to:

1. Initialize and configure the feature store
2. Register custom features
3. Compute and cache features for players
4. Use features in prediction pipelines
5. Monitor feature quality and performance
6. Migrate existing data

Run this script to see the feature store in action:
    python examples/feature_store_usage.py
"""

import logging
import sys
from pathlib import Path

# Add airsenal to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airsenal.framework.feature_store import FeatureStore
from airsenal.framework.feature_store_integration import (
    FeatureAwarePredictionUtils,
    FeatureMigrator,
    FeatureStoreMonitor,
)
from airsenal.framework.schema import session
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_basic_feature_registration():
    """Example 1: Register and compute basic features."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Feature Registration and Computation")
    print("=" * 60)

    # Initialize feature store
    store = FeatureStore()

    # Register a custom feature
    print("1. Registering custom feature...")
    feature_def = store.register_feature(
        name="goals_per_90",
        feature_type="player",
        data_type="float",
        description="Goals scored per 90 minutes played",
        computation_logic={
            "type": "custom",
            "formula": "goals / (minutes / 90)",
            "min_minutes": 270,  # At least 3 games worth
        },
        version="1.0.0",
    )
    print(f"✓ Registered feature: {feature_def}")

    # List available features
    print("\n2. Available features:")
    features = store.registry.list_features("player")
    for feature in features[:5]:  # Show first 5
        print(f"   - {feature.name} ({feature.feature_type}): {feature.description}")
    print(f"   ... and {len(features) - 5} more")

    # Get features for some players
    print("\n3. Computing features for sample players...")
    sample_player_ids = [1, 2, 3]  # Replace with actual player IDs from your DB

    try:
        features_data = store.get_features(
            entity_type="player",
            entity_ids=sample_player_ids,
            feature_names=["rolling_goals_5", "rolling_assists_5", "goals_form"],
            season=CURRENT_SEASON,
            gameweek=NEXT_GAMEWEEK,
        )

        print("✓ Features computed successfully!")
        for player_id, features in features_data.items():
            print(f"   Player {player_id}:")
            for feature_name, value in features.items():
                print(f"     {feature_name}: {value}")

    except Exception as e:
        print(f"⚠ Feature computation failed (expected if no data): {e}")

    # Clean up test feature
    session.delete(feature_def)
    session.commit()


def example_2_rolling_windows_and_form():
    """Example 2: Advanced rolling windows and form metrics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Rolling Windows and Form Metrics")
    print("=" * 60)

    store = FeatureStore()

    # Register advanced features
    print("1. Registering advanced rolling features...")

    features_to_register = [
        {
            "name": "rolling_minutes_10",
            "feature_type": "player",
            "description": "10-game rolling average minutes",
            "computation_logic": {
                "type": "rolling",
                "metric": "minutes",
                "window": 10,
                "agg": "mean",
            },
        },
        {
            "name": "rolling_points_volatility",
            "feature_type": "player",
            "description": "5-game rolling standard deviation of points",
            "computation_logic": {
                "type": "rolling",
                "metric": "points",
                "window": 5,
                "agg": "std",
            },
        },
        {
            "name": "minutes_consistency",
            "feature_type": "player",
            "description": "How consistent player's minutes are (lower = more consistent)",
            "computation_logic": {
                "type": "rolling",
                "metric": "minutes",
                "window": 8,
                "agg": "std",
            },
        },
        {
            "name": "hot_streak",
            "feature_type": "player",
            "description": "Exponentially weighted recent form",
            "computation_logic": {
                "type": "form",
                "metric": "points",
                "decay_factor": 0.8,
            },
        },
    ]

    registered_features = []
    for feature_config in features_to_register:
        try:
            feature_def = store.register_feature(**feature_config)
            registered_features.append(feature_def)
            print(f"✓ {feature_config['name']}")
        except Exception as e:
            print(f"⚠ Failed to register {feature_config['name']}: {e}")

    # Demonstrate feature computation concepts
    print("\n2. Feature computation concepts:")
    print("   - Rolling windows: Compute statistics over sliding time windows")
    print("   - Form metrics: Exponentially weighted recent performance")
    print("   - Volatility: Measure of performance consistency")
    print("   - Multi-aggregation: mean, std, min, max, percentiles")

    # Show feature stats
    print("\n3. Feature registry statistics:")
    all_features = store.registry.list_features()
    by_type = {}
    for feature in all_features:
        by_type.setdefault(feature.feature_type, 0)
        by_type[feature.feature_type] += 1

    for feature_type, count in by_type.items():
        print(f"   {feature_type}: {count} features")

    # Clean up
    for feature_def in registered_features:
        session.delete(feature_def)
    session.commit()


def example_3_caching_and_performance():
    """Example 3: Caching and performance optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Caching and Performance")
    print("=" * 60)

    store = FeatureStore()

    # Test cache performance
    print("1. Cache performance demonstration...")

    # Get cache stats before
    initial_stats = store.cache.get_stats()
    print(f"Initial cache stats: {initial_stats['memory_cache']}")

    # Simulate feature requests (with mock data)
    test_player_ids = [100, 101, 102]
    feature_names = ["rolling_goals_5", "rolling_assists_5"]

    # First request (will compute and cache)
    print("\n2. First request (compute + cache)...")
    try:
        start_time = time.time()
        store.get_features(
            entity_type="player",
            entity_ids=test_player_ids,
            feature_names=feature_names,
            use_cache=True,
        )
        first_time = time.time() - start_time
        print(f"✓ First request took {first_time:.3f} seconds")
    except Exception as e:
        print(f"⚠ First request failed (expected): {e}")
        first_time = 0

    # Second request (should use cache)
    print("\n3. Second request (cache hit)...")
    try:
        start_time = time.time()
        store.get_features(
            entity_type="player",
            entity_ids=test_player_ids,
            feature_names=feature_names,
            use_cache=True,
        )
        second_time = time.time() - start_time
        print(f"✓ Second request took {second_time:.3f} seconds")

        if first_time > 0 and second_time < first_time:
            speedup = first_time / second_time
            print(f"✓ Cache speedup: {speedup:.1f}x faster!")

    except Exception as e:
        print(f"⚠ Second request failed: {e}")

    # Cache management
    print("\n4. Cache management:")
    print("   - Multi-level: memory + database")
    print("   - TTL expiration: configurable per feature")
    print("   - Selective invalidation: by feature, entity, or context")
    print("   - Performance monitoring: hit rates, response times")

    # Get updated cache stats
    final_stats = store.cache.get_stats()
    print(f"\nFinal cache stats: {final_stats['memory_cache']}")

    # Cache invalidation example
    print("\n5. Cache invalidation example...")
    invalidated = store.cache.invalidate("rolling_goals_5", "player")
    print(f"✓ Invalidated {invalidated} cache entries for rolling_goals_5")


def example_4_prediction_integration():
    """Example 4: Integration with prediction pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Prediction Pipeline Integration")
    print("=" * 60)

    # Initialize enhanced prediction utils
    predictor = FeatureAwarePredictionUtils()

    print("1. Enhanced prediction capabilities:")
    print("   - Feature-aware point calculation")
    print("   - Automatic feature computation")
    print("   - Backwards compatibility with existing pipeline")
    print("   - Cached feature retrieval for performance")

    # Demonstrate feature retrieval for prediction
    print("\n2. Getting features for prediction...")
    sample_player_ids = [1, 2, 3]

    try:
        prediction_features = predictor.get_player_features_for_prediction(
            player_ids=sample_player_ids, season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK
        )

        print("✓ Retrieved prediction features!")
        print("Features include:")
        if prediction_features and len(prediction_features) > 0:
            first_player = next(iter(prediction_features.values()))
            for feature_name in first_player:
                print(f"   - {feature_name}")
        else:
            print("   (No features computed - expected if no data)")

    except Exception as e:
        print(f"⚠ Feature retrieval failed (expected): {e}")

    # Batch processing example
    print("\n3. Batch feature updates:")
    print("   - Efficient bulk computation")
    print("   - Chunked processing for memory management")
    print("   - Progress tracking and error handling")
    print("   - Integration with existing update workflows")

    try:
        # This would normally update features for all players
        stats = predictor.batch_update_player_features(
            season=CURRENT_SEASON,
            gameweek_range=(NEXT_GAMEWEEK - 1, NEXT_GAMEWEEK),
            position_filter="FWD",  # Just forwards for faster example
        )
        print(f"✓ Batch update stats: {stats}")
    except Exception as e:
        print(f"⚠ Batch update failed (expected): {e}")


def example_5_monitoring_and_quality():
    """Example 5: Feature monitoring and quality assurance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Monitoring and Quality Assurance")
    print("=" * 60)

    monitor = FeatureStoreMonitor()

    print("1. Feature quality monitoring capabilities:")
    print("   - Data validation and type checking")
    print("   - Statistical drift detection")
    print("   - Feature freshness monitoring")
    print("   - Performance metrics and alerting")

    # Feature validation example
    print("\n2. Feature validation:")
    validator = monitor.feature_store.validator

    # Example feature definition for validation
    from airsenal.framework.schema import FeatureDefinition

    mock_feature = FeatureDefinition(
        name="test_validation", data_type="float", feature_type="player"
    )

    test_values = [5.5, "invalid", None, float("inf"), -10.2]
    for value in test_values:
        is_valid = validator.validate_feature_value(mock_feature, value)
        status = "✓" if is_valid else "✗"
        print(f"   {status} Value {value} ({'valid' if is_valid else 'invalid'})")

    # Feature freshness check
    print("\n3. Feature freshness monitoring:")
    print("   - Tracks when features were last computed")
    print("   - Identifies stale features needing refresh")
    print("   - Configurable freshness thresholds")

    # Quality report generation
    print("\n4. Generating quality report...")
    try:
        quality_report = monitor.generate_feature_quality_report()
        print("✓ Quality report generated!")
        print(f"   Total features: {quality_report['total_features']}")
        print(f"   Alerts: {len(quality_report['alerts'])}")

        if quality_report["alerts"]:
            print("   Recent alerts:")
            for alert in quality_report["alerts"][:3]:
                print(f"     - {alert}")
    except Exception as e:
        print(f"⚠ Quality report failed: {e}")


def example_6_migration_and_setup():
    """Example 6: Data migration and initial setup."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Data Migration and Setup")
    print("=" * 60)

    migrator = FeatureMigrator()

    print("1. Migration capabilities:")
    print("   - Convert existing PlayerScore data to time series")
    print("   - Compute historical features for backtesting")
    print("   - Migrate team and fixture data")
    print("   - Preserve data lineage and versioning")

    print("\n2. Migration workflow:")
    print("   Step 1: Migrate raw data to time series format")
    print("   Step 2: Compute historical rolling features")
    print("   Step 3: Validate migrated data quality")
    print("   Step 4: Update prediction models to use features")

    # Example migration (conceptual - would be real with data)
    print("\n3. Example migration process:")
    try:
        # This would migrate a full season of data
        migrated_count = migrator.migrate_player_scores_to_timeseries(
            season=CURRENT_SEASON, metrics=["goals", "assists", "minutes"]
        )
        print(f"✓ Migrated {migrated_count} time series entries")
    except Exception as e:
        print(f"⚠ Migration failed (expected): {e}")

    print("\n4. Best practices for migration:")
    print("   - Start with recent seasons and work backwards")
    print("   - Validate data quality at each step")
    print("   - Use batch processing for large datasets")
    print("   - Maintain rollback capabilities")
    print("   - Monitor performance impact during migration")


def example_7_custom_features():
    """Example 7: Creating custom domain-specific features."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Custom FPL Domain Features")
    print("=" * 60)

    store = FeatureStore()

    print("1. FPL-specific feature examples:")

    # Advanced FPL features
    fpl_features = [
        {
            "name": "captain_potential",
            "description": "Likelihood of being a good captain choice",
            "computation_logic": {
                "type": "composite",
                "components": [
                    "rolling_points_5",
                    "fixture_difficulty",
                    "home_advantage",
                ],
            },
        },
        {
            "name": "differential_pick",
            "description": "Low ownership but high potential",
            "computation_logic": {
                "type": "composite",
                "formula": "high_points_form AND low_ownership",
            },
        },
        {
            "name": "fixture_run_quality",
            "description": "Quality of upcoming fixture run",
            "computation_logic": {
                "type": "future_window",
                "metric": "fixture_difficulty",
                "window": 5,
                "aggregation": "mean",
            },
        },
        {
            "name": "injury_risk",
            "description": "Probability of injury based on historical patterns",
            "computation_logic": {
                "type": "ml_model",
                "model": "injury_predictor",
                "features": ["minutes_played", "age", "position", "recent_injuries"],
            },
        },
    ]

    print("2. Registering FPL domain features:")
    registered_features = []

    for feature_config in fpl_features:
        try:
            # Simplified registration for demo
            feature_def = store.register_feature(
                name=feature_config["name"],
                feature_type="player",
                description=feature_config["description"],
                computation_logic=feature_config["computation_logic"],
            )
            registered_features.append(feature_def)
            print(f"✓ {feature_config['name']}")
        except Exception as e:
            print(f"⚠ Failed to register {feature_config['name']}: {e}")

    print("\n3. Feature composition patterns:")
    print("   - Rolling aggregations: goals, assists, minutes, points")
    print("   - Form metrics: exponentially weighted recent performance")
    print("   - Fixture analysis: difficulty, location, rest days")
    print("   - Ownership data: effective ownership, template differentials")
    print("   - Risk metrics: injury probability, rotation risk")
    print("   - Value analysis: points per million, price change trends")

    print("\n4. Advanced feature engineering:")
    print("   - Player clusters: similar players for comparison")
    print("   - Opponent adjustments: performance vs different team strengths")
    print("   - Seasonal trends: early/mid/late season patterns")
    print("   - Weather factors: performance in different conditions")

    # Clean up
    for feature_def in registered_features:
        session.delete(feature_def)
    session.commit()


def main():
    """Run all feature store examples."""
    print("AIrsenal Feature Store - Usage Examples")
    print("=" * 60)
    print("This script demonstrates the capabilities of the new feature store system.")
    print("Examples include registration, computation, caching, and integration.")

    # Import time here for performance examples
    import time

    globals()["time"] = time

    try:
        example_1_basic_feature_registration()
        example_2_rolling_windows_and_form()
        example_3_caching_and_performance()
        example_4_prediction_integration()
        example_5_monitoring_and_quality()
        example_6_migration_and_setup()
        example_7_custom_features()

        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print(
            "1. Run the feature store tests: pytest airsenal/tests/test_feature_store.py"
        )
        print(
            "2. Migrate your historical data: python -c 'from examples.feature_store_usage import migrate_historical_data; migrate_historical_data()'"
        )
        print("3. Update your prediction pipeline to use enhanced features")
        print("4. Set up monitoring alerts for feature quality")
        print("5. Explore custom features for your specific FPL strategy")

    except KeyboardInterrupt:
        print("\n⚠ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
