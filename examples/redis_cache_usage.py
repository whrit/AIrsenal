"""
Redis Cache Usage Examples for AIrsenal

This example demonstrates how to use the Redis caching layer for improved
performance in AIrsenal feature engineering and prediction pipelines.

Setup:
1. Install Redis server locally or configure Redis connection
2. Set environment variables for Redis configuration
3. Enable Redis caching in your AIrsenal configuration

Environment Variables:
- AIRSENAL_REDIS_ENABLE=true
- AIRSENAL_REDIS_HOST=localhost (default)
- AIRSENAL_REDIS_PORT=6379 (default)
- AIRSENAL_REDIS_PASSWORD=your_password (if needed)
- AIRSENAL_REDIS_COMPRESSION=zstd (optional)
- AIRSENAL_REDIS_DEFAULT_TTL=3600 (optional)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from airsenal.framework.redis_cache import redis_cache
from airsenal.framework.feature_store import FeatureStore, FeatureCacheManager
from airsenal.framework.schema import session


def example_basic_redis_operations():
    """Example of basic Redis cache operations."""
    print("=== Basic Redis Cache Operations ===")
    
    # Check if Redis is available
    if not redis_cache.is_available():
        print("Redis is not available - falling back to database cache only")
        return
    
    print(f"Redis available: {redis_cache.is_available()}")
    
    # Basic key-value operations
    redis_cache.set("example_key", "example_value", ttl=300)
    value = redis_cache.get("example_key")
    print(f"Retrieved value: {value}")
    
    # Store complex data types
    array_data = np.array([1.5, 2.3, 3.7, 4.1])
    redis_cache.set("numpy_array", array_data, ttl=600)
    retrieved_array = redis_cache.get("numpy_array")
    print(f"NumPy array equal: {np.array_equal(array_data, retrieved_array)}")
    
    # Store DataFrame
    df = pd.DataFrame({
        'player_id': [1, 2, 3],
        'goals': [2, 1, 0],
        'minutes': [90, 85, 60]
    })
    redis_cache.set("dataframe", df, ttl=600)
    retrieved_df = redis_cache.get("dataframe")
    print(f"DataFrame equal: {df.equals(retrieved_df)}")


def example_feature_caching():
    """Example of feature-specific caching operations."""
    print("\n=== Feature Caching Operations ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Cache player features
    player_id = 123
    feature_name = "rolling_goals_5"
    feature_value = 2.5
    
    # Set feature cache
    success = redis_cache.set_feature_cache(
        feature_name=feature_name,
        entity_type="player",
        entity_id=player_id,
        value=feature_value,
        context={"season": "2425", "gameweek": 10},
        ttl=1800  # 30 minutes
    )
    print(f"Feature cached successfully: {success}")
    
    # Retrieve feature from cache
    cached_value = redis_cache.get_feature_cache(
        feature_name=feature_name,
        entity_type="player",
        entity_id=player_id,
        context={"season": "2425", "gameweek": 10}
    )
    print(f"Retrieved feature value: {cached_value}")


def example_prediction_caching():
    """Example of prediction caching operations."""
    print("\n=== Prediction Caching Operations ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Example prediction data
    player_id = 456
    gameweek = 15
    season = "2425"
    predictions = np.array([2.3, 1.1, 0.8])  # goals, assists, clean_sheets
    
    # Cache predictions
    success = redis_cache.set_prediction_cache(
        player_id=player_id,
        gameweek=gameweek,
        season=season,
        predictions=predictions,
        model_version="v2.1",
        ttl=7*24*3600  # 7 days
    )
    print(f"Predictions cached successfully: {success}")
    
    # Retrieve predictions
    cached_predictions = redis_cache.get_prediction_cache(
        player_id=player_id,
        gameweek=gameweek,
        season=season,
        model_version="v2.1"
    )
    print(f"Predictions equal: {np.array_equal(predictions, cached_predictions)}")


def example_cache_decorators():
    """Example of using cache decorators for automatic caching."""
    print("\n=== Cache Decorators ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Example: Cache expensive feature computation
    @redis_cache.cache_feature(ttl=1800)
    def compute_rolling_goals(player_id, window=5):
        """Simulate expensive rolling goals computation."""
        print(f"Computing rolling goals for player {player_id} (window={window})")
        # In reality, this would query the database and compute statistics
        return np.random.rand() * 3  # Simulate computed value
    
    # First call will compute and cache
    result1 = compute_rolling_goals(123, window=5)
    print(f"First call result: {result1:.3f}")
    
    # Second call will use cache
    result2 = compute_rolling_goals(123, window=5)
    print(f"Second call result: {result2:.3f} (should be same)")
    
    # Different parameters will compute again
    result3 = compute_rolling_goals(123, window=10)
    print(f"Different params result: {result3:.3f} (will be different)")
    
    # Example: Cache database queries
    @redis_cache.cache_query(ttl=600, key_prefix="fixtures")
    def get_upcoming_fixtures(team_id, season):
        """Simulate expensive database query."""
        print(f"Querying upcoming fixtures for team {team_id}")
        # Simulate database result
        return [
            {"opponent": "Arsenal", "difficulty": 4},
            {"opponent": "Brighton", "difficulty": 2},
        ]
    
    fixtures1 = get_upcoming_fixtures(1, "2425")
    fixtures2 = get_upcoming_fixtures(1, "2425")  # Will use cache
    print(f"Fixtures cached: {fixtures1 == fixtures2}")


def example_bulk_operations():
    """Example of bulk cache operations for efficiency."""
    print("\n=== Bulk Cache Operations ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Bulk set multiple features
    feature_data = {
        "player:123:rolling_goals": 2.5,
        "player:123:rolling_assists": 1.2,
        "player:456:rolling_goals": 1.8,
        "player:456:rolling_assists": 2.1,
    }
    
    success = redis_cache.mset(feature_data, ttl=1800)
    print(f"Bulk set successful: {success}")
    
    # Bulk get multiple features
    keys = list(feature_data.keys())
    values = redis_cache.mget(keys)
    
    print("Bulk retrieved values:")
    for key, value in zip(keys, values):
        print(f"  {key}: {value}")


def example_cache_warming():
    """Example of cache warming for hot data."""
    print("\n=== Cache Warming ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Simulate prediction function
    def generate_predictions(player_id, gameweek, season):
        """Simulate ML model prediction generation."""
        print(f"Generating predictions for player {player_id}, GW {gameweek}")
        return np.random.rand(3) * 5  # Random predictions
    
    # Warm cache for upcoming gameweeks
    player_ids = [123, 456, 789]
    gameweeks = [16, 17, 18]
    season = "2425"
    
    warmed_count = redis_cache.warm_predictions_cache(
        player_ids=player_ids,
        gameweeks=gameweeks,
        season=season,
        prediction_func=generate_predictions
    )
    print(f"Warmed {warmed_count} prediction cache entries")
    
    # Simulate feature computation function
    def compute_feature(feature_name, entity_type, entity_id):
        """Simulate feature computation."""
        print(f"Computing {feature_name} for {entity_type} {entity_id}")
        return np.random.rand() * 10
    
    # Warm cache for hot features
    feature_names = ["rolling_goals_5", "form_rating", "fixture_difficulty"]
    entity_type = "player"
    entity_ids = [123, 456, 789]
    
    warmed_count = redis_cache.warm_features_cache(
        feature_names=feature_names,
        entity_type=entity_type,
        entity_ids=entity_ids,
        compute_func=compute_feature
    )
    print(f"Warmed {warmed_count} feature cache entries")


def example_cache_invalidation():
    """Example of cache invalidation strategies."""
    print("\n=== Cache Invalidation ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Set some test data
    redis_cache.set_prediction_cache(123, 10, "2425", np.array([1, 2, 3]))
    redis_cache.set_prediction_cache(123, 11, "2425", np.array([4, 5, 6]))
    redis_cache.set_feature_cache("rolling_goals", "player", 123, 2.5)
    
    # Invalidate all cache for a specific player
    invalidated = redis_cache.invalidate_player_cache(123)
    print(f"Invalidated {invalidated} entries for player 123")
    
    # Verify data is gone
    cached_pred = redis_cache.get_prediction_cache(123, 10, "2425")
    cached_feature = redis_cache.get_feature_cache("rolling_goals", "player", 123)
    print(f"Data invalidated: pred={cached_pred is None}, feature={cached_feature is None}")


def example_integrated_feature_store():
    """Example of using Redis with integrated FeatureCacheManager."""
    print("\n=== Integrated Feature Store with Redis ===")
    
    # Initialize feature cache manager (automatically uses Redis if available)
    cache_manager = FeatureCacheManager(dbsession=session, default_ttl=3600)
    
    print(f"Redis enabled in cache manager: {cache_manager.redis_enabled}")
    
    # Use the three-tier cache (memory -> Redis -> database)
    feature_name = "integrated_test_feature"
    entity_type = "player"
    entity_id = 789
    test_value = 42.5
    
    # Set feature (will update all cache levels)
    cache_manager.set(feature_name, entity_type, entity_id, test_value, ttl=1800)
    print(f"Set feature value: {test_value}")
    
    # Get feature (will check memory -> Redis -> database)
    retrieved_value = cache_manager.get(feature_name, entity_type, entity_id)
    print(f"Retrieved feature value: {retrieved_value}")
    
    # Get comprehensive statistics
    stats = cache_manager.get_stats()
    print("\nCache Statistics:")
    print(f"  Overall hit rate: {stats['overall']['hit_rate']:.3f}")
    print(f"  Memory cache entries: {stats['memory_cache']['entries']}")
    
    if stats['redis_cache']['available']:
        print(f"  Redis cache available: {stats['redis_cache']['available']}")
        print(f"  Redis compression: {stats['redis_cache']['compression']}")
    else:
        print(f"  Redis cache: {stats['redis_cache']['reason']}")


def example_monitoring_and_metrics():
    """Example of monitoring cache performance."""
    print("\n=== Cache Monitoring and Metrics ===")
    
    if not redis_cache.is_available():
        print("Redis not available")
        return
    
    # Perform some cache operations to generate metrics
    for i in range(10):
        key = f"metric_test_{i}"
        value = f"value_{i}"
        redis_cache.set(key, value)
        
        # Mix of hits and misses
        if i % 2 == 0:
            redis_cache.get(key)  # Hit
        else:
            redis_cache.get(f"nonexistent_{i}")  # Miss
    
    # Get detailed metrics
    metrics = redis_cache.get_metrics()
    print("\nRedis Cache Metrics:")
    print(f"  Available: {metrics['available']}")
    print(f"  Hit rate: {metrics['hit_rate']:.3f}")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Cache size: {metrics['total_size_mb']:.2f} MB")
    print(f"  Avg retrieval time: {metrics['avg_retrieval_time_ms']:.2f} ms")
    print(f"  Connection status: {metrics['connection_status']}")
    print(f"  Compression: {metrics['compression']}")
    
    # Get Redis server info
    info = redis_cache.get_info()
    if not info.get("error"):
        print("\nRedis Server Info:")
        print(f"  Version: {info.get('redis_version')}")
        print(f"  Memory usage: {info.get('used_memory_human')}")
        print(f"  Connected clients: {info.get('connected_clients')}")


def main():
    """Run all examples."""
    print("AIrsenal Redis Cache Usage Examples")
    print("=" * 50)
    
    # Check Redis availability first
    print(f"Redis library available: {redis_cache.connection_manager is not None}")
    print(f"Redis cache available: {redis_cache.is_available()}")
    
    if not redis_cache.is_available():
        print("\nNote: Redis is not available. Some examples will be skipped.")
        print("To enable Redis:")
        print("1. Install and start Redis server")
        print("2. Set AIRSENAL_REDIS_ENABLE=true")
        print("3. Configure connection settings if needed")
    
    # Run examples
    example_basic_redis_operations()
    example_feature_caching()
    example_prediction_caching()
    example_cache_decorators()
    example_bulk_operations()
    example_cache_warming()
    example_cache_invalidation()
    example_integrated_feature_store()
    example_monitoring_and_metrics()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()