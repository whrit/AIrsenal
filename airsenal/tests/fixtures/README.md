# AIrsenal Test Fixtures Documentation

This directory provides comprehensive test fixtures for all AIrsenal enhanced components added in Sprint 00. The fixtures are designed to provide realistic, consistent, and efficient test data for unit tests, integration tests, and performance testing.

## Overview

The test fixtures system includes:

- **Factory_boy factories** for generating realistic test data
- **Mock services** for external dependencies (FPL API, Redis, etc.)
- **Pytest fixtures** for common testing scenarios
- **Database seeding utilities** for large-scale testing
- **Realistic FPL data** based on actual Premier League players and teams

## Quick Start

### Basic Usage

```python
import pytest
from airsenal.tests.fixtures import PlayerAttributesExtendedFactory, MockFPLDataFetcher

def test_player_creation():
    # Create a player with realistic extended attributes
    player_attrs = PlayerAttributesExtendedFactory()
    
    assert player_attrs.xg_per_90 is not None
    assert player_attrs.form_3_games is not None
    assert player_attrs.position in ["GK", "DEF", "MID", "FWD"]

def test_with_mock_fpl_api(mock_fpl_fetcher):
    # Use mock FPL API in tests
    bootstrap_data = mock_fpl_fetcher.get_bootstrap_data()
    assert "elements" in bootstrap_data
    assert len(bootstrap_data["elements"]) > 0
```

### Using Database Fixtures

```python
def test_with_comprehensive_data(enhanced_session, fpl_squad):
    # enhanced_session provides comprehensive test data
    # fpl_squad provides a complete 15-player squad
    
    assert len(fpl_squad) == 15
    
    # Verify squad composition
    positions = [player.position("2324") for player in fpl_squad]
    assert positions.count("GK") == 2
    assert positions.count("DEF") == 5
    assert positions.count("MID") == 5
    assert positions.count("FWD") == 3
```

## Fixtures Directory Structure

```
fixtures/
├── __init__.py              # Main imports and exports
├── fpl_data.py             # Realistic FPL player names and statistical data
├── player_factories.py     # Player and PlayerAttributes factories
├── model_factories.py      # Model versioning component factories
├── feature_factories.py    # Feature store component factories
├── database_factories.py   # Database versioning component factories
├── mock_services.py        # Mock implementations for external services
├── pytest_fixtures.py     # Comprehensive pytest fixtures
├── test_database.py        # Database seeding and management utilities
└── README.md               # This documentation file
```

## Factory Classes

### Player Factories

#### Basic Factories

```python
from airsenal.tests.fixtures import (
    PlayerFactory,
    PlayerAttributesFactory,
    PlayerAttributesExtendedFactory
)

# Create basic player
player = PlayerFactory()

# Create player with basic attributes
basic_attrs = PlayerAttributesFactory()

# Create player with all extended attributes (19 new fields)
extended_attrs = PlayerAttributesExtendedFactory()
print(f"xG per 90: {extended_attrs.xg_per_90}")
print(f"Form (3 games): {extended_attrs.form_3_games}")
print(f"Is penalty taker: {extended_attrs.is_penalty_taker}")
```

#### Position-Specific Factories

```python
from airsenal.tests.fixtures import (
    GoalkeeperAttributesFactory,
    DefenderAttributesFactory,
    MidfielderAttributesFactory,
    ForwardAttributesFactory
)

# Create position-specific players with realistic stats
goalkeeper = GoalkeeperAttributesFactory()
assert goalkeeper.position == "GK"
assert goalkeeper.xg_per_90 < 0.1  # Goalkeepers have very low xG

midfielder = MidfielderAttributesFactory()
assert midfielder.position == "MID"
assert midfielder.key_passes_per_90 > 1.0  # Midfielders create chances
```

#### Scenario-Specific Factories

```python
from airsenal.tests.fixtures import (
    HighFormPlayerAttributesFactory,
    LowFormPlayerAttributesFactory,
    PenaltyTakerPlayerAttributesFactory,
    EasyFixturesPlayerAttributesFactory
)

# Create players for specific testing scenarios
high_form_player = HighFormPlayerAttributesFactory()
assert high_form_player.form_3_games >= 8.0

penalty_taker = PenaltyTakerPlayerAttributesFactory()
assert penalty_taker.is_penalty_taker is True
assert penalty_taker.role_confidence >= 0.7
```

### Model Versioning Factories

```python
from airsenal.tests.fixtures import (
    ModelRegistryFactory,
    ModelVersionFactory,
    ProductionModelVersionFactory,
    ModelArtifactFactory,
    ModelPerformanceFactory
)

# Create model registry
registry = ModelRegistryFactory(model_name="player_model_mid")

# Create model version
version = ModelVersionFactory(registry=registry, version="1.2.0")

# Create production model
prod_model = ProductionModelVersionFactory(
    registry=registry, 
    version="1.1.0",
    is_production=True
)

# Create model artifact
artifact = ModelArtifactFactory(
    version=version,
    artifact_type="full_model"
)

# Create performance record
performance = ModelPerformanceFactory(
    version=version,
    dataset_type="validation",
    mae=0.15
)
```

### Feature Store Factories

```python
from airsenal.tests.fixtures import (
    FeatureDefinitionFactory,
    ComputedFeatureFactory,
    FeatureCacheFactory,
    FeatureTimeSeriesFactory
)

# Create feature definition
feature_def = FeatureDefinitionFactory(
    name="rolling_goals_5",
    feature_type="player",
    data_type="float"
)

# Create computed feature
computed = ComputedFeatureFactory(
    feature_definition=feature_def,
    entity_id=123,
    value=2.5
)

# Create cache entry
cache_entry = FeatureCacheFactory(
    feature_name="rolling_goals_5",
    entity_id=123,
    value=2.5,
    hit_count=50
)

# Create time series data
timeseries = FeatureTimeSeriesFactory(
    entity_type="player",
    entity_id=123,
    metric_name="goals",
    gameweek=10,
    value=2.0
)
```

## Mock Services

### Mock FPL Data Fetcher

```python
from airsenal.tests.fixtures import MockFPLDataFetcher

# Create mock FPL fetcher
fetcher = MockFPLDataFetcher(season="2324")
fetcher.current_gameweek = 15

# Get bootstrap data (players, teams, gameweeks)
bootstrap = fetcher.get_bootstrap_data()
players = bootstrap["elements"]
teams = bootstrap["teams"]

# Get fixtures
fixtures = fetcher.get_fixtures(gameweek=15)

# Get player details
player_data = fetcher.get_player_data(player_id=123)
print(f"Fixtures: {len(player_data['fixtures'])}")
print(f"History: {len(player_data['history'])}")

# Mock authentication and team data
fetcher.login("test_user", "test_password")
team_data = fetcher.get_fpl_team_data(team_id=742663)
```

### Mock Redis Cache

```python
from airsenal.tests.fixtures import MockRedisCache

# Create mock Redis cache
cache = MockRedisCache()

# Basic operations
cache.set("player:123:form", "7.5", ex=3600)
value = cache.get("player:123:form")
assert value.decode() == "7.5"

# Check existence and TTL
assert cache.exists("player:123:form")
assert cache.ttl("player:123:form") > 0

# Bulk operations
cache.set("player:456:xg", "0.8")
cache.set("team:1:strength", "1.2")
keys = cache.keys("player:*")
assert len(keys) == 2

# Cache statistics
info = cache.info()
print(f"Cache hits: {info['stats']['hits']}")
print(f"Cache misses: {info['stats']['misses']}")
```

### Mock Player Models

```python
from airsenal.tests.fixtures import MockPlayerModel
import numpy as np

# Create mock player model
model = MockPlayerModel(position="MID")

# Train the model
X = np.random.random((100, 10))  # 100 samples, 10 features
y = np.random.uniform(0, 20, 100)  # Points 0-20
model.fit(X, y)

# Make predictions
X_test = np.random.random((10, 10))
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get feature importance
importance = model.get_feature_importance()
print(f"Most important feature: {max(importance.keys(), key=importance.get)}")

# Save and load model
model.save_model("/tmp/test_model.json")
loaded_model = MockPlayerModel.load_model("/tmp/test_model.json")
```

## Pytest Fixtures

### Database Fixtures

```python
def test_with_database_fixtures(test_session, enhanced_session):
    # test_session: Basic SQLAlchemy session
    # enhanced_session: Session with factory_boy configured
    
    from airsenal.tests.fixtures import PlayerAttributesExtendedFactory
    
    # Create player using enhanced session
    player_attrs = PlayerAttributesExtendedFactory()
    enhanced_session.commit()
    
    # Query the created player
    retrieved = enhanced_session.query(PlayerAttributes).first()
    assert retrieved.player_id == player_attrs.player_id
```

### Comprehensive Scenarios

```python
def test_transfer_optimization(transfer_optimization_scenario):
    """Test using comprehensive transfer optimization scenario."""
    scenario = transfer_optimization_scenario
    
    current_squad = scenario["current_squad"]
    transfer_targets = scenario["transfer_targets"]
    fetcher = scenario["fetcher"]
    
    # Run transfer optimization logic
    assert len(current_squad) == 15
    assert len(transfer_targets) > 0
    
    # Use FPL fetcher for additional data
    bootstrap = fetcher.get_bootstrap_data()
    assert "elements" in bootstrap

def test_model_deployment(model_testing_scenario):
    """Test using model deployment scenario."""
    scenario = model_testing_scenario
    
    pipeline = scenario["pipeline"]
    ab_test = scenario["ab_test"]
    storage_dir = scenario["storage_dir"]
    
    # Test model deployment pipeline
    production_models = pipeline["production"]
    assert len(production_models) > 0
    assert production_models[0].is_production
    
    # Test A/B testing
    experiment = ab_test["experiment"]
    assert experiment.status == "running"

def test_feature_engineering(feature_engineering_scenario):
    """Test using feature engineering scenario."""
    scenario = feature_engineering_scenario
    
    pipeline = scenario["pipeline"]
    cache = scenario["cache"]
    
    # Test feature computation
    definitions = pipeline["definitions"]
    computed_features = pipeline["computed_features"]
    
    assert len(definitions) > 0
    assert len(computed_features) > 0
    
    # Test caching
    assert cache.exists("feature:rolling_goals_5:1:gw*")
```

### Parameterized Testing

```python
@pytest.mark.parametrize("position", ["GK", "DEF", "MID", "FWD"])
def test_position_specific_logic(position, enhanced_session):
    """Test logic for each position using parameterized fixture."""
    from airsenal.tests.fixtures import PlayerAttributesExtendedFactory
    
    player_attrs = PlayerAttributesExtendedFactory(position=position)
    
    if position == "GK":
        assert player_attrs.xg_per_90 < 0.1
        assert player_attrs.is_penalty_taker is False
    elif position == "FWD":
        assert player_attrs.xg_per_90 > 0.2
        assert player_attrs.shots_per_90 > 2.0

def test_multiple_gameweeks(multi_gameweek_scenario):
    """Test using parameterized gameweek scenarios."""
    scenario = multi_gameweek_scenario
    
    gameweek = scenario["gameweek"]
    players = scenario["players"]
    
    # Test gameweek-specific logic
    assert all(p.gameweek == gameweek for p in players)
    
    if gameweek <= 5:
        # Early season logic
        pass
    elif gameweek >= 30:
        # End of season logic
        pass
```

## Database Management

### Test Database Manager

```python
from airsenal.tests.fixtures import TestDatabaseManager, create_test_database

# Create test database with preset
db_manager = create_test_database(preset="standard")

# Seed specific data types
basic_data = db_manager.seed_basic_data(num_players=100)
season_data = db_manager.seed_season_data(season="2324", num_gameweeks=20)
model_data = db_manager.seed_model_versioning_data()

# Get seeded data references
players = db_manager.get_seeded_data("players")
print(f"Created {len(players)} players")

# Verify data integrity
integrity_report = db_manager.verify_data_integrity()
print(f"Table counts: {integrity_report['table_counts']}")

# Cleanup
db_manager.destroy_database()
```

### Context Manager Usage

```python
from airsenal.tests.fixtures import TestDatabaseContext

# Use context manager for automatic cleanup
with TestDatabaseContext(preset="comprehensive") as db_manager:
    # Database is automatically created and seeded
    session = db_manager.get_session()
    
    # Run tests
    players = session.query(Player).all()
    assert len(players) > 0
    
    # Database is automatically cleaned up when exiting context
```

### Quick Database Creation

```python
from airsenal.tests.fixtures import quick_test_database, benchmark_database

# Quick minimal database for simple tests
quick_db = quick_test_database(num_players=20)

# Benchmark database for performance tests
benchmark_db = benchmark_database(size="medium")
```

## Realistic Test Data

### FPL Player Names and Teams

The `fpl_data.py` module provides realistic data based on actual Premier League:

```python
from airsenal.tests.fixtures.fpl_data import (
    REALISTIC_PLAYERS,
    PREMIER_LEAGUE_TEAMS,
    get_realistic_player_name,
    get_random_team,
    generate_realistic_player_data
)

# Get realistic player names by position
gk_name = get_realistic_player_name("GK")  # e.g., "Aaron Ramsdale"
mid_name = get_realistic_player_name("MID")  # e.g., "Martin Odegaard"

# Get team information
team_code, team_name = get_random_team()  # e.g., ("ARS", "Arsenal")

# Generate complete realistic player data
player_data = generate_realistic_player_data(position="MID", team="ARS")
print(f"Name: {player_data['name']}")
print(f"xG per 90: {player_data['xg_per_90']}")
print(f"Form (3 games): {player_data['form_3_games']}")
```

### Statistical Distributions

Realistic statistical distributions ensure test data mirrors actual FPL performance:

```python
from airsenal.tests.fixtures.fpl_data import STATISTICAL_DISTRIBUTIONS

# Get position-specific distributions
mid_stats = STATISTICAL_DISTRIBUTIONS["MID"]
print(f"xG mean for midfielders: {mid_stats['xg_per_90'].mean}")
print(f"Form mean for midfielders: {mid_stats['form_3_games'].mean}")

# Generate realistic values
xg_value = mid_stats["xg_per_90"].generate()
form_value = mid_stats["form_3_games"].generate()
```

## Performance Testing

### Stress Testing

```python
def test_performance_with_large_dataset(stress_test_scenario):
    """Test performance with large datasets."""
    scenario = stress_test_scenario
    
    players = scenario["players"]
    cache_entries = scenario["cache_entries"]
    
    # Performance test with 100 players and 500 cache entries
    assert scenario["player_count"] == 100
    assert scenario["cache_count"] == 500
    
    # Run performance-critical operations
    start_time = time.time()
    
    # ... performance test logic ...
    
    end_time = time.time()
    assert (end_time - start_time) < 5.0  # Should complete in 5 seconds

def test_benchmark_scenario(performance_test_db):
    """Test using performance-optimized database."""
    session = performance_test_db.get_session()
    
    # Benchmark database operations
    players = session.query(Player).all()
    assert len(players) >= 500  # Medium-sized dataset
```

### Memory and Resource Testing

```python
def test_memory_usage(benchmark_data):
    """Test memory usage with standardized data."""
    data = benchmark_data
    
    players = data["players"]
    
    # Test memory-efficient operations
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform operations
    for player in players:
        # ... operations ...
        pass
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert reasonable memory usage
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Best Practices

### 1. Use Appropriate Factory Scope

```python
# Function scope for isolated tests
def test_isolated_player():
    player = PlayerAttributesExtendedFactory()
    # Each test gets fresh data

# Use build() for objects without database persistence
def test_without_database():
    player_data = PlayerAttributesExtendedFactory.build()
    # Creates object without saving to database
```

### 2. Combine Factories for Complex Scenarios

```python
def test_transfer_scenario():
    # Create current squad
    current_squad = [PlayerAttributesExtendedFactory() for _ in range(15)]
    
    # Create high-value transfer targets
    transfer_targets = [
        HighFormPlayerAttributesFactory() for _ in range(5)
    ]
    
    # Create players to transfer out
    transfer_candidates = [
        LowFormPlayerAttributesFactory() for _ in range(3)
    ]
```

### 3. Use Mock Services for External Dependencies

```python
def test_with_external_apis(mock_fpl_fetcher, mock_redis_cache):
    # All external API calls are mocked
    # Tests run fast and reliably
    
    # Mock FPL API
    bootstrap = mock_fpl_fetcher.get_bootstrap_data()
    
    # Mock Redis cache
    mock_redis_cache.set("test_key", "test_value")
    assert mock_redis_cache.get("test_key") == b"test_value"
```

### 4. Validate Test Data

```python
def test_with_validation(data_validators):
    player_attrs = PlayerAttributesExtendedFactory()
    
    # Use built-in validators
    data_validators["player_attributes"](player_attrs)
    
    squad = [PlayerAttributesExtendedFactory() for _ in range(15)]
    players = [attrs.player for attrs in squad]
    data_validators["fpl_squad"](players)
```

### 5. Use Database Presets Appropriately

```python
# Minimal preset for unit tests
def test_unit_functionality(isolated_test_db):
    # Fast, minimal data
    pass

# Standard preset for integration tests  
def test_integration_scenario(module_test_db):
    # Comprehensive data, shared across module
    pass

# Comprehensive preset for full system tests
def test_full_system(test_database_manager):
    # Complete data for end-to-end testing
    pass
```

## Troubleshooting

### Common Issues

1. **Factory_boy Session Issues**
   ```python
   # Ensure factory_boy uses correct session
   PlayerAttributesExtendedFactory._meta.sqlalchemy_session = your_session
   ```

2. **Database Cleanup**
   ```python
   # Manual cleanup if automatic cleanup fails
   from airsenal.tests.fixtures import cleanup_test_databases
   cleanup_test_databases()
   ```

3. **Mock Service State**
   ```python
   # Reset mock services between tests
   mock_redis_cache.flushall()
   mock_fpl_fetcher = MockFPLDataFetcher()  # Create fresh instance
   ```

### Debugging Tips

1. **Inspect Generated Data**
   ```python
   player_attrs = PlayerAttributesExtendedFactory()
   print(f"Generated player: {player_attrs}")
   print(f"xG per 90: {player_attrs.xg_per_90}")
   print(f"Form: {player_attrs.form_3_games}")
   ```

2. **Check Database State**
   ```python
   db_manager = create_test_database(preset="standard")
   integrity_report = db_manager.verify_data_integrity()
   print(f"Database integrity: {integrity_report}")
   ```

3. **Monitor Resource Usage**
   ```python
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

## Contributing

When adding new fixtures:

1. **Follow naming conventions**: `ComponentNameFactory` for factories
2. **Add realistic data**: Use actual FPL player names and realistic statistics
3. **Include documentation**: Document new fixtures in this README
4. **Add validation**: Include data validation functions
5. **Test performance**: Ensure fixtures perform well with large datasets
6. **Maintain backward compatibility**: Don't break existing tests

### Example New Factory

```python
class NewComponentFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for new component with realistic data."""
    
    class Meta:
        model = NewComponent
        sqlalchemy_session_persistence = "commit"
    
    # Define realistic fields using appropriate patterns
    name = factory.LazyFunction(lambda: get_realistic_name())
    value = factory.LazyAttribute(lambda obj: calculate_realistic_value(obj.name))
    
    # Use existing data patterns for consistency
    season = factory.LazyFunction(lambda: generate_season_gameweek()[0])
    
    @factory.post_generation
    def add_relationships(obj, create, extracted, **kwargs):
        """Add related objects if needed."""
        if create:
            # Create related objects
            pass
```

This comprehensive fixture system provides everything needed to test AIrsenal's enhanced components effectively, from simple unit tests to complex integration scenarios and performance testing.