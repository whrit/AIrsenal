"""
Comprehensive Test Fixtures for AIrsenal Enhanced Components

This package provides factory_boy factories and fixtures for testing all the new
Sprint 00 components including extended player attributes, model versioning,
feature store, database versioning, and infrastructure components.

Modules:
- fpl_data: Realistic FPL player names, teams, and statistical data
- player_factories: Factory_boy factories for Player and PlayerAttributes
- model_factories: Factory_boy factories for model versioning components
- feature_factories: Factory_boy factories for feature store components
- database_factories: Factory_boy factories for database versioning components
- mock_services: Mock implementations for external services
- pytest_fixtures: Pytest fixtures for common testing scenarios
"""

__version__ = "1.0.0"
__author__ = "AIrsenal Test Infrastructure"

# Import all factories for easy access
from .database_factories import *
from .feature_factories import *

# Import realistic data sets
from .fpl_data import *
from .mock_services import *
from .model_factories import *
from .player_factories import *

__all__ = [
    "PREMIER_LEAGUE_TEAMS",
    # Realistic data
    "REALISTIC_PLAYERS",
    "STATISTICAL_DISTRIBUTIONS",
    "ComputedFeatureFactory",
    # Database versioning factories
    "DatabaseVersionFactory",
    "FeatureCacheFactory",
    # Feature store factories
    "FeatureDefinitionFactory",
    "FeatureTimeSeriesFactory",
    "MigrationHistoryFactory",
    # Mock services
    "MockFPLDataFetcher",
    "MockPlayerModel",
    "MockRedisCache",
    "ModelArtifactFactory",
    "ModelExperimentFactory",
    "ModelPerformanceFactory",
    # Model versioning factories
    "ModelRegistryFactory",
    "ModelVersionFactory",
    "PlayerAttributesExtendedFactory",
    "PlayerAttributesFactory",
    # Player factories
    "PlayerFactory",
    "SchemaCompatibilityFactory",
]
