"""
Test database seeding automation and cleanup procedures.

Provides utilities for setting up and tearing down test databases with
realistic data for comprehensive testing scenarios.
"""

import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

from airsenal.framework.schema import Base, get_connection_string
from .player_factories import (
    PlayerAttributesExtendedFactory, GoalkeeperAttributesFactory,
    DefenderAttributesFactory, MidfielderAttributesFactory, ForwardAttributesFactory,
    create_squad_players, create_gameweek_players, create_form_comparison_players,
    create_historical_progression
)
from .model_factories import (
    create_model_family, create_ab_test_scenario, create_model_development_pipeline
)
from .feature_factories import (
    create_feature_computation_pipeline, create_player_feature_timeline,
    create_feature_performance_test_data
)
from .database_factories import (
    create_migration_timeline, create_compatibility_matrix, create_failed_migration_scenario
)
from .fpl_data import PREMIER_LEAGUE_TEAMS


logger = logging.getLogger(__name__)


class TestDatabaseManager:
    """
    Manages test database creation, seeding, and cleanup.
    
    Provides methods to create isolated test databases with realistic data
    for different testing scenarios.
    """
    
    def __init__(self, database_url: str = None):
        """Initialize test database manager."""
        self.database_url = database_url or "sqlite:///:memory:"
        self.engine = None
        self.session_factory = None
        self.current_session = None
        self._seeded_data = {}
    
    def create_database(self) -> None:
        """Create test database with schema."""
        self.engine = create_engine(self.database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        logger.info(f"Created test database: {self.database_url}")
    
    def get_session(self):
        """Get database session."""
        if not self.session_factory:
            self.create_database()
        
        if not self.current_session:
            self.current_session = self.session_factory()
        
        return self.current_session
    
    def close_session(self):
        """Close current database session."""
        if self.current_session:
            self.current_session.close()
            self.current_session = None
    
    def destroy_database(self):
        """Destroy test database and cleanup resources."""
        self.close_session()
        
        if self.engine:
            Base.metadata.drop_all(self.engine)
            self.engine.dispose()
            self.engine = None
            self.session_factory = None
        
        self._seeded_data.clear()
        logger.info("Destroyed test database")
    
    def seed_basic_data(self, num_players: int = 50) -> Dict[str, Any]:
        """
        Seed database with basic player and team data.
        
        Args:
            num_players: Number of players to create
            
        Returns:
            Dictionary containing created data references
        """
        session = self.get_session()
        
        # Create players with extended attributes
        players = []
        for i in range(num_players):
            position = random.choice(["GK", "DEF", "MID", "FWD"])
            
            factory_class = {
                "GK": GoalkeeperAttributesFactory,
                "DEF": DefenderAttributesFactory,
                "MID": MidfielderAttributesFactory,
                "FWD": ForwardAttributesFactory,
            }[position]
            
            attrs = factory_class()
            players.append(attrs.player)
        
        session.commit()
        
        seeded_data = {
            "players": players,
            "num_players": len(players),
            "positions": ["GK", "DEF", "MID", "FWD"],
            "teams": list(PREMIER_LEAGUE_TEAMS.keys()),
        }
        
        self._seeded_data.update(seeded_data)
        logger.info(f"Seeded basic data: {num_players} players")
        return seeded_data
    
    def seed_season_data(self, season: str = "2324", num_gameweeks: int = 10) -> Dict[str, Any]:
        """
        Seed database with complete season data.
        
        Args:
            season: Season identifier
            num_gameweeks: Number of gameweeks to create data for
            
        Returns:
            Dictionary containing season data references
        """
        session = self.get_session()
        
        # Create players for each gameweek
        season_players = {}
        all_players = []
        
        for gameweek in range(1, num_gameweeks + 1):
            gw_players = create_gameweek_players(
                session, season=season, gameweek=gameweek, num_players=100
            )
            season_players[gameweek] = gw_players
            all_players.extend(gw_players)
        
        # Create form comparison players
        form_players = create_form_comparison_players(session)
        
        # Create feature timelines for key players
        feature_timelines = {}
        for i, player_attrs in enumerate(all_players[:20]):  # Top 20 players
            timeline = create_player_feature_timeline(
                session, 
                player_id=player_attrs.player_id, 
                season=season
            )
            feature_timelines[player_attrs.player_id] = timeline
        
        session.commit()
        
        seeded_data = {
            "season": season,
            "num_gameweeks": num_gameweeks,
            "season_players": season_players,
            "all_players": all_players,
            "form_players": form_players,
            "feature_timelines": feature_timelines,
        }
        
        self._seeded_data.update(seeded_data)
        logger.info(f"Seeded season data: {season}, {num_gameweeks} gameweeks, {len(all_players)} player records")
        return seeded_data
    
    def seed_model_versioning_data(self) -> Dict[str, Any]:
        """
        Seed database with model versioning data.
        
        Returns:
            Dictionary containing model versioning data references
        """
        session = self.get_session()
        
        # Create model families for different model types
        model_families = {}
        model_names = [
            "player_model_gk", "player_model_def", "player_model_mid", "player_model_fwd",
            "team_model", "transfer_optimizer", "injury_predictor"
        ]
        
        for model_name in model_names:
            family = create_model_family(session, model_name=model_name, num_versions=5)
            model_families[model_name] = family
        
        # Create A/B testing scenarios
        ab_tests = []
        for i in range(3):
            ab_test = create_ab_test_scenario(session)
            ab_tests.append(ab_test)
        
        # Create development pipeline
        dev_pipeline = create_model_development_pipeline(session)
        
        session.commit()
        
        seeded_data = {
            "model_families": model_families,
            "ab_tests": ab_tests,
            "dev_pipeline": dev_pipeline,
            "model_names": model_names,
        }
        
        self._seeded_data.update(seeded_data)
        logger.info(f"Seeded model versioning data: {len(model_names)} model families, {len(ab_tests)} A/B tests")
        return seeded_data
    
    def seed_feature_store_data(self) -> Dict[str, Any]:
        """
        Seed database with feature store data.
        
        Returns:
            Dictionary containing feature store data references
        """
        session = self.get_session()
        
        # Create feature computation pipelines
        feature_pipelines = []
        for i in range(3):
            pipeline = create_feature_computation_pipeline(session)
            feature_pipelines.append(pipeline)
        
        # Create performance test data
        performance_data = create_feature_performance_test_data(session)
        
        session.commit()
        
        seeded_data = {
            "feature_pipelines": feature_pipelines,
            "performance_data": performance_data,
        }
        
        self._seeded_data.update(seeded_data)
        logger.info(f"Seeded feature store data: {len(feature_pipelines)} pipelines")
        return seeded_data
    
    def seed_database_versioning_data(self) -> Dict[str, Any]:
        """
        Seed database with database versioning data.
        
        Returns:
            Dictionary containing database versioning data references
        """
        session = self.get_session()
        
        # Create migration timeline
        migration_timeline = create_migration_timeline(
            session, start_version="1.0.0", num_versions=10
        )
        
        # Create compatibility matrix
        compatibility_matrix = create_compatibility_matrix(session)
        
        # Create failed migration scenario
        failed_scenario = create_failed_migration_scenario(session)
        
        session.commit()
        
        seeded_data = {
            "migration_timeline": migration_timeline,
            "compatibility_matrix": compatibility_matrix,
            "failed_scenario": failed_scenario,
        }
        
        self._seeded_data.update(seeded_data)
        logger.info(f"Seeded database versioning data: {len(migration_timeline)} versions, {len(compatibility_matrix)} compatibility entries")
        return seeded_data
    
    def seed_comprehensive_data(self) -> Dict[str, Any]:
        """
        Seed database with comprehensive data for full integration testing.
        
        Returns:
            Dictionary containing all seeded data references
        """
        logger.info("Starting comprehensive data seeding...")
        
        # Seed all data types
        basic_data = self.seed_basic_data(num_players=200)
        season_data = self.seed_season_data(season="2324", num_gameweeks=20)
        model_data = self.seed_model_versioning_data()
        feature_data = self.seed_feature_store_data()
        db_version_data = self.seed_database_versioning_data()
        
        comprehensive_data = {
            "basic": basic_data,
            "season": season_data,
            "models": model_data,
            "features": feature_data,
            "database_versions": db_version_data,
            "summary": {
                "total_players": len(season_data["all_players"]),
                "total_gameweeks": season_data["num_gameweeks"],
                "total_models": len(model_data["model_families"]),
                "total_features": len(feature_data["feature_pipelines"]),
                "total_migrations": len(db_version_data["migration_timeline"]),
            }
        }
        
        self._seeded_data.update(comprehensive_data)
        logger.info("Completed comprehensive data seeding")
        return comprehensive_data
    
    def get_seeded_data(self, data_type: str = None) -> Dict[str, Any]:
        """
        Get references to seeded data.
        
        Args:
            data_type: Specific type of data to retrieve (e.g., 'players', 'models')
            
        Returns:
            Dictionary containing requested data references
        """
        if data_type:
            return self._seeded_data.get(data_type, {})
        return self._seeded_data.copy()
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of seeded data.
        
        Returns:
            Dictionary containing verification results
        """
        session = self.get_session()
        
        # Check table counts
        table_counts = {}
        for table in Base.metadata.tables:
            count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            table_counts[table] = count
        
        # Check relationships
        relationship_checks = {}
        
        # Verify player-attributes relationships
        players_with_attrs = session.execute(text("""
            SELECT COUNT(DISTINCT p.player_id) 
            FROM player p 
            JOIN player_attributes pa ON p.player_id = pa.player_id
        """)).scalar()
        
        relationship_checks["players_with_attributes"] = players_with_attrs
        
        # Verify model-version relationships
        models_with_versions = session.execute(text("""
            SELECT COUNT(DISTINCT mr.id) 
            FROM model_registry mr 
            JOIN model_version mv ON mr.id = mv.registry_id
        """)).scalar()
        
        relationship_checks["models_with_versions"] = models_with_versions
        
        verification_results = {
            "table_counts": table_counts,
            "relationship_checks": relationship_checks,
            "timestamp": datetime.now().isoformat(),
            "database_url": self.database_url,
        }
        
        logger.info(f"Data integrity verification completed: {table_counts}")
        return verification_results
    
    def export_test_data(self, output_file: Path) -> None:
        """
        Export seeded data metadata to file for test reproducibility.
        
        Args:
            output_file: Path to output JSON file
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "database_url": self.database_url,
            "seeded_data_summary": self.get_seeded_data(),
            "integrity_check": self.verify_data_integrity(),
        }
        
        # Remove complex objects that can't be serialized
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() 
                       if not k.startswith('_') and not callable(v)}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return f"<{obj.__class__.__name__} object>"
            else:
                return obj
        
        clean_data = clean_for_json(export_data)
        
        with open(output_file, 'w') as f:
            json.dump(clean_data, f, indent=2, default=str)
        
        logger.info(f"Exported test data metadata to {output_file}")


class TestDataPresets:
    """
    Predefined test data configurations for common scenarios.
    """
    
    @staticmethod
    def minimal_preset() -> Dict[str, Any]:
        """Minimal data for basic testing."""
        return {
            "num_players": 20,
            "num_gameweeks": 3,
            "include_models": False,
            "include_features": False,
            "include_db_versions": False,
        }
    
    @staticmethod
    def standard_preset() -> Dict[str, Any]:
        """Standard data for typical testing."""
        return {
            "num_players": 100,
            "num_gameweeks": 10,
            "include_models": True,
            "include_features": True,
            "include_db_versions": True,
            "num_model_families": 3,
            "num_ab_tests": 2,
        }
    
    @staticmethod
    def comprehensive_preset() -> Dict[str, Any]:
        """Comprehensive data for full integration testing."""
        return {
            "num_players": 500,
            "num_gameweeks": 38,  # Full season
            "include_models": True,
            "include_features": True,
            "include_db_versions": True,
            "num_model_families": 7,
            "num_ab_tests": 5,
            "include_performance_data": True,
        }
    
    @staticmethod
    def performance_preset() -> Dict[str, Any]:
        """Data optimized for performance testing."""
        return {
            "num_players": 1000,
            "num_gameweeks": 20,
            "include_models": True,
            "include_features": True,
            "include_db_versions": False,
            "high_volume_features": True,
            "cache_entries": 10000,
        }


def create_test_database(preset: str = "standard", database_url: str = None) -> TestDatabaseManager:
    """
    Create and seed a test database using a preset configuration.
    
    Args:
        preset: Preset configuration name ('minimal', 'standard', 'comprehensive', 'performance')
        database_url: Database URL (defaults to in-memory SQLite)
        
    Returns:
        Configured TestDatabaseManager instance
    """
    db_manager = TestDatabaseManager(database_url)
    db_manager.create_database()
    
    # Get preset configuration
    preset_configs = {
        "minimal": TestDataPresets.minimal_preset(),
        "standard": TestDataPresets.standard_preset(),
        "comprehensive": TestDataPresets.comprehensive_preset(),
        "performance": TestDataPresets.performance_preset(),
    }
    
    config = preset_configs.get(preset, TestDataPresets.standard_preset())
    
    # Seed data based on configuration
    if preset == "comprehensive":
        db_manager.seed_comprehensive_data()
    else:
        # Selective seeding based on config
        db_manager.seed_basic_data(num_players=config["num_players"])
        db_manager.seed_season_data(num_gameweeks=config["num_gameweeks"])
        
        if config.get("include_models", False):
            db_manager.seed_model_versioning_data()
        
        if config.get("include_features", False):
            db_manager.seed_feature_store_data()
        
        if config.get("include_db_versions", False):
            db_manager.seed_database_versioning_data()
    
    logger.info(f"Created test database with '{preset}' preset")
    return db_manager


def cleanup_test_databases():
    """
    Cleanup any lingering test database resources.
    
    This function can be called to ensure test databases are properly cleaned up,
    especially useful in CI environments or when running large test suites.
    """
    # Close any open SQLAlchemy engines
    from sqlalchemy.pool import StaticPool
    StaticPool._reset()
    
    # Clean up temporary files if any
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    test_db_files = temp_dir.glob("test_airsenal_*.db")
    
    for db_file in test_db_files:
        try:
            db_file.unlink()
            logger.info(f"Cleaned up test database file: {db_file}")
        except OSError:
            logger.warning(f"Could not clean up test database file: {db_file}")
    
    logger.info("Test database cleanup completed")


# Context manager for test database lifecycle
class TestDatabaseContext:
    """Context manager for test database lifecycle management."""
    
    def __init__(self, preset: str = "standard", database_url: str = None):
        self.preset = preset
        self.database_url = database_url
        self.db_manager = None
    
    def __enter__(self) -> TestDatabaseManager:
        """Create and seed test database."""
        self.db_manager = create_test_database(self.preset, self.database_url)
        return self.db_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup test database."""
        if self.db_manager:
            self.db_manager.destroy_database()
        cleanup_test_databases()


# Utility functions for test database operations
def quick_test_database(num_players: int = 20) -> TestDatabaseManager:
    """Quickly create a minimal test database for simple tests."""
    db_manager = TestDatabaseManager()
    db_manager.create_database()
    db_manager.seed_basic_data(num_players=num_players)
    return db_manager


def benchmark_database(size: str = "medium") -> TestDatabaseManager:
    """Create database optimized for benchmarking."""
    size_configs = {
        "small": {"num_players": 100, "num_gameweeks": 5},
        "medium": {"num_players": 500, "num_gameweeks": 15},
        "large": {"num_players": 1000, "num_gameweeks": 38},
    }
    
    config = size_configs.get(size, size_configs["medium"])
    
    db_manager = TestDatabaseManager()
    db_manager.create_database()
    db_manager.seed_basic_data(num_players=config["num_players"])
    db_manager.seed_season_data(num_gameweeks=config["num_gameweeks"])
    
    return db_manager