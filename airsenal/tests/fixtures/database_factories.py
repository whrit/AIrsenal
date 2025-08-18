"""
Factory_boy factories for Database Versioning components.

Provides realistic test data generation for database versioning, migration history,
and schema compatibility tracking used in database lifecycle management.
"""

import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Any

import factory

from airsenal.framework.schema import (
    DatabaseVersion,
    MigrationHistory,
    SchemaCompatibility,
)


class DatabaseVersionFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating DatabaseVersion instances with realistic version metadata."""

    class Meta:
        model = DatabaseVersion
        sqlalchemy_session_persistence = "commit"

    version = factory.LazyFunction(lambda: _generate_app_version())

    schema_version = factory.LazyAttribute(
        lambda obj: _generate_schema_version(obj.version)
    )

    app_version = factory.SelfAttribute("version")  # Same as version for simplicity

    migration_id = factory.LazyFunction(
        lambda: f"alembic_{hashlib.md5(f'migration_{random.randint(1000, 9999)}'.encode()).hexdigest()[:12]}"
    )

    # Migration metadata
    applied_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat()
    )

    applied_by = factory.fuzzy.FuzzyChoice(
        [
            "alembic_auto",
            "admin_user",
            "deployment_system",
            "developer_alice",
            "migration_script",
            "database_admin",
            "automated_deployment",
        ]
    )

    migration_type = factory.fuzzy.FuzzyChoice(
        ["initial", "upgrade", "rollback", "manual"], weights=[1, 8, 2, 1]
    )  # Mostly upgrades

    migration_source = factory.fuzzy.FuzzyChoice(
        ["alembic", "manual", "automated"], weights=[7, 2, 1]
    )  # Mostly alembic

    # Migration details
    migration_description = factory.LazyAttribute(
        lambda obj: _generate_migration_description(obj.version, obj.migration_type)
    )

    migration_checksum = factory.LazyFunction(
        lambda: hashlib.md5(
            f"migration_content_{random.randint(10000, 99999)}".encode()
        ).hexdigest()
    )

    rollback_info = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_rollback_info(obj.migration_type))
    )

    # Compatibility information
    min_app_version = factory.LazyAttribute(
        lambda obj: _generate_min_version(obj.version)
    )

    max_app_version = factory.LazyAttribute(
        lambda obj: _generate_max_version(obj.version)
    )

    compatibility_notes = factory.LazyAttribute(
        lambda obj: _generate_compatibility_notes(obj.version)
    )

    # Status and validation
    is_active = True

    validation_status = factory.fuzzy.FuzzyChoice(
        ["validated", "pending", "failed"], weights=[8, 1, 1]
    )  # Mostly validated

    validation_errors = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_validation_errors())
        if obj.validation_status == "failed"
        else None
    )

    # Performance impact tracking
    migration_duration_seconds = factory.fuzzy.FuzzyFloat(
        1.0, 300.0
    )  # 1 second to 5 minutes

    affected_tables = factory.LazyAttribute(
        lambda obj: _generate_affected_tables(obj.migration_description)
    )

    records_migrated = factory.LazyAttribute(
        lambda obj: random.randint(0, 1000000)
        if "data" in obj.migration_description.lower()
        else 0
    )


class MigrationHistoryFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating MigrationHistory instances with detailed migration execution data."""

    class Meta:
        model = MigrationHistory
        sqlalchemy_session_persistence = "commit"

    database_version = factory.SubFactory(DatabaseVersionFactory)

    # Migration identification
    migration_name = factory.LazyAttribute(
        lambda obj: _generate_migration_name(obj.database_version.version)
    )

    migration_hash = factory.LazyFunction(
        lambda: hashlib.sha256(
            f"migration_{random.randint(100000, 999999)}".encode()
        ).hexdigest()[:16]
    )

    sequence_number = factory.Sequence(lambda n: n + 1)

    # Execution details
    executed_at = factory.LazyFunction(
        lambda: (
            datetime.now() - timedelta(minutes=random.randint(0, 43200))
        ).isoformat()  # Last 30 days
    )

    execution_duration_seconds = factory.fuzzy.FuzzyFloat(
        0.5, 120.0
    )  # 0.5 sec to 2 minutes

    executed_by = factory.fuzzy.FuzzyChoice(
        [
            "alembic",
            "migration_script",
            "database_admin",
            "automated_system",
            "deployment_pipeline",
            "developer_local",
            "testing_framework",
        ]
    )

    execution_context = factory.fuzzy.FuzzyChoice(
        ["deployment", "development", "testing", "rollback", "hotfix"]
    )

    # Migration content and impact
    migration_sql = factory.LazyAttribute(
        lambda obj: _generate_migration_sql(obj.migration_name)
    )

    tables_affected = factory.LazyAttribute(
        lambda obj: _generate_tables_from_sql(obj.migration_sql)
    )

    records_before = factory.fuzzy.FuzzyInteger(0, 500000)

    records_after = factory.LazyAttribute(
        lambda obj: obj.records_before + random.randint(-1000, 10000)
    )

    records_changed = factory.LazyAttribute(
        lambda obj: abs(obj.records_after - obj.records_before)
    )

    # Success and error tracking
    status = factory.fuzzy.FuzzyChoice(
        ["success", "failed", "partial", "rolled_back"], weights=[85, 10, 3, 2]
    )  # Mostly successful

    error_message = factory.LazyAttribute(
        lambda obj: _generate_error_message()
        if obj.status in ["failed", "partial"]
        else None
    )

    warning_count = factory.LazyAttribute(
        lambda obj: random.randint(1, 5) if obj.status == "partial" else 0
    )

    # Rollback information
    rollback_sql = factory.LazyAttribute(
        lambda obj: _generate_rollback_sql(obj.migration_sql)
        if random.random() < 0.7
        else None
    )

    rollback_tested = factory.LazyAttribute(
        lambda obj: random.random() < 0.8 if obj.rollback_sql else False
    )

    rollback_notes = factory.LazyAttribute(
        lambda obj: _generate_rollback_notes() if obj.rollback_sql else None
    )

    # Performance and monitoring
    memory_usage_mb = factory.fuzzy.FuzzyFloat(10.0, 500.0)

    cpu_usage_percent = factory.fuzzy.FuzzyFloat(5.0, 95.0)

    lock_conflicts = factory.LazyAttribute(
        lambda obj: random.randint(0, 3) if "alter" in obj.migration_sql.lower() else 0
    )

    # Dependencies and prerequisites
    depends_on = factory.LazyAttribute(
        lambda obj: _generate_dependencies(obj.migration_name)
    )

    blocks = factory.LazyAttribute(
        lambda obj: _generate_blocking_migrations(obj.migration_name)
    )


class SchemaCompatibilityFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating SchemaCompatibility instances for version compatibility tracking."""

    class Meta:
        model = SchemaCompatibility
        sqlalchemy_session_persistence = "commit"

    app_version_min = factory.LazyFunction(lambda: _generate_app_version())

    app_version_max = factory.LazyAttribute(
        lambda obj: _increment_version(obj.app_version_min, increment_type="minor")
    )

    schema_version_min = factory.LazyAttribute(
        lambda obj: _generate_schema_version(obj.app_version_min)
    )

    schema_version_max = factory.LazyAttribute(
        lambda obj: _generate_schema_version(obj.app_version_max)
    )

    # Compatibility metadata
    compatibility_level = factory.fuzzy.FuzzyChoice(
        ["full", "limited", "deprecated", "incompatible"], weights=[6, 2, 1, 1]
    )  # Mostly full compatibility

    compatibility_notes = factory.LazyAttribute(
        lambda obj: _generate_compatibility_notes_detailed(obj.compatibility_level)
    )

    migration_required = factory.LazyAttribute(
        lambda obj: obj.compatibility_level in ["limited", "incompatible"]
    )

    migration_priority = factory.LazyAttribute(
        lambda obj: _determine_migration_priority(obj.compatibility_level)
    )

    # Validation and testing
    tested_combinations = factory.LazyAttribute(
        lambda obj: json.dumps(
            _generate_tested_combinations(obj.app_version_min, obj.app_version_max)
        )
    )

    known_issues = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_known_issues(obj.compatibility_level))
    )

    workarounds = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_workarounds(obj.compatibility_level))
    )

    # Lifecycle management
    created_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
    )

    updated_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
    )

    deprecated_at = factory.LazyAttribute(
        lambda obj: (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
        if obj.compatibility_level == "deprecated"
        else None
    )

    removed_at = factory.LazyAttribute(
        lambda obj: (
            datetime.now() + timedelta(days=random.randint(30, 180))
        ).isoformat()
        if obj.compatibility_level == "incompatible"
        else None
    )

    # Performance impact
    performance_impact = factory.LazyAttribute(
        lambda obj: _determine_performance_impact(obj.compatibility_level)
    )

    performance_notes = factory.LazyAttribute(
        lambda obj: _generate_performance_notes(obj.performance_impact)
    )


# Specialized factories for specific scenarios
class SuccessfulMigrationHistoryFactory(MigrationHistoryFactory):
    """Factory for successful migration scenarios."""

    status = "success"
    error_message = None
    warning_count = 0
    execution_duration_seconds = factory.fuzzy.FuzzyFloat(0.5, 30.0)  # Quick execution


class FailedMigrationHistoryFactory(MigrationHistoryFactory):
    """Factory for failed migration scenarios."""

    status = "failed"
    error_message = factory.LazyFunction(lambda: _generate_error_message())
    warning_count = factory.fuzzy.FuzzyInteger(3, 10)
    rollback_tested = True


class HotfixDatabaseVersionFactory(DatabaseVersionFactory):
    """Factory for hotfix/patch version scenarios."""

    version = factory.LazyFunction(lambda: _generate_patch_version())
    migration_type = "upgrade"
    migration_description = factory.LazyAttribute(
        lambda obj: f"Hotfix for version {obj.version}: Critical bug fixes and security patches"
    )
    migration_duration_seconds = factory.fuzzy.FuzzyFloat(5.0, 60.0)  # Quick hotfix


class IncompatibleSchemaCompatibilityFactory(SchemaCompatibilityFactory):
    """Factory for incompatible schema scenarios."""

    compatibility_level = "incompatible"
    migration_required = True
    migration_priority = "critical"
    performance_impact = "high"


# Helper functions for realistic data generation
def _generate_app_version() -> str:
    """Generate realistic application version number."""
    major = random.randint(1, 3)
    minor = random.randint(0, 15)
    patch = random.randint(0, 25)
    return f"{major}.{minor}.{patch}"


def _generate_patch_version() -> str:
    """Generate patch version for hotfixes."""
    base_version = _generate_app_version()
    parts = base_version.split(".")
    patch = int(parts[2]) + random.randint(1, 3)
    return f"{parts[0]}.{parts[1]}.{patch}"


def _generate_schema_version(app_version: str) -> str:
    """Generate schema version based on app version."""
    parts = app_version.split(".")
    # Schema version increments less frequently than app version
    schema_major = parts[0]
    schema_minor = str(int(parts[1]) // 2)  # Half the frequency
    return f"{schema_major}.{schema_minor}"


def _increment_version(version: str, increment_type: str = "minor") -> str:
    """Increment version number."""
    parts = list(map(int, version.split(".")))

    if increment_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif increment_type == "minor":
        parts[1] += random.randint(1, 3)
        parts[2] = 0
    else:  # patch
        parts[2] += random.randint(1, 5)

    return ".".join(map(str, parts))


def _generate_min_version(current_version: str) -> str:
    """Generate minimum compatible version."""
    parts = list(map(int, current_version.split(".")))
    # Min version is usually 1-2 minor versions behind
    min_minor = max(0, parts[1] - random.randint(1, 3))
    return f"{parts[0]}.{min_minor}.0"


def _generate_max_version(current_version: str) -> str:
    """Generate maximum compatible version."""
    parts = list(map(int, current_version.split(".")))
    # Max version is usually 1-2 minor versions ahead
    max_minor = parts[1] + random.randint(1, 3)
    return f"{parts[0]}.{max_minor}.99"


def _generate_migration_description(version: str, migration_type: str) -> str:
    """Generate realistic migration description."""
    descriptions = {
        "initial": f"Initial database schema creation for AIrsenal v{version}",
        "upgrade": f"Schema upgrade to v{version}: {random.choice(_get_upgrade_features())}",
        "rollback": f"Rollback from v{version} due to {random.choice(['compatibility issues', 'data integrity concerns', 'performance degradation'])}",
        "manual": f"Manual schema adjustment for v{version}: {random.choice(['data cleanup', 'index optimization', 'constraint fixes'])}",
    }
    return descriptions.get(migration_type, f"Database migration for v{version}")


def _get_upgrade_features() -> list[str]:
    """Get list of realistic upgrade features."""
    return [
        "Added extended PlayerAttributes with xG metrics",
        "Implemented model versioning system",
        "Added feature store infrastructure",
        "Enhanced database versioning tracking",
        "Added structured logging support",
        "Implemented Redis caching layer",
        "Added A/B testing framework",
        "Enhanced transfer optimization algorithms",
        "Added injury prediction models",
        "Improved fixture difficulty calculations",
    ]


def _generate_compatibility_notes(version: str) -> str:
    """Generate compatibility notes."""
    notes = [
        f"Version {version} maintains backward compatibility with previous minor versions",
        f"Database schema changes in {version} require migration",
        f"New features in {version} are optional and don't break existing functionality",
        f"Performance optimizations in {version} may require configuration updates",
    ]
    return random.choice(notes)


def _generate_rollback_info(migration_type: str) -> dict[str, Any]:
    """Generate rollback information."""
    if migration_type == "rollback":
        return {
            "rollback_reason": "Migration validation failed",
            "rollback_steps": [
                "Stop application",
                "Execute rollback SQL",
                "Restart with previous version",
            ],
            "data_loss_risk": "low",
            "estimated_downtime_minutes": random.randint(5, 30),
        }

    return {
        "rollback_available": True,
        "rollback_tested": random.choice([True, False]),
        "rollback_complexity": random.choice(["low", "medium", "high"]),
        "estimated_rollback_time_minutes": random.randint(2, 60),
    }


def _generate_validation_errors() -> list[str]:
    """Generate realistic validation errors."""
    errors = [
        "Foreign key constraint violation in player_attributes table",
        "Duplicate index names detected",
        "Column type mismatch in migration script",
        "Missing NOT NULL constraint on required field",
        "Table reference to non-existent index",
        "Data type incompatibility detected",
    ]
    return random.sample(errors, k=random.randint(1, 3))


def _generate_affected_tables(description: str) -> str:
    """Generate list of affected tables based on migration description."""
    all_tables = [
        "player",
        "player_attributes",
        "team",
        "fixture",
        "player_score",
        "model_registry",
        "model_version",
        "model_artifact",
        "model_performance",
        "feature_definition",
        "computed_feature",
        "feature_cache",
        "feature_timeseries",
        "database_version",
        "migration_history",
        "schema_compatibility",
    ]

    # Determine affected tables based on description
    if "player" in description.lower():
        affected = ["player", "player_attributes", "player_score"]
    elif "model" in description.lower():
        affected = [
            "model_registry",
            "model_version",
            "model_artifact",
            "model_performance",
        ]
    elif "feature" in description.lower():
        affected = ["feature_definition", "computed_feature", "feature_cache"]
    else:
        affected = random.sample(all_tables, k=random.randint(1, 4))

    return ",".join(affected)


def _generate_migration_name(version: str) -> str:
    """Generate realistic migration name."""
    migration_types = [
        "add_extended_player_attributes",
        "create_model_versioning_tables",
        "implement_feature_store_schema",
        "add_database_versioning_support",
        "optimize_player_indexes",
        "add_redis_cache_support",
        "implement_ab_testing_framework",
        "enhance_transfer_optimization",
        "add_injury_prediction_tables",
        "update_fixture_difficulty_calculation",
    ]

    base_name = random.choice(migration_types)
    version_suffix = version.replace(".", "_")
    return f"{base_name}_v{version_suffix}"


def _generate_migration_sql(migration_name: str) -> str:
    """Generate realistic migration SQL based on migration name."""
    sql_templates = {
        "add_extended_player_attributes": """
            ALTER TABLE player_attributes
            ADD COLUMN xg_per_90 FLOAT,
            ADD COLUMN xa_per_90 FLOAT,
            ADD COLUMN form_3_games FLOAT,
            ADD COLUMN is_penalty_taker BOOLEAN DEFAULT FALSE;

            CREATE INDEX ix_form_3_games ON player_attributes(form_3_games);
        """,
        "create_model_versioning_tables": """
            CREATE TABLE model_registry (
                id INTEGER PRIMARY KEY,
                model_name VARCHAR(100),
                model_type VARCHAR(100),
                created_at VARCHAR(100)
            );

            CREATE TABLE model_version (
                id INTEGER PRIMARY KEY,
                registry_id INTEGER,
                version VARCHAR(100),
                FOREIGN KEY (registry_id) REFERENCES model_registry(id)
            );
        """,
        "implement_feature_store_schema": """
            CREATE TABLE feature_definition (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                feature_type VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE
            );

            CREATE TABLE computed_feature (
                id INTEGER PRIMARY KEY,
                feature_definition_id INTEGER,
                entity_id INTEGER,
                value FLOAT,
                FOREIGN KEY (feature_definition_id) REFERENCES feature_definition(id)
            );
        """,
    }

    for key, template in sql_templates.items():
        if key in migration_name:
            return template.strip()

    return f"-- Migration SQL for {migration_name}\nALTER TABLE player ADD COLUMN new_field VARCHAR(100);"


def _generate_tables_from_sql(sql: str) -> str:
    """Extract table names from SQL."""
    tables = []
    sql_lower = sql.lower()

    # Simple extraction of table names
    common_tables = [
        "player",
        "player_attributes",
        "team",
        "fixture",
        "model_registry",
        "model_version",
        "feature_definition",
        "computed_feature",
    ]

    for table in common_tables:
        if table in sql_lower:
            tables.append(table)

    return ",".join(set(tables)) if tables else "unknown"


def _generate_error_message() -> str:
    """Generate realistic migration error message."""
    errors = [
        "ERROR: column 'new_field' of relation 'player_attributes' already exists",
        "ERROR: foreign key constraint 'fk_player_id' cannot be implemented",
        "ERROR: duplicate key value violates unique constraint 'player_name_unique'",
        "ERROR: column 'old_field' does not exist",
        "ERROR: syntax error at or near 'CRATE' (should be CREATE)",
        "ERROR: relation 'non_existent_table' does not exist",
        "ERROR: out of memory for query result",
        "ERROR: lock timeout exceeded",
        "ERROR: permission denied for table 'system_table'",
    ]
    return random.choice(errors)


def _generate_rollback_sql(migration_sql: str) -> str:
    """Generate rollback SQL based on migration SQL."""
    if "ADD COLUMN" in migration_sql:
        return "ALTER TABLE player_attributes DROP COLUMN IF EXISTS xg_per_90, DROP COLUMN IF EXISTS xa_per_90;"
    if "CREATE TABLE" in migration_sql:
        return "DROP TABLE IF EXISTS model_registry CASCADE; DROP TABLE IF EXISTS model_version CASCADE;"
    if "CREATE INDEX" in migration_sql:
        return "DROP INDEX IF EXISTS ix_form_3_games;"
    return "-- Rollback SQL not available for this migration"


def _generate_rollback_notes() -> str:
    """Generate rollback notes."""
    notes = [
        "Rollback tested in development environment",
        "Data backup created before rollback execution",
        "Rollback may result in loss of recent feature data",
        "Application restart required after rollback",
        "Monitor performance after rollback completion",
    ]
    return random.choice(notes)


def _generate_dependencies(migration_name: str) -> str:
    """Generate migration dependencies."""
    base_migrations = [
        "initial_schema_v1_0_0",
        "add_player_tables_v1_1_0",
        "create_fixtures_v1_2_0",
    ]

    if "model" in migration_name:
        return "add_player_tables_v1_1_0,create_fixtures_v1_2_0"
    if "feature" in migration_name:
        return "add_player_tables_v1_1_0"
    return random.choice(base_migrations)


def _generate_blocking_migrations(migration_name: str) -> str:
    """Generate migrations that are blocked by this one."""
    if "extended_player" in migration_name:
        return "update_player_stats_v2_1_0,add_player_analytics_v2_2_0"
    if "model_versioning" in migration_name:
        return "implement_ab_testing_v2_3_0"
    return ""


def _generate_compatibility_notes_detailed(compatibility_level: str) -> str:
    """Generate detailed compatibility notes."""
    notes = {
        "full": "Complete compatibility maintained. All features work seamlessly across version range.",
        "limited": "Most features compatible. Some new features may not work with older versions.",
        "deprecated": "Compatibility maintained but deprecated. Migration recommended within 6 months.",
        "incompatible": "Breaking changes present. Immediate migration required.",
    }
    return notes.get(compatibility_level, "Compatibility status unknown")


def _determine_migration_priority(compatibility_level: str) -> str:
    """Determine migration priority based on compatibility level."""
    priority_map = {
        "full": "low",
        "limited": "normal",
        "deprecated": "high",
        "incompatible": "critical",
    }
    return priority_map.get(compatibility_level, "normal")


def _generate_tested_combinations(min_version: str, max_version: str) -> list[str]:
    """Generate list of tested version combinations."""
    combinations = []

    # Add some intermediate versions
    for _ in range(random.randint(2, 5)):
        combinations.append(f"{min_version} -> {max_version}")
        combinations.append(
            f"schema_{_generate_schema_version(min_version)} + app_{max_version}"
        )

    return combinations


def _generate_known_issues(compatibility_level: str) -> list[str]:
    """Generate known compatibility issues."""
    if compatibility_level == "full":
        return []

    issues = [
        "Performance degradation with large datasets",
        "Some API endpoints return deprecated format",
        "New feature toggles not recognized by older versions",
        "Database connection pooling issues",
        "Cache invalidation problems with mixed versions",
    ]

    num_issues = {"limited": 1, "deprecated": 2, "incompatible": 3}.get(
        compatibility_level, 0
    )
    return random.sample(issues, k=min(num_issues, len(issues)))


def _generate_workarounds(compatibility_level: str) -> list[str]:
    """Generate available workarounds."""
    if compatibility_level in ["full", "incompatible"]:
        return []

    workarounds = [
        "Use legacy API endpoints for backward compatibility",
        "Disable new features in configuration",
        "Run database schema validation before startup",
        "Use compatibility mode flag in application settings",
        "Implement gradual rollout strategy",
    ]

    num_workarounds = {"limited": 2, "deprecated": 1}.get(compatibility_level, 0)
    return random.sample(workarounds, k=num_workarounds)


def _determine_performance_impact(compatibility_level: str) -> str:
    """Determine performance impact based on compatibility level."""
    impact_map = {
        "full": "none",
        "limited": "low",
        "deprecated": "medium",
        "incompatible": "high",
    }
    return impact_map.get(compatibility_level, "none")


def _generate_performance_notes(performance_impact: str) -> str:
    """Generate performance impact notes."""
    notes = {
        "none": "No measurable performance impact expected",
        "low": "Minor performance overhead during compatibility checks",
        "medium": "Noticeable performance impact due to legacy support overhead",
        "high": "Significant performance degradation. Migration strongly recommended",
    }
    return notes.get(performance_impact)


# Batch creation functions for testing scenarios
def create_migration_timeline(
    session, start_version: str = "1.0.0", num_versions: int = 10
) -> list[DatabaseVersion]:
    """Create a realistic migration timeline with multiple versions."""
    versions = []
    current_version = start_version

    for i in range(num_versions):
        # Create database version
        db_version = DatabaseVersionFactory(
            session=session,
            version=current_version,
            applied_at=(datetime.now() - timedelta(days=num_versions - i)).isoformat(),
        )
        versions.append(db_version)

        # Create migration history entries
        for j in range(random.randint(1, 3)):  # 1-3 migrations per version
            MigrationHistoryFactory(
                session=session, database_version=db_version, sequence_number=j + 1
            )

        # Create schema compatibility
        SchemaCompatibilityFactory(
            session=session,
            app_version_min=current_version,
            app_version_max=_increment_version(current_version, "minor"),
        )

        # Increment version for next iteration
        current_version = _increment_version(current_version, "patch")

    return versions


def create_compatibility_matrix(session) -> list[SchemaCompatibility]:
    """Create a compatibility matrix for multiple version combinations."""
    versions = ["1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.1.0"]
    compatibility_entries = []

    for i, min_version in enumerate(versions):
        for j, max_version in enumerate(versions[i:], i):
            if i == j:
                continue  # Skip same version combinations

            # Determine compatibility level based on version difference
            min_parts = list(map(int, min_version.split(".")))
            max_parts = list(map(int, max_version.split(".")))

            if max_parts[0] > min_parts[0]:  # Major version change
                level = "incompatible" if max_parts[0] - min_parts[0] > 1 else "limited"
            elif max_parts[1] - min_parts[1] > 2:  # Too many minor versions
                level = "deprecated"
            else:
                level = "full"

            compat = SchemaCompatibilityFactory(
                session=session,
                app_version_min=min_version,
                app_version_max=max_version,
                compatibility_level=level,
            )
            compatibility_entries.append(compat)

    return compatibility_entries


def create_failed_migration_scenario(session) -> dict[str, Any]:
    """Create a scenario with failed migrations and rollbacks."""
    # Create failed migration
    failed_version = DatabaseVersionFactory(
        session=session, version="2.0.0", validation_status="failed"
    )

    failed_history = FailedMigrationHistoryFactory(
        session=session, database_version=failed_version, execution_context="deployment"
    )

    # Create rollback
    rollback_version = DatabaseVersionFactory(
        session=session,
        version="1.9.9",
        migration_type="rollback",
        applied_at=datetime.now().isoformat(),
    )

    rollback_history = MigrationHistoryFactory(
        session=session,
        database_version=rollback_version,
        status="success",
        execution_context="rollback",
    )

    return {
        "failed_version": failed_version,
        "failed_history": failed_history,
        "rollback_version": rollback_version,
        "rollback_history": rollback_history,
    }
