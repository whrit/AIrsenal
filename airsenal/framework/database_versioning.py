"""
Database versioning system for AIrsenal.

This module provides utilities for tracking database schema versions,
ensuring compatibility between application and database versions,
and managing migrations with proper rollback capabilities.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

try:
    from packaging import version as pkg_version
except ImportError:
    # Fallback for basic version comparison
    class _FallbackPkgVersion:
        @staticmethod
        def parse(version_string: str):
            # Simple version parsing fallback
            parts = version_string.split(".")
            return SimpleVersion([int(p) for p in parts[:3] if p.isdigit()])

    pkg_version = _FallbackPkgVersion()  # type: ignore


class SimpleVersion:
    def __init__(self, parts: list[int]):
        self.major = parts[0] if len(parts) > 0 else 0
        self.minor = parts[1] if len(parts) > 1 else 0
        self.micro = parts[2] if len(parts) > 2 else 0

    def __lt__(self, other):
        return (self.major, self.minor, self.micro) < (
            other.major,
            other.minor,
            other.micro,
        )

    def __le__(self, other):
        return (self.major, self.minor, self.micro) <= (
            other.major,
            other.minor,
            other.micro,
        )

    def __gt__(self, other):
        return (self.major, self.minor, self.micro) > (
            other.major,
            other.minor,
            other.micro,
        )

    def __ge__(self, other):
        return (self.major, self.minor, self.micro) >= (
            other.major,
            other.minor,
            other.micro,
        )

    def __eq__(self, other):
        return (self.major, self.minor, self.micro) == (
            other.major,
            other.minor,
            other.micro,
        )


from sqlalchemy import desc, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from airsenal.framework.schema import (
    DatabaseVersion,
    MigrationHistory,
    SchemaCompatibility,
)

# Current application version - imported from pyproject.toml
CURRENT_APP_VERSION = "1.11.0"

# Current schema version - increment this when schema changes
CURRENT_SCHEMA_VERSION = "1.11.0"

# Minimum supported schema version for backward compatibility
MIN_SUPPORTED_SCHEMA_VERSION = "1.10.0"

logger = logging.getLogger(__name__)


class DatabaseVersionError(Exception):
    """Raised when database version compatibility issues are detected."""


class MigrationError(Exception):
    """Raised when migration operations fail."""


def get_current_app_version() -> str:
    """
    Get the current application version from pyproject.toml or package metadata.

    Returns:
        str: Current application version (e.g., "1.11.0")
    """
    try:
        # Try to get version from pyproject.toml
        toml_lib: Any = None
        try:
            import tomllib

            toml_lib = tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli

                toml_lib = tomli  # fallback for older Python
            except ImportError:
                pass

        if toml_lib is not None:
            import pathlib

            project_root = pathlib.Path(__file__).parent.parent.parent
            pyproject_path = project_root / "pyproject.toml"

            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    pyproject_data = toml_lib.load(f)
                    return pyproject_data.get("project", {}).get(
                        "version", CURRENT_APP_VERSION
                    )
    except (ImportError, FileNotFoundError, KeyError):
        pass

    # Fallback to package metadata
    try:
        import importlib.metadata

        return importlib.metadata.version("airsenal")
    except Exception:
        # Final fallback to hardcoded version
        return CURRENT_APP_VERSION


def get_current_database_version(dbsession: Session) -> DatabaseVersion | None:
    """
    Get the current active database version from the database.

    Args:
        dbsession: SQLAlchemy database session

    Returns:
        DatabaseVersion object if found, None if no version exists
    """
    try:
        return (
            dbsession.query(DatabaseVersion)
            .filter_by(is_active=True)
            .order_by(desc(DatabaseVersion.applied_at))
            .first()
        )
    except SQLAlchemyError as e:
        logger.warning(f"Could not query database version (table may not exist): {e}")
        return None


def create_initial_database_version(
    dbsession: Session,
    app_version: str | None = None,
    schema_version: str | None = None,
    applied_by: str = "system",
    migration_description: str = "Initial database version creation",
) -> DatabaseVersion:
    """
    Create the initial database version record.

    Args:
        dbsession: SQLAlchemy database session
        app_version: Application version (defaults to current)
        schema_version: Schema version (defaults to current)
        applied_by: User/system that applied the version
        migration_description: Description of the migration

    Returns:
        DatabaseVersion object that was created
    """
    if app_version is None:
        app_version = get_current_app_version()
    if schema_version is None:
        schema_version = CURRENT_SCHEMA_VERSION

    now = datetime.now(timezone.utc).isoformat()

    db_version = DatabaseVersion(
        version=app_version,
        schema_version=schema_version,
        app_version=app_version,
        applied_at=now,
        applied_by=applied_by,
        migration_type="initial",
        migration_source="manual",
        migration_description=migration_description,
        min_app_version=app_version,
        max_app_version=None,  # No upper limit for initial version
        is_active=True,
        validation_status="validated",
        migration_duration_seconds=0.0,
        affected_tables="all",
        records_migrated=0,
    )

    dbsession.add(db_version)
    dbsession.commit()

    logger.info(f"Created initial database version: {db_version}")
    return db_version


def check_version_compatibility(
    app_version: str, schema_version: str, dbsession: Session
) -> tuple[bool, str, list[str]]:
    """
    Check if the given application and schema versions are compatible.

    Args:
        app_version: Application version to check
        schema_version: Database schema version to check
        dbsession: SQLAlchemy database session

    Returns:
        Tuple of (is_compatible, compatibility_level, warnings)
    """
    try:
        # Check explicit compatibility matrix first
        compatibility = (
            dbsession.query(SchemaCompatibility)
            .filter(
                SchemaCompatibility.app_version_min <= app_version,
                SchemaCompatibility.app_version_max >= app_version,
                SchemaCompatibility.schema_version_min <= schema_version,
                SchemaCompatibility.schema_version_max >= schema_version,
            )
            .first()
        )

        warnings = []

        if compatibility:
            level = compatibility.compatibility_level
            is_compatible = level in ["full", "limited"]

            if level == "limited":
                warnings.append(
                    f"Limited compatibility: {compatibility.compatibility_notes}"
                )
            if compatibility.known_issues:
                issues = json.loads(compatibility.known_issues)
                warnings.extend([f"Known issue: {issue}" for issue in issues])

            return is_compatible, level, warnings

    except SQLAlchemyError as e:
        logger.warning(f"Could not check compatibility matrix: {e}")

    # Fallback to version comparison logic
    return _check_version_compatibility_fallback(app_version, schema_version)


def _check_version_compatibility_fallback(
    app_version: str, schema_version: str
) -> tuple[bool, str, list[str]]:
    """
    Fallback compatibility check using semantic versioning rules.

    Args:
        app_version: Application version
        schema_version: Schema version

    Returns:
        Tuple of (is_compatible, compatibility_level, warnings)
    """
    try:
        app_ver = pkg_version.parse(app_version)
        schema_ver = pkg_version.parse(schema_version)
        min_supported = pkg_version.parse(MIN_SUPPORTED_SCHEMA_VERSION)

        warnings = []

        # Schema is too old
        if schema_ver < min_supported:
            return (
                False,
                "incompatible",
                ["Schema version is too old and no longer supported"],
            )

        # Schema is newer than app - may indicate app needs updating
        if schema_ver > app_ver:
            # Allow minor version differences
            if (
                schema_ver.major == app_ver.major
                and (schema_ver.minor - app_ver.minor) <= 2
            ):
                warnings.append(
                    "Schema is newer than application - consider updating the application"
                )
                return True, "limited", warnings
            return (
                False,
                "incompatible",
                ["Schema version is too new for this application version"],
            )

        # App is newer than schema - check if migration is needed
        if app_ver > schema_ver:
            if app_ver.major == schema_ver.major:
                if (app_ver.minor - schema_ver.minor) <= 3:
                    warnings.append(
                        "Application is newer than schema - migration may be needed"
                    )
                    return True, "limited", warnings
                warnings.append(
                    "Significant version gap - migration strongly recommended"
                )
                return True, "limited", warnings
            return (
                False,
                "incompatible",
                ["Major version mismatch - migration required"],
            )

        # Versions match exactly
        return True, "full", warnings

    except Exception as e:
        logger.error(f"Error parsing versions: {e}")
        return False, "incompatible", [f"Could not parse version numbers: {e}"]


def validate_database_on_startup(
    dbsession: Session, force_migration: bool = False, error_on_mismatch: bool = True
) -> bool:
    """
    Validate database version compatibility on application startup.

    Args:
        dbsession: SQLAlchemy database session
        force_migration: If True, attempt automatic migration
        error_on_mismatch: If True, raise error on version mismatch

    Returns:
        bool: True if database is compatible, False otherwise

    Raises:
        DatabaseVersionError: If versions are incompatible and error_on_mismatch is True
    """
    current_app_ver = get_current_app_version()

    # Get current database version
    db_version = get_current_database_version(dbsession)

    if db_version is None:
        # No version tracking exists - create initial version
        logger.info("No database version found. Creating initial version record.")
        try:
            create_initial_database_version(
                dbsession=dbsession,
                app_version=current_app_ver,
                schema_version=CURRENT_SCHEMA_VERSION,
                applied_by="startup_validation",
                migration_description="Initial version created during startup validation",
            )
            return True
        except Exception as e:
            error_msg = f"Failed to create initial database version: {e}"
            logger.error(error_msg)
            if error_on_mismatch:
                raise DatabaseVersionError(error_msg)
            return False

    # Check compatibility
    is_compatible, level, warnings = check_version_compatibility(
        current_app_ver, db_version.schema_version, dbsession
    )

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    if is_compatible:
        if level == "full":
            logger.info(
                f"Database version {db_version.schema_version} is fully compatible with app version {current_app_ver}"
            )
        else:
            logger.info(
                f"Database version {db_version.schema_version} has {level} compatibility with app version {current_app_ver}"
            )
        return True
    error_msg = (
        f"Database version {db_version.schema_version} is incompatible with "
        f"application version {current_app_ver}. "
        f"Compatibility level: {level}. "
        f"Warnings: {'; '.join(warnings)}"
    )
    logger.error(error_msg)

    if force_migration:
        logger.info("Attempting automatic migration...")
        try:
            success = attempt_auto_migration(dbsession, current_app_ver)
            if success:
                logger.info("Automatic migration completed successfully")
                return True
            logger.error("Automatic migration failed")
        except Exception as e:
            logger.error(f"Automatic migration failed with error: {e}")

    if error_on_mismatch:
        raise DatabaseVersionError(error_msg)
    return False


def record_migration(
    dbsession: Session,
    migration_name: str,
    new_version: str,
    new_schema_version: str,
    migration_type: str = "upgrade",
    migration_source: str = "manual",
    executed_by: str = "system",
    execution_context: str = "deployment",
    migration_sql: str | None = None,
    rollback_sql: str | None = None,
    tables_affected: list[str] | None = None,
    execution_duration: float | None = None,
    description: str | None = None,
) -> tuple[DatabaseVersion, MigrationHistory]:
    """
    Record a successful migration in the database.

    Args:
        dbsession: SQLAlchemy database session
        migration_name: Human-readable name for the migration
        new_version: New application version
        new_schema_version: New schema version
        migration_type: Type of migration ("upgrade", "rollback", "manual")
        migration_source: Source of migration ("alembic", "manual", "automated")
        executed_by: User or system that executed migration
        execution_context: Context of execution
        migration_sql: SQL executed during migration
        rollback_sql: SQL for rolling back migration
        tables_affected: List of table names affected
        execution_duration: Duration in seconds
        description: Migration description

    Returns:
        Tuple of (DatabaseVersion, MigrationHistory) objects created
    """
    now = datetime.now(timezone.utc).isoformat()

    # Deactivate previous version
    previous_version = get_current_database_version(dbsession)
    if previous_version:
        previous_version.is_active = False

    # Create migration hash
    migration_content = f"{migration_name}:{new_version}:{new_schema_version}:{now}"
    migration_hash = hashlib.md5(migration_content.encode()).hexdigest()

    # Create new database version
    db_version = DatabaseVersion(
        version=new_version,
        schema_version=new_schema_version,
        app_version=new_version,
        applied_at=now,
        applied_by=executed_by,
        migration_type=migration_type,
        migration_source=migration_source,
        migration_description=description,
        migration_checksum=migration_hash,
        is_active=True,
        validation_status="validated",
        migration_duration_seconds=execution_duration or 0.0,
        affected_tables=",".join(tables_affected) if tables_affected else None,
    )

    dbsession.add(db_version)
    dbsession.flush()  # Get the ID

    # Create migration history record
    migration_history = MigrationHistory(
        database_version_id=db_version.id,
        migration_name=migration_name,
        migration_hash=migration_hash,
        sequence_number=1,  # TODO: Calculate proper sequence
        executed_at=now,
        execution_duration_seconds=execution_duration or 0.0,
        executed_by=executed_by,
        execution_context=execution_context,
        migration_sql=migration_sql,
        tables_affected=",".join(tables_affected) if tables_affected else None,
        status="success",
        rollback_sql=rollback_sql,
        rollback_tested=False,
    )

    dbsession.add(migration_history)
    dbsession.commit()

    logger.info(f"Recorded migration: {migration_name} -> {new_version}")
    return db_version, migration_history


def attempt_auto_migration(dbsession: Session, target_version: str) -> bool:
    """
    Attempt automatic migration to target version.

    Args:
        dbsession: SQLAlchemy database session
        target_version: Target application version

    Returns:
        bool: True if migration succeeded, False otherwise
    """
    logger.warning("Automatic migration is not yet implemented")
    # TODO: Implement automatic migration logic
    # This would involve:
    # 1. Checking for available migration scripts
    # 2. Determining migration path
    # 3. Executing migrations in order
    # 4. Recording migration history
    return False


def create_compatibility_matrix(dbsession: Session) -> None:
    """
    Create initial compatibility matrix with known version combinations.

    Args:
        dbsession: SQLAlchemy database session
    """
    now = datetime.now(timezone.utc).isoformat()

    # Define known compatibility combinations
    compatibility_entries = [
        {
            "app_version_min": "1.11.0",
            "app_version_max": "1.11.99",
            "schema_version_min": "1.11.0",
            "schema_version_max": "1.11.99",
            "compatibility_level": "full",
            "compatibility_notes": "Current stable version",
            "migration_required": False,
            "migration_priority": "normal",
        },
        {
            "app_version_min": "1.10.0",
            "app_version_max": "1.10.99",
            "schema_version_min": "1.10.0",
            "schema_version_max": "1.11.99",
            "compatibility_level": "limited",
            "compatibility_notes": "Legacy version with limited feature support",
            "migration_required": True,
            "migration_priority": "high",
        },
    ]

    for entry in compatibility_entries:
        existing = (
            dbsession.query(SchemaCompatibility)
            .filter_by(
                app_version_min=entry["app_version_min"],
                app_version_max=entry["app_version_max"],
                schema_version_min=entry["schema_version_min"],
                schema_version_max=entry["schema_version_max"],
            )
            .first()
        )

        if not existing:
            compatibility = SchemaCompatibility(created_at=now, updated_at=now, **entry)
            dbsession.add(compatibility)

    dbsession.commit()
    logger.info("Created compatibility matrix")


def get_migration_history(
    dbsession: Session, limit: int | None = None, migration_type: str | None = None
) -> list[MigrationHistory]:
    """
    Get migration history records.

    Args:
        dbsession: SQLAlchemy database session
        limit: Maximum number of records to return
        migration_type: Filter by migration type

    Returns:
        List of MigrationHistory objects
    """
    query = dbsession.query(MigrationHistory).order_by(
        desc(MigrationHistory.executed_at)
    )

    if migration_type:
        query = query.join(DatabaseVersion).filter(
            DatabaseVersion.migration_type == migration_type
        )

    if limit:
        query = query.limit(limit)

    return query.all()


def validate_schema_integrity(dbsession: Session) -> tuple[bool, list[str]]:
    """
    Validate database schema integrity.

    Args:
        dbsession: SQLAlchemy database session

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        # Check if versioning tables exist
        tables_to_check = [
            "database_version",
            "migration_history",
            "schema_compatibility",
        ]

        for table_name in tables_to_check:
            try:
                result = dbsession.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                result.fetchone()
            except SQLAlchemyError as e:
                errors.append(f"Table {table_name} is missing or corrupted: {e}")

        # Check for active database version
        active_versions = (
            dbsession.query(DatabaseVersion).filter_by(is_active=True).count()
        )
        if active_versions == 0:
            errors.append("No active database version found")
        elif active_versions > 1:
            errors.append(f"Multiple active database versions found: {active_versions}")

        # Additional integrity checks can be added here

    except Exception as e:
        errors.append(f"Schema validation failed: {e}")

    return len(errors) == 0, errors


def get_database_info(dbsession: Session) -> dict:
    """
    Get comprehensive database version and compatibility information.

    Args:
        dbsession: SQLAlchemy database session

    Returns:
        Dictionary with database version information
    """
    current_app_ver = get_current_app_version()
    db_version = get_current_database_version(dbsession)

    info: dict[str, Any] = {
        "current_app_version": current_app_ver,
        "current_schema_version": CURRENT_SCHEMA_VERSION,
        "min_supported_schema_version": MIN_SUPPORTED_SCHEMA_VERSION,
        "database_version": None,
        "is_compatible": False,
        "compatibility_level": "unknown",
        "warnings": [],
        "migration_required": False,
        "last_migration": None,
        "schema_valid": False,
        "schema_errors": [],
    }

    if db_version:
        info["database_version"] = {
            "version": db_version.version,
            "schema_version": db_version.schema_version,
            "applied_at": db_version.applied_at,
            "applied_by": db_version.applied_by,
            "migration_type": db_version.migration_type,
        }

        is_compatible, level, warnings = check_version_compatibility(
            current_app_ver, db_version.schema_version, dbsession
        )

        info["is_compatible"] = is_compatible
        info["compatibility_level"] = level
        info["warnings"] = warnings
        info["migration_required"] = not is_compatible or level == "limited"

    # Get last migration
    recent_migrations = get_migration_history(dbsession, limit=1)
    if recent_migrations:
        info["last_migration"] = {
            "name": recent_migrations[0].migration_name,
            "executed_at": recent_migrations[0].executed_at,
            "status": recent_migrations[0].status,
        }

    # Validate schema
    schema_valid, schema_errors = validate_schema_integrity(dbsession)
    info["schema_valid"] = schema_valid
    info["schema_errors"] = schema_errors

    return info
