# AIrsenal Database Versioning System

This document describes the comprehensive database versioning system implemented for AIrsenal, which provides schema change tracking, compatibility management, and migration history.

## Overview

The database versioning system ensures compatibility between application versions and database schema versions, tracks all migrations, and provides tools for safe deployment and rollback operations.

## Features

- **Schema Version Tracking**: Track database schema versions with semantic versioning
- **Compatibility Matrix**: Define and check compatibility between app and database versions
- **Migration History**: Complete audit trail of all database changes
- **Startup Validation**: Automatic version checking on application startup
- **CI/CD Integration**: Tools for automated deployment validation
- **Rollback Support**: Track rollback procedures for safe migrations

## Core Components

### 1. Database Models

#### DatabaseVersion
Tracks the current database schema version and metadata:
- Version numbers (app version, schema version)
- Migration metadata (who, when, how long)
- Compatibility information
- Validation status

#### MigrationHistory
Detailed history of all database migrations:
- Migration identification and execution details
- Performance metrics (duration, memory usage)
- Success/failure tracking
- Rollback information

#### SchemaCompatibility
Matrix defining compatibility between versions:
- Version ranges for app and schema
- Compatibility levels (full, limited, deprecated, incompatible)
- Known issues and workarounds

### 2. Core Utilities (`database_versioning.py`)

Key functions include:
- `validate_database_on_startup()`: Check compatibility on app startup
- `check_version_compatibility()`: Determine if versions are compatible
- `record_migration()`: Log completed migrations
- `get_database_info()`: Comprehensive version status

### 3. Management CLI (`airsenal_db_version`)

Command-line tool for version management:
```bash
# Check current status
airsenal_db_version status

# Show migration history
airsenal_db_version history --limit 10

# Validate compatibility
airsenal_db_version validate

# Initialize version tracking
airsenal_db_version init

# Record manual migration
airsenal_db_version record "migration_name" "1.12.0"
```

### 4. CI/CD Integration (`cicd_integration.py`)

Tools for automated deployment:
- Pre-deployment validation
- Post-deployment verification
- Migration prerequisites checking
- GitHub Actions workflow generation

## Installation and Setup

### 1. Install Dependencies

The versioning system uses standard dependencies already included in AIrsenal:
- SQLAlchemy for database operations
- Click for CLI interface
- Standard library modules for version parsing

### 2. Initialize Version Tracking

For existing databases:
```bash
airsenal_db_version init
```

For new databases, version tracking is automatically initialized during `airsenal_setup_initial_db`.

### 3. Verify Installation

Check that everything is working:
```bash
airsenal_db_version status
```

## Usage Guide

### Checking Version Status

Get comprehensive version information:
```bash
airsenal_db_version status
```

Example output:
```
=== AIrsenal Database Version Status ===
Application Version: 1.11.0
Current Schema Version: 1.11.0
Minimum Supported Schema: 1.10.0

Database Version:
  Version: 1.11.0
  Schema Version: 1.11.0
  Applied At: 2024-01-15T10:30:00Z
  Applied By: database_initialization
  Migration Type: initial

✓ Compatibility: FULL

Schema Valid: ✓
```

### Managing Migrations

View migration history:
```bash
airsenal_db_version history --limit 5
```

Record a manual migration:
```bash
airsenal_db_version record "add_player_attributes" "1.12.0" \
  --description "Added xG and form metrics to player attributes"
```

### Application Integration

The versioning system automatically validates compatibility when AIrsenal applications start:

```python
from airsenal.framework.database_versioning import validate_database_on_startup
from airsenal.framework.schema import session_scope

with session_scope() as dbsession:
    try:
        compatible = validate_database_on_startup(
            dbsession=dbsession,
            force_migration=False,
            error_on_mismatch=True
        )
        if compatible:
            print("Database is compatible - proceeding with application startup")
        else:
            print("Database compatibility issues detected")
    except DatabaseVersionError as e:
        print(f"Version validation failed: {e}")
```

### CI/CD Integration

#### GitHub Actions

Generate a workflow file:
```python
from airsenal.framework.cicd_integration import create_github_actions_workflow
workflow_content = create_github_actions_workflow()
# Save to .github/workflows/db-version-check.yml
```

#### Manual CI Validation

In your CI pipeline:
```bash
python -c "
from airsenal.framework.cicd_integration import run_ci_validation
import sys
success = run_ci_validation()
sys.exit(0 if success else 1)
"
```

#### Deployment Validation

For deployment pipelines:
```bash
python -c "
from airsenal.framework.cicd_integration import run_cd_deployment
success = run_cd_deployment(perform_migration=True)
sys.exit(0 if success else 1)
"
```

## Version Compatibility Rules

### Semantic Versioning

The system uses semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Compatibility Levels

1. **Full**: Complete compatibility, no issues expected
2. **Limited**: Compatible but with warnings or minor limitations
3. **Deprecated**: Compatible but deprecated, upgrade recommended
4. **Incompatible**: Not compatible, migration required

### Default Compatibility Rules

- Same major.minor versions: **Full** compatibility
- Schema 1-2 minor versions newer than app: **Limited** compatibility
- Schema 3+ minor versions different: **Incompatible**
- Different major versions: **Incompatible**

## Migration Best Practices

### 1. Always Test Migrations

- Test on a copy of production data
- Verify rollback procedures work
- Document expected downtime

### 2. Record All Changes

```python
from airsenal.framework.database_versioning import record_migration

record_migration(
    dbsession=session,
    migration_name="add_fixture_difficulty_ratings",
    new_version="1.12.0",
    new_schema_version="1.12.0",
    description="Added fixture difficulty rating system",
    tables_affected=["fixture", "player_attributes"],
    rollback_sql="DROP COLUMN difficulty_rating FROM fixture;"
)
```

### 3. Use Compatibility Matrix

Define explicit compatibility for special cases:
```python
from airsenal.framework.schema import SchemaCompatibility

compatibility = SchemaCompatibility(
    app_version_min="1.11.0",
    app_version_max="1.11.99",
    schema_version_min="1.10.0",
    schema_version_max="1.12.0",
    compatibility_level="limited",
    compatibility_notes="Requires manual data backfill for new features"
)
```

## Troubleshooting

### Common Issues

1. **"No database version found"**
   - Run `airsenal_db_version init` to initialize tracking
   - Check database connectivity

2. **"Version incompatible"**
   - Check compatibility with `airsenal_db_version status`
   - Run necessary migrations
   - Update application version

3. **"Schema validation failed"**
   - Verify database schema integrity
   - Check for missing tables or columns
   - Restore from backup if necessary

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger('airsenal.framework.database_versioning').setLevel(logging.DEBUG)
```

### Manual Recovery

If version tracking is corrupted:
```sql
-- Check current versions
SELECT * FROM database_version WHERE is_active = 1;

-- Manually update if needed (use with caution)
UPDATE database_version SET is_active = 0;
INSERT INTO database_version (...) VALUES (...);
```

## Testing

Run the test suite:
```bash
pytest airsenal/tests/test_database_versioning.py -v
```

The tests cover:
- Version creation and retrieval
- Compatibility checking
- Migration recording
- Error handling
- CI/CD integration

## Files Created/Modified

### New Files
- `airsenal/framework/database_versioning.py`: Core versioning utilities
- `airsenal/framework/cicd_integration.py`: CI/CD integration tools
- `airsenal/scripts/db_version_manager.py`: CLI management tool
- `airsenal/tests/test_database_versioning.py`: Comprehensive test suite
- `DATABASE_VERSIONING.md`: This documentation

### Modified Files
- `airsenal/framework/schema.py`: Added DatabaseVersion, MigrationHistory, SchemaCompatibility models
- `airsenal/scripts/airsenal_run_pipeline.py`: Added startup version validation
- `airsenal/scripts/fill_db_init.py`: Added version tracking to database initialization
- `airsenal/scripts/update_db.py`: Added version validation to database updates
- `pyproject.toml`: Added `airsenal_db_version` command entry point

## Future Enhancements

Potential improvements for future versions:

1. **Automatic Migration Scripts**: Integration with Alembic for automated migrations
2. **Version Branching**: Support for feature branch schema versions
3. **Performance Monitoring**: Track performance impact of schema changes
4. **Backup Integration**: Automatic backup creation before migrations
5. **Multi-Environment Support**: Different compatibility rules for dev/staging/prod

## Support

For questions or issues with the database versioning system:

1. Check this documentation first
2. Run diagnostics: `airsenal_db_version status --json-output`
3. Review logs with debug logging enabled
4. Check the test suite for usage examples
5. Submit issues with full diagnostic output

The database versioning system provides a robust foundation for managing schema changes safely and ensuring compatibility across different environments and deployment scenarios.