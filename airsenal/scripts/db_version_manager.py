#!/usr/bin/env python3
"""
Database Version Manager for AIrsenal

This script provides utilities for managing database schema versions,
checking compatibility, and performing migrations.
"""

import json
import sys

import click

from airsenal.framework.database_versioning import (
    CURRENT_SCHEMA_VERSION,
    MIN_SUPPORTED_SCHEMA_VERSION,
    DatabaseVersionError,
    check_version_compatibility,
    create_compatibility_matrix,
    create_initial_database_version,
    get_current_app_version,
    get_current_database_version,
    get_database_info,
    get_migration_history,
    record_migration,
    validate_database_on_startup,
)
from airsenal.framework.schema import session_scope


@click.group()
def cli():
    """AIrsenal Database Version Manager"""


@cli.command()
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def status(json_output: bool):
    """Show current database version status and compatibility"""
    with session_scope() as dbsession:
        try:
            info = get_database_info(dbsession)

            if json_output:
                click.echo(json.dumps(info, indent=2))
                return

            click.echo("=== AIrsenal Database Version Status ===")
            click.echo(f"Application Version: {info['current_app_version']}")
            click.echo(f"Current Schema Version: {info['current_schema_version']}")
            click.echo(
                f"Minimum Supported Schema: {info['min_supported_schema_version']}"
            )
            click.echo()

            if info["database_version"]:
                db_ver = info["database_version"]
                click.echo("Database Version:")
                click.echo(f"  Version: {db_ver['version']}")
                click.echo(f"  Schema Version: {db_ver['schema_version']}")
                click.echo(f"  Applied At: {db_ver['applied_at']}")
                click.echo(f"  Applied By: {db_ver['applied_by']}")
                click.echo(f"  Migration Type: {db_ver['migration_type']}")
                click.echo()

                # Compatibility status
                if info["is_compatible"]:
                    status_color = (
                        "green" if info["compatibility_level"] == "full" else "yellow"
                    )
                    click.echo(
                        click.style(
                            f"✓ Compatibility: {info['compatibility_level'].upper()}",
                            fg=status_color,
                            bold=True,
                        )
                    )
                else:
                    click.echo(
                        click.style(
                            f"✗ Compatibility: {info['compatibility_level'].upper()}",
                            fg="red",
                            bold=True,
                        )
                    )

                if info["warnings"]:
                    click.echo("\nWarnings:")
                    for warning in info["warnings"]:
                        click.echo(f"  - {warning}")

                if info["migration_required"]:
                    click.echo(
                        click.style(
                            "\n⚠ Migration required or recommended",
                            fg="yellow",
                            bold=True,
                        )
                    )
            else:
                click.echo(
                    click.style(
                        "No database version found - database may not be initialized",
                        fg="red",
                    )
                )

            # Schema validation
            click.echo(f"\nSchema Valid: {'✓' if info['schema_valid'] else '✗'}")
            if info["schema_errors"]:
                click.echo("Schema Errors:")
                for error in info["schema_errors"]:
                    click.echo(f"  - {error}")

            # Last migration
            if info["last_migration"]:
                last_mig = info["last_migration"]
                click.echo("\nLast Migration:")
                click.echo(f"  Name: {last_mig['name']}")
                click.echo(f"  Executed: {last_mig['executed_at']}")
                click.echo(f"  Status: {last_mig['status']}")

        except Exception as e:
            click.echo(f"Error getting database status: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.option("--limit", default=10, help="Number of recent migrations to show")
@click.option("--migration-type", help="Filter by migration type")
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def history(limit: int, migration_type: str | None, json_output: bool):
    """Show migration history"""
    with session_scope() as dbsession:
        try:
            migrations = get_migration_history(
                dbsession=dbsession, limit=limit, migration_type=migration_type
            )

            if json_output:
                migration_data = []
                for mig in migrations:
                    migration_data.append(
                        {
                            "name": mig.migration_name,
                            "executed_at": mig.executed_at,
                            "executed_by": mig.executed_by,
                            "status": mig.status,
                            "duration_seconds": mig.execution_duration_seconds,
                            "context": mig.execution_context,
                            "tables_affected": mig.tables_affected,
                        }
                    )
                click.echo(json.dumps(migration_data, indent=2))
                return

            click.echo("=== Migration History ===")
            if not migrations:
                click.echo("No migrations found.")
                return

            for mig in migrations:
                status_symbol = "✓" if mig.status == "success" else "✗"
                click.echo(f"{status_symbol} {mig.migration_name}")
                click.echo(f"    Executed: {mig.executed_at} by {mig.executed_by}")
                click.echo(f"    Status: {mig.status}")
                click.echo(f"    Duration: {mig.execution_duration_seconds:.2f}s")
                if mig.tables_affected:
                    click.echo(f"    Tables: {mig.tables_affected}")
                click.echo()

        except Exception as e:
            click.echo(f"Error getting migration history: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.option("--force", is_flag=True, help="Force validation even if database exists")
def validate(force: bool):
    """Validate database version compatibility"""
    with session_scope() as dbsession:
        try:
            click.echo("Validating database version compatibility...")

            compatible = validate_database_on_startup(
                dbsession=dbsession, force_migration=False, error_on_mismatch=False
            )

            if compatible:
                click.echo(
                    click.style("✓ Database is compatible", fg="green", bold=True)
                )
            else:
                click.echo(
                    click.style(
                        "✗ Database compatibility issues found", fg="red", bold=True
                    )
                )
                click.echo("Run 'airsenal_db_version status' for more details")
                sys.exit(1)

        except DatabaseVersionError as e:
            click.echo(f"Validation failed: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error during validation: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.option(
    "--force", is_flag=True, help="Force initialization even if version exists"
)
def init(force: bool):
    """Initialize database version tracking"""
    with session_scope() as dbsession:
        try:
            existing_version = get_current_database_version(dbsession)

            if existing_version and not force:
                click.echo("Database version tracking already exists.")
                click.echo("Use --force to reinitialize.")
                return

            click.echo("Initializing database version tracking...")

            db_version = create_initial_database_version(
                dbsession=dbsession,
                applied_by="manual_initialization",
                migration_description="Manual initialization of database version tracking",
            )

            click.echo("Creating compatibility matrix...")
            create_compatibility_matrix(dbsession)

            click.echo(
                click.style(
                    f"✓ Database version tracking initialized: {db_version.version}",
                    fg="green",
                    bold=True,
                )
            )

        except Exception as e:
            click.echo(f"Error initializing database version: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.argument("migration_name")
@click.argument("new_version")
@click.option("--schema-version", help="New schema version (defaults to new version)")
@click.option("--description", help="Migration description")
@click.option("--migration-type", default="manual", help="Migration type")
@click.option("--executed-by", default="manual", help="Who executed the migration")
def record(
    migration_name: str,
    new_version: str,
    schema_version: str | None,
    description: str | None,
    migration_type: str,
    executed_by: str,
):
    """Record a manual migration"""
    with session_scope() as dbsession:
        try:
            if schema_version is None:
                schema_version = new_version

            click.echo(f"Recording migration: {migration_name} -> {new_version}")

            db_version, migration_history = record_migration(
                dbsession=dbsession,
                migration_name=migration_name,
                new_version=new_version,
                new_schema_version=schema_version,
                migration_type=migration_type,
                executed_by=executed_by,
                description=description,
            )

            click.echo(
                click.style("✓ Migration recorded successfully", fg="green", bold=True)
            )
            click.echo(f"New version: {db_version.version}")
            click.echo(f"Schema version: {db_version.schema_version}")

        except Exception as e:
            click.echo(f"Error recording migration: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.argument("app_version")
@click.argument("schema_version")
def check_compatibility(app_version: str, schema_version: str):
    """Check compatibility between specific versions"""
    with session_scope() as dbsession:
        try:
            is_compatible, level, warnings = check_version_compatibility(
                app_version, schema_version, dbsession
            )

            click.echo(f"App Version: {app_version}")
            click.echo(f"Schema Version: {schema_version}")
            click.echo()

            if is_compatible:
                status_color = "green" if level == "full" else "yellow"
                click.echo(
                    click.style(f"✓ Compatible ({level})", fg=status_color, bold=True)
                )
            else:
                click.echo(
                    click.style(f"✗ Incompatible ({level})", fg="red", bold=True)
                )

            if warnings:
                click.echo("\nWarnings:")
                for warning in warnings:
                    click.echo(f"  - {warning}")

        except Exception as e:
            click.echo(f"Error checking compatibility: {e}", err=True)
            sys.exit(1)


@cli.command()
def info():
    """Show version system information"""
    click.echo("=== AIrsenal Database Version System ===")
    click.echo(f"Current Application Version: {get_current_app_version()}")
    click.echo(f"Current Schema Version: {CURRENT_SCHEMA_VERSION}")
    click.echo(f"Minimum Supported Schema: {MIN_SUPPORTED_SCHEMA_VERSION}")
    click.echo()
    click.echo("Available Commands:")
    click.echo("  status    - Show current version status")
    click.echo("  history   - Show migration history")
    click.echo("  validate  - Validate compatibility")
    click.echo("  init      - Initialize version tracking")
    click.echo("  record    - Record a manual migration")
    click.echo("  check-compatibility - Check specific version compatibility")


def main():
    """Command-line interface for database version management"""
    cli()


if __name__ == "__main__":
    main()
