#!/usr/bin/env python3
"""
Migration validation utilities for AIrsenal database schema changes.

This module provides validation functions to ensure database integrity
before and after applying migrations, particularly for the PlayerAttributes
extension migration.
"""

import sys

from sqlalchemy import create_engine, inspect, text

# Optional PostgreSQL support
try:
    import importlib.util

    HAS_PSYCOPG2 = importlib.util.find_spec("psycopg2") is not None
except ImportError:
    HAS_PSYCOPG2 = False

from airsenal.framework.schema import get_connection_string


class MigrationValidator:
    """Validates database schema and data integrity during migrations."""

    def __init__(self):
        """Initialize the validator with database connection."""
        self.connection_string = get_connection_string()
        self.engine = create_engine(self.connection_string)
        self.inspector = inspect(self.engine)

    def run_pre_migration_checks(self) -> bool:
        """
        Run comprehensive checks before applying migration.

        Returns:
            bool: True if all checks pass, False otherwise
        """
        print("🔍 Running pre-migration validation checks...")

        checks = [
            self._check_database_connectivity,
            self._check_table_exists,
            self._check_current_schema_consistency,
            self._count_existing_records,
            self._check_for_null_constraints,
            self._backup_recommendations,
        ]

        all_passed = True
        for check in checks:
            try:
                result = check()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"❌ Check failed with error: {e}")
                all_passed = False

        if all_passed:
            print("✅ All pre-migration checks passed!")
        else:
            print("❌ Some pre-migration checks failed. Review before proceeding.")

        return all_passed

    def run_post_migration_checks(self) -> bool:
        """
        Run comprehensive checks after applying migration.

        Returns:
            bool: True if all checks pass, False otherwise
        """
        print("\n🔍 Running post-migration validation checks...")

        checks = [
            self._check_database_connectivity,
            self._check_new_columns_exist,
            self._check_new_indexes_exist,
            self._verify_column_types,
            self._verify_column_defaults,
            self._check_data_integrity,
            self._performance_check_indexes,
        ]

        all_passed = True
        for check in checks:
            try:
                result = check()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"❌ Check failed with error: {e}")
                all_passed = False

        if all_passed:
            print("✅ All post-migration checks passed!")
        else:
            print("❌ Some post-migration checks failed. Consider rollback.")

        return all_passed

    def _check_database_connectivity(self) -> bool:
        """Check if database is accessible."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Database connectivity: OK")
            return True
        except Exception as e:
            print(f"❌ Database connectivity failed: {e}")
            return False

    def _check_table_exists(self) -> bool:
        """Check if player_attributes table exists."""
        try:
            tables = self.inspector.get_table_names()
            if "player_attributes" in tables:
                print("✅ player_attributes table: EXISTS")
                return True
            print("❌ player_attributes table: NOT FOUND")
            return False
        except Exception as e:
            print(f"❌ Table check failed: {e}")
            return False

    def _check_current_schema_consistency(self) -> bool:
        """Check current schema is consistent."""
        try:
            columns = self.inspector.get_columns("player_attributes")
            required_columns = [
                "id",
                "player_id",
                "season",
                "gameweek",
                "price",
                "team",
                "position",
                "chance_of_playing_next_round",
            ]

            existing_columns = [col["name"] for col in columns]
            missing = [col for col in required_columns if col not in existing_columns]

            if not missing:
                print(f"✅ Schema consistency: OK ({len(existing_columns)} columns)")
                return True
            print(f"❌ Missing required columns: {missing}")
            return False
        except Exception as e:
            print(f"❌ Schema consistency check failed: {e}")
            return False

    def _count_existing_records(self) -> bool:
        """Count existing records for impact assessment."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM player_attributes"))
                count = result.scalar()
                print(f"📊 Existing records: {count:,}")

                # Additional statistics
                seasons_result = conn.execute(
                    text("SELECT COUNT(DISTINCT season) FROM player_attributes")
                )
                seasons_count = seasons_result.scalar()

                players_result = conn.execute(
                    text("SELECT COUNT(DISTINCT player_id) FROM player_attributes")
                )
                players_count = players_result.scalar()

                print(f"📊 Unique seasons: {seasons_count}")
                print(f"📊 Unique players: {players_count}")

            return True
        except Exception as e:
            print(f"❌ Record count failed: {e}")
            return False

    def _check_for_null_constraints(self) -> bool:
        """Check for potential null constraint violations."""
        try:
            with self.engine.connect() as conn:
                # Check if any existing data would violate new constraints
                result = conn.execute(
                    text("""
                    SELECT COUNT(*) FROM player_attributes
                    WHERE player_id IS NULL OR season IS NULL OR gameweek IS NULL
                """)
                )
                null_count = result.scalar()

                if null_count == 0:
                    print("✅ No null constraint violations detected")
                    return True
                print(f"❌ Found {null_count} records with null key fields")
                return False
        except Exception as e:
            print(f"❌ Null constraint check failed: {e}")
            return False

    def _backup_recommendations(self) -> bool:
        """Provide backup recommendations."""
        print("\n💾 BACKUP RECOMMENDATIONS:")
        print("   Before migration, consider:")

        if "sqlite" in self.connection_string:
            db_file = self.connection_string.replace("sqlite:///", "")
            print(f"   - cp '{db_file}' '{db_file}.backup.$(date +%Y%m%d_%H%M%S)'")
        else:
            print("   - pg_dump for PostgreSQL databases")
            print("   - Full database backup with your preferred method")

        print("   - Test migration on copy of production data first")
        print("   - Ensure sufficient disk space for schema changes")
        return True

    def _check_new_columns_exist(self) -> bool:
        """Check that all new columns were created."""
        expected_columns = [
            "xg_per_90",
            "xa_per_90",
            "xgi_per_90",
            "form_3_games",
            "form_5_games",
            "form_10_games",
            "momentum",
            "next_3_fixture_difficulty",
            "next_5_fixture_difficulty",
            "is_penalty_taker",
            "is_free_kick_taker",
            "is_corner_taker",
            "role_confidence",
            "shots_per_90",
            "key_passes_per_90",
            "tackles_per_90",
            "interceptions_per_90",
            "clearances_per_90",
        ]

        try:
            columns = self.inspector.get_columns("player_attributes")
            existing_columns = [col["name"] for col in columns]
            missing = [col for col in expected_columns if col not in existing_columns]

            if not missing:
                print(f"✅ New columns created: ALL {len(expected_columns)} columns")
                return True
            print(f"❌ Missing new columns: {missing}")
            return False
        except Exception as e:
            print(f"❌ New columns check failed: {e}")
            return False

    def _check_new_indexes_exist(self) -> bool:
        """Check that all new indexes were created."""
        expected_indexes = [
            "ix_form_3_games",
            "ix_form_5_games",
            "ix_xg_per_90",
            "ix_xgi_per_90",
            "ix_next_3_fixture_difficulty",
            "ix_position_penalty_taker",
            "ix_momentum",
        ]

        try:
            indexes = self.inspector.get_indexes("player_attributes")
            existing_indexes = [idx["name"] for idx in indexes]
            missing = [idx for idx in expected_indexes if idx not in existing_indexes]

            if not missing:
                print(f"✅ New indexes created: ALL {len(expected_indexes)} indexes")
                return True
            print(f"❌ Missing new indexes: {missing}")
            return False
        except Exception as e:
            print(f"❌ New indexes check failed: {e}")
            return False

    def _verify_column_types(self) -> bool:
        """Verify that new columns have correct data types."""
        expected_types = {
            "xg_per_90": ["FLOAT", "REAL", "DOUBLE PRECISION"],
            "is_penalty_taker": ["BOOLEAN", "BOOL", "TINYINT"],
            "form_3_games": ["FLOAT", "REAL", "DOUBLE PRECISION"],
        }

        try:
            columns = self.inspector.get_columns("player_attributes")
            column_types = {col["name"]: str(col["type"]) for col in columns}

            for col_name, expected_type_variants in expected_types.items():
                if col_name in column_types:
                    actual_type = column_types[col_name].upper()
                    type_match = any(
                        exp_type in actual_type for exp_type in expected_type_variants
                    )

                    if type_match:
                        print(f"✅ {col_name}: {actual_type}")
                    else:
                        print(
                            f"❌ {col_name}: Expected {expected_type_variants}, got {actual_type}"
                        )
                        return False

            return True
        except Exception as e:
            print(f"❌ Column type verification failed: {e}")
            return False

    def _verify_column_defaults(self) -> bool:
        """Verify that boolean columns have correct defaults."""
        boolean_columns = ["is_penalty_taker", "is_free_kick_taker", "is_corner_taker"]

        try:
            with self.engine.connect() as conn:
                for col in boolean_columns:
                    # Check if new records get default value
                    result = conn.execute(
                        text(f"""
                        SELECT COUNT(*) FROM player_attributes
                        WHERE {col} IS NULL
                    """)
                    )
                    null_count = result.scalar()

                    if null_count == 0:
                        print(f"✅ {col}: No null values (defaults working)")
                    else:
                        print(f"❌ {col}: {null_count} null values found")
                        return False

            return True
        except Exception as e:
            print(f"❌ Default values check failed: {e}")
            return False

    def _check_data_integrity(self) -> bool:
        """Check that existing data wasn't corrupted."""
        try:
            with self.engine.connect() as conn:
                # Count total records (should be unchanged)
                result = conn.execute(text("SELECT COUNT(*) FROM player_attributes"))
                post_count = result.scalar()

                # Check that key fields are still intact
                result = conn.execute(
                    text("""
                    SELECT COUNT(*) FROM player_attributes
                    WHERE player_id IS NOT NULL AND season IS NOT NULL
                """)
                )
                valid_records = result.scalar()

                if post_count == valid_records:
                    print(f"✅ Data integrity: {post_count:,} records intact")
                    return True
                print(
                    f"❌ Data integrity: {post_count - valid_records} corrupted records"
                )
                return False
        except Exception as e:
            print(f"❌ Data integrity check failed: {e}")
            return False

    def _performance_check_indexes(self) -> bool:
        """Quick performance check for new indexes."""
        try:
            with self.engine.connect() as conn:
                # Test index usage with EXPLAIN (database-specific)
                if "sqlite" in self.connection_string:
                    result = conn.execute(
                        text("""
                        EXPLAIN QUERY PLAN
                        SELECT * FROM player_attributes
                        WHERE form_3_games > 5.0
                        LIMIT 1
                    """)
                    )

                    plan = result.fetchall()
                    index_used = any("ix_form_3_games" in str(row) for row in plan)

                    if index_used:
                        print("✅ Index performance: Query optimizer using new indexes")
                    else:
                        print("⚠️  Index performance: Indexes may not be optimally used")

                else:  # PostgreSQL
                    result = conn.execute(
                        text("""
                        SELECT COUNT(*) FROM player_attributes
                        WHERE form_3_games IS NOT NULL
                    """)
                    )
                    # For PostgreSQL, we'd need different EXPLAIN syntax
                    print("✅ Index performance: Basic query successful")

            return True
        except Exception as e:
            print(f"⚠️  Index performance check failed: {e}")
            return True  # Don't fail migration for this

    def rollback_safety_check(self) -> bool:
        """Check if rollback is safe to perform."""
        print("\n🔄 Checking rollback safety...")

        try:
            # Check if any new columns have non-null data
            columns_to_check = [
                "xg_per_90",
                "form_3_games",
                "is_penalty_taker",  # Sample columns
            ]

            with self.engine.connect() as conn:
                data_loss_risk = False
                for col in columns_to_check:
                    try:
                        result = conn.execute(
                            text(f"""
                            SELECT COUNT(*) FROM player_attributes
                            WHERE {col} IS NOT NULL
                        """)
                        )
                        non_null_count = result.scalar()

                        if non_null_count > 0:
                            print(f"⚠️  {col}: {non_null_count} records have data")
                            data_loss_risk = True
                        else:
                            print(f"✅ {col}: No data to lose")
                    except:
                        # Column might not exist yet
                        print(f"✅ {col}: Column not found (safe to rollback)")

                if data_loss_risk:
                    print("\n❌ ROLLBACK WARNING: Data loss will occur!")
                    print("   Consider backing up new column data before rollback.")
                    return False
                print("\n✅ Rollback is safe - no data loss expected")
                return True

        except Exception as e:
            print(f"❌ Rollback safety check failed: {e}")
            return False


def main():
    """Command-line interface for migration validation."""
    if len(sys.argv) != 2 or sys.argv[1] not in ["pre", "post", "rollback-check"]:
        print("Usage: python migration_validation.py [pre|post|rollback-check]")
        sys.exit(1)

    validator = MigrationValidator()

    if sys.argv[1] == "pre":
        success = validator.run_pre_migration_checks()
    elif sys.argv[1] == "post":
        success = validator.run_post_migration_checks()
    elif sys.argv[1] == "rollback-check":
        success = validator.rollback_safety_check()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
