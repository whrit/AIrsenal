"""
Tests for the database versioning system.

These tests ensure that version tracking, compatibility checking,
and migration history work correctly.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.database_versioning import (
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
    validate_schema_integrity,
)
from airsenal.framework.schema import (
    Base,
    DatabaseVersion,
    MigrationHistory,
    SchemaCompatibility,
)


@pytest.fixture
def test_session():
    """Create a test database session with in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()


@pytest.fixture
def sample_database_version(test_session):
    """Create a sample database version for testing."""
    now = datetime.now(timezone.utc).isoformat()

    db_version = DatabaseVersion(
        version="1.11.0",
        schema_version="1.11.0",
        app_version="1.11.0",
        applied_at=now,
        applied_by="test",
        migration_type="initial",
        migration_source="test",
        migration_description="Test database version",
        is_active=True,
        validation_status="validated",
    )

    test_session.add(db_version)
    test_session.commit()

    return db_version


@pytest.fixture
def sample_compatibility_matrix(test_session):
    """Create sample compatibility entries for testing."""
    now = datetime.now(timezone.utc).isoformat()

    compatibilities = [
        SchemaCompatibility(
            app_version_min="1.11.0",
            app_version_max="1.11.99",
            schema_version_min="1.11.0",
            schema_version_max="1.11.99",
            compatibility_level="full",
            compatibility_notes="Current stable version",
            migration_required=False,
            created_at=now,
            updated_at=now,
        ),
        SchemaCompatibility(
            app_version_min="1.10.0",
            app_version_max="1.10.99",
            schema_version_min="1.10.0",
            schema_version_max="1.11.99",
            compatibility_level="limited",
            compatibility_notes="Legacy version with limited support",
            migration_required=True,
            created_at=now,
            updated_at=now,
        ),
    ]

    for comp in compatibilities:
        test_session.add(comp)

    test_session.commit()

    return compatibilities


class TestVersionUtilities:
    """Test version utility functions."""

    def test_get_current_app_version(self):
        """Test getting current application version."""
        version = get_current_app_version()
        assert isinstance(version, str)
        assert len(version.split(".")) >= 2  # At least major.minor

    def test_get_current_database_version_empty(self, test_session):
        """Test getting database version when none exists."""
        result = get_current_database_version(test_session)
        assert result is None

    def test_get_current_database_version_exists(
        self, test_session, sample_database_version
    ):
        """Test getting database version when one exists."""
        result = get_current_database_version(test_session)
        assert result is not None
        assert result.version == "1.11.0"
        assert result.is_active is True


class TestVersionCreation:
    """Test database version creation."""

    def test_create_initial_database_version(self, test_session):
        """Test creating initial database version."""
        db_version = create_initial_database_version(
            test_session,
            app_version="1.11.0",
            schema_version="1.11.0",
            applied_by="test_user",
            migration_description="Test initialization",
        )

        assert db_version.version == "1.11.0"
        assert db_version.schema_version == "1.11.0"
        assert db_version.app_version == "1.11.0"
        assert db_version.applied_by == "test_user"
        assert db_version.migration_type == "initial"
        assert db_version.is_active is True
        assert db_version.validation_status == "validated"

        # Verify it's in database
        retrieved = get_current_database_version(test_session)
        assert retrieved is not None
        assert retrieved.id == db_version.id


class TestCompatibilityChecking:
    """Test version compatibility checking."""

    def test_version_compatibility_with_matrix(
        self, test_session, sample_compatibility_matrix
    ):
        """Test compatibility checking using explicit matrix."""
        # Test full compatibility
        is_compatible, level, warnings = check_version_compatibility(
            "1.11.0", "1.11.0", test_session
        )
        assert is_compatible is True
        assert level == "full"
        assert len(warnings) == 0

        # Test limited compatibility
        is_compatible, level, warnings = check_version_compatibility(
            "1.10.0", "1.11.0", test_session
        )
        assert is_compatible is True
        assert level == "limited"
        assert len(warnings) >= 1

    def test_version_compatibility_fallback(self, test_session):
        """Test compatibility checking fallback logic."""
        # Test exact match
        is_compatible, level, warnings = check_version_compatibility(
            "1.11.0", "1.11.0", test_session
        )
        assert is_compatible is True
        assert level == "full"

        # Test newer schema (should be limited compatibility)
        is_compatible, level, warnings = check_version_compatibility(
            "1.10.0", "1.11.0", test_session
        )
        assert is_compatible is True
        assert level == "limited"
        assert any("newer than application" in w for w in warnings)

        # Test very old schema (should be incompatible)
        is_compatible, level, warnings = check_version_compatibility(
            "1.11.0", "1.8.0", test_session
        )
        assert is_compatible is False
        assert level == "incompatible"

    def test_version_parsing_edge_cases(self, test_session):
        """Test version parsing with edge cases."""
        # Test beta versions
        is_compatible, level, warnings = check_version_compatibility(
            "1.11.0-beta", "1.11.0", test_session
        )
        # Should handle gracefully without crashing
        assert isinstance(is_compatible, bool)
        assert isinstance(level, str)
        assert isinstance(warnings, list)


class TestStartupValidation:
    """Test startup validation functionality."""

    def test_validate_database_startup_no_version(self, test_session):
        """Test validation when no database version exists."""
        # Should create initial version
        result = validate_database_on_startup(
            test_session, force_migration=False, error_on_mismatch=False
        )
        assert result is True

        # Verify version was created
        db_version = get_current_database_version(test_session)
        assert db_version is not None
        assert db_version.migration_type == "initial"

    def test_validate_database_startup_compatible(
        self, test_session, sample_database_version
    ):
        """Test validation with compatible version."""
        result = validate_database_on_startup(
            test_session, force_migration=False, error_on_mismatch=True
        )
        assert result is True

    @patch("airsenal.framework.database_versioning.get_current_app_version")
    def test_validate_database_startup_incompatible(
        self, mock_get_version, test_session, sample_database_version
    ):
        """Test validation with incompatible version."""
        # Mock a much newer app version
        mock_get_version.return_value = "2.0.0"

        # Should fail with error_on_mismatch=True
        with pytest.raises(DatabaseVersionError):
            validate_database_on_startup(
                test_session, force_migration=False, error_on_mismatch=True
            )

        # Should return False with error_on_mismatch=False
        result = validate_database_on_startup(
            test_session, force_migration=False, error_on_mismatch=False
        )
        assert result is False


class TestMigrationRecording:
    """Test migration recording functionality."""

    def test_record_migration(self, test_session, sample_database_version):
        """Test recording a migration."""
        db_version, migration_history = record_migration(
            dbsession=test_session,
            migration_name="test_migration",
            new_version="1.12.0",
            new_schema_version="1.12.0",
            migration_type="upgrade",
            executed_by="test_user",
            description="Test migration",
            tables_affected=["player", "fixture"],
            execution_duration=5.5,
        )

        # Check database version
        assert db_version.version == "1.12.0"
        assert db_version.schema_version == "1.12.0"
        assert db_version.is_active is True
        assert db_version.migration_type == "upgrade"

        # Check migration history
        assert migration_history.migration_name == "test_migration"
        assert migration_history.executed_by == "test_user"
        assert migration_history.status == "success"
        assert migration_history.execution_duration_seconds == 5.5
        assert migration_history.tables_affected == "player,fixture"

        # Verify old version is deactivated
        old_version = (
            test_session.query(DatabaseVersion).filter_by(version="1.11.0").first()
        )
        assert old_version.is_active is False

    def test_get_migration_history(self, test_session):
        """Test retrieving migration history."""
        # Create some migration history
        now = datetime.now(timezone.utc).isoformat()

        db_version = DatabaseVersion(
            version="1.11.0",
            schema_version="1.11.0",
            app_version="1.11.0",
            applied_at=now,
            applied_by="test",
            migration_type="initial",
            migration_source="test",
            is_active=True,
        )
        test_session.add(db_version)
        test_session.flush()

        history_entries = [
            MigrationHistory(
                database_version_id=db_version.id,
                migration_name="migration_1",
                migration_hash="hash1",
                sequence_number=1,
                executed_at=now,
                execution_duration_seconds=2.0,
                executed_by="user1",
                execution_context="deployment",
                status="success",
            ),
            MigrationHistory(
                database_version_id=db_version.id,
                migration_name="migration_2",
                migration_hash="hash2",
                sequence_number=2,
                executed_at=now,
                execution_duration_seconds=3.0,
                executed_by="user2",
                execution_context="deployment",
                status="success",
            ),
        ]

        for entry in history_entries:
            test_session.add(entry)

        test_session.commit()

        # Test retrieval
        history = get_migration_history(test_session, limit=10)
        assert len(history) == 2

        # Test limit
        history_limited = get_migration_history(test_session, limit=1)
        assert len(history_limited) == 1


class TestCompatibilityMatrix:
    """Test compatibility matrix functionality."""

    def test_create_compatibility_matrix(self, test_session):
        """Test creating compatibility matrix."""
        create_compatibility_matrix(test_session)

        # Verify entries were created
        entries = test_session.query(SchemaCompatibility).all()
        assert len(entries) > 0

        # Check that we have at least current version compatibility
        current_entry = (
            test_session.query(SchemaCompatibility)
            .filter_by(app_version_min="1.11.0", compatibility_level="full")
            .first()
        )
        assert current_entry is not None


class TestDatabaseInfo:
    """Test database information gathering."""

    def test_get_database_info_no_version(self, test_session):
        """Test getting database info when no version exists."""
        info = get_database_info(test_session)

        assert "current_app_version" in info
        assert "current_schema_version" in info
        assert info["database_version"] is None
        assert info["is_compatible"] is False
        assert info["schema_valid"] is False  # Should fail without version tables

    def test_get_database_info_with_version(
        self, test_session, sample_database_version
    ):
        """Test getting database info with existing version."""
        info = get_database_info(test_session)

        assert info["database_version"] is not None
        assert info["database_version"]["version"] == "1.11.0"
        assert isinstance(info["is_compatible"], bool)
        assert isinstance(info["warnings"], list)

    def test_validate_schema_integrity(self, test_session):
        """Test schema integrity validation."""
        is_valid, errors = validate_schema_integrity(test_session)

        # Should be valid with our test database
        assert is_valid is True
        assert len(errors) == 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_database_version_error(self):
        """Test DatabaseVersionError exception."""
        with pytest.raises(DatabaseVersionError):
            msg = "Test error"
            raise DatabaseVersionError(msg)

    def test_invalid_version_format(self, test_session):
        """Test handling of invalid version formats."""
        # Should handle gracefully without crashing
        try:
            is_compatible, level, warnings = check_version_compatibility(
                "invalid.version", "1.11.0", test_session
            )
            # Should return incompatible for invalid versions
            assert is_compatible is False
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            assert "version" in str(e).lower()

    def test_database_connection_failure(self):
        """Test behavior with database connection issues."""
        # Create a broken session mock
        broken_session = MagicMock()
        broken_session.query.side_effect = Exception("Database connection failed")

        result = get_current_database_version(broken_session)
        assert result is None  # Should return None instead of raising


if __name__ == "__main__":
    pytest.main([__file__])
