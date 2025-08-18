#!/usr/bin/env python
"""
Integration test script for the historical data backfilling system.

This script tests the backfill functionality without requiring a fully migrated database.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock

from airsenal.framework.data_backfill import (
    BackfillProgress,
    BackfillResult,
    BackfillStatus,
    BackfillValidator,
    HistoricalDataBackfiller,
    ValidationSeverity,
)
from airsenal.framework.xg_data_provider import create_xg_manager


def test_backfiller_initialization():
    """Test that the backfiller can be initialized properly"""
    print("🔧 Testing backfiller initialization...")

    # Create a mock xG manager to avoid API dependencies
    mock_xg_manager = MagicMock()

    # Initialize backfiller
    backfiller = HistoricalDataBackfiller(mock_xg_manager)

    assert backfiller.xg_manager == mock_xg_manager
    assert isinstance(backfiller.validator, BackfillValidator)

    print("✅ Backfiller initialization test passed")
    return True


def test_progress_tracking():
    """Test progress tracking functionality"""
    print("📊 Testing progress tracking...")

    progress = BackfillProgress(total_items=100, completed_items=75, failed_items=10)

    assert progress.progress_percentage == 75.0
    assert progress.success_rate == 0.75 / 0.85  # 75 / (75 + 10)

    print("✅ Progress tracking test passed")
    return True


def test_result_handling():
    """Test result creation and validation issue handling"""
    print("🎯 Testing result handling...")

    result = BackfillResult(
        status=BackfillStatus.RUNNING,
        progress=BackfillProgress(total_items=50, completed_items=25),
    )

    # Add validation issues
    result.add_validation_issue(
        ValidationSeverity.WARNING,
        "test_category",
        "Test warning message",
        "test_entity",
        entity_id=123,
    )

    result.add_validation_issue(
        ValidationSeverity.CRITICAL,
        "critical_category",
        "Critical error message",
        "test_entity",
    )

    assert len(result.validation_issues) == 2
    assert result.has_critical_issues

    issues_by_severity = result.issues_by_severity
    assert len(issues_by_severity[ValidationSeverity.WARNING]) == 1
    assert len(issues_by_severity[ValidationSeverity.CRITICAL]) == 1

    print("✅ Result handling test passed")
    return True


def test_configuration():
    """Test backfiller configuration"""
    print("⚙️  Testing configuration...")

    mock_xg_manager = MagicMock()
    backfiller = HistoricalDataBackfiller(mock_xg_manager)

    # Test default configuration
    assert backfiller._batch_size == 50
    assert backfiller._max_workers == 4
    assert backfiller._retry_attempts == 3

    # Test custom configuration
    backfiller.configure(
        batch_size=100, max_workers=8, retry_attempts=5, retry_delay=2.0
    )

    assert backfiller._batch_size == 100
    assert backfiller._max_workers == 8
    assert backfiller._retry_attempts == 5
    assert backfiller._retry_delay == 2.0

    print("✅ Configuration test passed")
    return True


def test_xg_manager_integration():
    """Test that we can create an xG manager (without needing API keys)"""
    print("🔗 Testing xG manager integration...")

    try:
        # This should work even without API keys (graceful degradation)
        xg_manager = create_xg_manager(sportmonks_api_key=None)
        assert xg_manager is not None

        # Should have sportmonks provider registered (even if not functional)
        assert "sportmonks" in xg_manager.providers

        print("✅ xG manager integration test passed")
        return True
    except Exception as e:
        print(f"❌ xG manager integration test failed: {e}")
        return False


def test_single_gameweek_backfill_simulation():
    """Test single gameweek backfill with mocked data"""
    print("🎮 Testing single gameweek backfill simulation...")

    # Create mock xG manager
    mock_xg_manager = MagicMock()
    mock_xg_manager.sync_xg_data_for_gameweek.return_value = 25  # 25 records updated

    backfiller = HistoricalDataBackfiller(mock_xg_manager)

    # Test successful backfill
    result = backfiller._backfill_single_gameweek("2324", 1)

    assert result.status == BackfillStatus.COMPLETED
    assert result.progress.completed_items == 1

    # Verify the mock was called correctly
    mock_xg_manager.sync_xg_data_for_gameweek.assert_called_once_with("2324", 1)

    print("✅ Single gameweek backfill simulation passed")
    return True


def test_validation_rules():
    """Test that validation rules are properly configured"""
    print("🔍 Testing validation rules...")

    validator = BackfillValidator()

    expected_rules = [
        "xg_data_consistency",
        "missing_data_detection",
        "statistical_outliers",
        "cross_reference_validation",
        "temporal_consistency",
    ]

    for rule in expected_rules:
        assert rule in validator.validation_rules
        assert callable(validator.validation_rules[rule])

    print("✅ Validation rules test passed")
    return True


def main():
    """Run all integration tests"""
    print("🚀 Starting AIrsenal Historical Data Backfilling Integration Tests")
    print("=" * 70)

    tests = [
        test_backfiller_initialization,
        test_progress_tracking,
        test_result_handling,
        test_configuration,
        test_xg_manager_integration,
        test_single_gameweek_backfill_simulation,
        test_validation_rules,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1
        print()

    print("=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All integration tests passed! The backfill system is ready to use.")
        print("\n📋 Next steps:")
        print("  1. Ensure you have a valid Sportmonks API key configured")
        print("  2. Run database migrations if needed for new schema columns")
        print(
            "  3. Start with a small test backfill: python airsenal/scripts/backfill_xg_data.py --season 2324 --gameweeks 1 --dry-run"
        )
        return 0
    print(
        "⚠️  Some tests failed. Please check the issues above before using the backfill system."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
