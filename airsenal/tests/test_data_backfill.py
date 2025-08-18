"""
Tests for Historical Data Backfilling System

This module provides comprehensive tests for the data backfilling functionality,
including unit tests, integration tests, and performance tests.
"""

import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pytest

from airsenal.framework.data_backfill import (
    BackfillProgress,
    BackfillProgressTracker,
    BackfillResult,
    BackfillStatus,
    BackfillValidator,
    HistoricalDataBackfiller,
    ValidationIssue,
    ValidationSeverity,
)
from airsenal.framework.xg_data_provider import XGDataPoint


class TestBackfillProgress:
    """Test the BackfillProgress dataclass"""

    def test_progress_calculation(self):
        """Test progress percentage calculation"""
        progress = BackfillProgress(total_items=100, completed_items=25)
        assert progress.progress_percentage == 25.0

        progress = BackfillProgress(total_items=0, completed_items=0)
        assert progress.progress_percentage == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        progress = BackfillProgress(completed_items=80, failed_items=20)
        assert progress.success_rate == 0.8

        progress = BackfillProgress(completed_items=0, failed_items=0)
        assert progress.success_rate == 1.0

    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation"""
        start_time = datetime.now() - timedelta(seconds=60)
        progress = BackfillProgress(start_time=start_time)

        elapsed = progress.elapsed_time
        assert 59 <= elapsed.total_seconds() <= 61  # Allow for small timing variations

    def test_completion_time_estimation(self):
        """Test completion time estimation"""
        start_time = datetime.now() - timedelta(seconds=60)
        progress = BackfillProgress(
            total_items=100, completed_items=50, start_time=start_time
        )

        estimated = progress.estimate_completion_time()
        assert estimated is not None
        assert estimated > datetime.now()


class TestBackfillProgressTracker:
    """Test the BackfillProgressTracker class"""

    def test_progress_tracker_initialization(self):
        """Test tracker initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("airsenal.framework.data_backfill.AIRSENAL_HOME", temp_dir):
                tracker = BackfillProgressTracker("test_session", 100)

                assert tracker.session_id == "test_session"
                assert tracker.progress.total_items == 100
                assert tracker.progress.completed_items == 0

    def test_progress_updates(self):
        """Test progress update functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("airsenal.framework.data_backfill.AIRSENAL_HOME", temp_dir):
                tracker = BackfillProgressTracker("test_session", 100)
                tracker.start()

                initial_time = tracker.progress.start_time
                assert initial_time is not None

                tracker.update(completed_delta=10, current_operation="Test operation")

                assert tracker.progress.completed_items == 10
                assert tracker.progress.current_operation == "Test operation"
                assert tracker.progress.last_update_time is not None

    def test_progress_persistence(self):
        """Test progress persistence to disk"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("airsenal.framework.data_backfill.AIRSENAL_HOME", temp_dir):
                # Create and update tracker
                tracker1 = BackfillProgressTracker("test_session", 100)
                tracker1.start()
                tracker1.update(completed_delta=25)

                # Create new tracker with same session ID
                tracker2 = BackfillProgressTracker("test_session", 100)

                # Should load previous progress
                assert tracker2.progress.completed_items == 25

    def test_callback_notifications(self):
        """Test progress callback notifications"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("airsenal.framework.data_backfill.AIRSENAL_HOME", temp_dir):
                tracker = BackfillProgressTracker("test_session", 100)

                callback_called = []

                def test_callback(progress):
                    callback_called.append(progress.completed_items)

                tracker.add_callback(test_callback)
                tracker.start()
                tracker.update(completed_delta=5)
                tracker.update(completed_delta=10)

                assert len(callback_called) == 2
                assert callback_called == [5, 15]


class TestBackfillValidator:
    """Test the BackfillValidator class"""

    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = BackfillValidator()

        assert len(validator.validation_rules) > 0
        assert "xg_data_consistency" in validator.validation_rules
        assert "missing_data_detection" in validator.validation_rules

    @patch("airsenal.framework.data_backfill.session_scope")
    def test_xg_data_consistency_validation(self, mock_session_scope):
        """Test xG data consistency validation"""
        # Mock database session and data
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Create mock PlayerScore with inconsistent xG data
        mock_score = MagicMock()
        mock_score.id = 1
        mock_score.expected_goals = 1.5
        mock_score.expected_assists = 0.5
        mock_score.expected_goal_involvements = 2.5  # Should be 2.0

        mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = [
            mock_score
        ]

        validator = BackfillValidator()
        issues = validator._validate_xg_data_consistency("2324", None)

        # Should find the inconsistency
        assert len(issues) > 0
        consistency_issues = [i for i in issues if i.category == "xg_consistency"]
        assert len(consistency_issues) > 0

    @patch("airsenal.framework.data_backfill.session_scope")
    def test_missing_data_detection(self, mock_session_scope):
        """Test missing data detection"""
        # Mock database session
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock low coverage scenario
        mock_session.query.return_value.filter.return_value.count.return_value = (
            100  # total scores
        )
        mock_session.query.return_value.filter.return_value.filter.return_value.count.return_value = 30  # with xG

        # Mock gameweeks
        mock_session.query.return_value.filter.return_value.distinct.return_value.all.return_value = [
            (1,),
            (2,),
            (3,),
        ]

        validator = BackfillValidator()
        issues = validator._validate_missing_data("2324", None)

        # Should detect low coverage
        coverage_issues = [i for i in issues if i.category == "data_coverage"]
        assert len(coverage_issues) > 0
        assert coverage_issues[0].severity == ValidationSeverity.WARNING

    @patch("airsenal.framework.data_backfill.session_scope")
    def test_statistical_outlier_detection(self, mock_session_scope):
        """Test statistical outlier detection"""
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock extreme xG value
        mock_score = MagicMock()
        mock_score.id = 1
        mock_score.expected_goals = 4.5  # Extremely high
        mock_score.player_id = 123
        mock_score.fixture_id = 456

        mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = [
            mock_score
        ]
        mock_session.query.return_value.filter.return_value.first.return_value = (
            MagicMock(count_xg=100)
        )

        validator = BackfillValidator()
        issues = validator._validate_statistical_outliers("2324", None)

        # Should detect outlier
        outlier_issues = [i for i in issues if i.category == "statistical_outlier"]
        assert len(outlier_issues) > 0


class TestBackfillResult:
    """Test the BackfillResult class"""

    def test_result_initialization(self):
        """Test result initialization"""
        progress = BackfillProgress(total_items=100, completed_items=50)
        result = BackfillResult(status=BackfillStatus.RUNNING, progress=progress)

        assert result.status == BackfillStatus.RUNNING
        assert result.progress.total_items == 100
        assert len(result.validation_issues) == 0

    def test_add_validation_issue(self):
        """Test adding validation issues"""
        result = BackfillResult(
            status=BackfillStatus.RUNNING, progress=BackfillProgress()
        )

        result.add_validation_issue(
            ValidationSeverity.ERROR,
            "test_category",
            "Test description",
            "test_entity",
            entity_id=123,
            test_metadata="value",
        )

        assert len(result.validation_issues) == 1
        issue = result.validation_issues[0]
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.category == "test_category"
        assert issue.entity_id == 123
        assert issue.metadata["test_metadata"] == "value"

    def test_critical_issues_detection(self):
        """Test critical issues detection"""
        result = BackfillResult(
            status=BackfillStatus.RUNNING, progress=BackfillProgress()
        )

        # Add non-critical issue
        result.add_validation_issue(
            ValidationSeverity.WARNING, "test", "Warning", "entity"
        )
        assert not result.has_critical_issues

        # Add critical issue
        result.add_validation_issue(
            ValidationSeverity.CRITICAL, "test", "Critical error", "entity"
        )
        assert result.has_critical_issues

    def test_issues_by_severity_grouping(self):
        """Test grouping issues by severity"""
        result = BackfillResult(
            status=BackfillStatus.RUNNING, progress=BackfillProgress()
        )

        result.add_validation_issue(
            ValidationSeverity.ERROR, "test", "Error 1", "entity"
        )
        result.add_validation_issue(
            ValidationSeverity.ERROR, "test", "Error 2", "entity"
        )
        result.add_validation_issue(
            ValidationSeverity.WARNING, "test", "Warning 1", "entity"
        )

        grouped = result.issues_by_severity
        assert len(grouped[ValidationSeverity.ERROR]) == 2
        assert len(grouped[ValidationSeverity.WARNING]) == 1


class TestHistoricalDataBackfiller:
    """Test the main HistoricalDataBackfiller class"""

    def test_backfiller_initialization(self):
        """Test backfiller initialization"""
        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        assert backfiller.xg_manager == mock_xg_manager
        assert isinstance(backfiller.validator, BackfillValidator)
        assert backfiller._batch_size == 50  # default
        assert backfiller._max_workers == 4  # default

    def test_configuration(self):
        """Test backfiller configuration"""
        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        backfiller.configure(
            batch_size=100, max_workers=8, retry_attempts=5, retry_delay=2.0
        )

        assert backfiller._batch_size == 100
        assert backfiller._max_workers == 8
        assert backfiller._retry_attempts == 5
        assert backfiller._retry_delay == 2.0

    @patch("airsenal.framework.data_backfill.session_scope")
    @patch("airsenal.framework.data_backfill.BackfillProgressTracker")
    def test_backfill_single_gameweek_success(
        self, mock_tracker_class, mock_session_scope
    ):
        """Test successful single gameweek backfill"""
        # Mock XG manager
        mock_xg_manager = MagicMock()
        mock_xg_manager.sync_xg_data_for_gameweek.return_value = (
            50  # 50 records updated
        )

        backfiller = HistoricalDataBackfiller(mock_xg_manager)
        result = backfiller._backfill_single_gameweek("2324", 1)

        assert result.status == BackfillStatus.COMPLETED
        assert result.progress.completed_items == 1
        mock_xg_manager.sync_xg_data_for_gameweek.assert_called_once_with("2324", 1)

    @patch("airsenal.framework.data_backfill.session_scope")
    @patch("airsenal.framework.data_backfill.BackfillProgressTracker")
    def test_backfill_single_gameweek_no_data(
        self, mock_tracker_class, mock_session_scope
    ):
        """Test gameweek backfill with no data updated"""
        # Mock XG manager returning 0 updates
        mock_xg_manager = MagicMock()
        mock_xg_manager.sync_xg_data_for_gameweek.return_value = 0

        backfiller = HistoricalDataBackfiller(mock_xg_manager)
        result = backfiller._backfill_single_gameweek("2324", 1)

        assert result.status == BackfillStatus.COMPLETED
        assert result.progress.skipped_items == 1
        assert len(result.validation_issues) == 1
        assert result.validation_issues[0].category == "no_updates"

    @patch("airsenal.framework.data_backfill.session_scope")
    @patch("airsenal.framework.data_backfill.BackfillProgressTracker")
    def test_backfill_single_gameweek_failure(
        self, mock_tracker_class, mock_session_scope
    ):
        """Test gameweek backfill failure"""
        # Mock XG manager raising exception
        mock_xg_manager = MagicMock()
        mock_xg_manager.sync_xg_data_for_gameweek.side_effect = Exception("API Error")

        backfiller = HistoricalDataBackfiller(mock_xg_manager)
        result = backfiller._backfill_single_gameweek("2324", 1)

        assert result.status == BackfillStatus.FAILED
        assert result.progress.failed_items == 1
        assert len(result.validation_issues) == 1
        assert "API Error" in result.validation_issues[0].description

    @patch("airsenal.framework.data_backfill.session_scope")
    @patch("airsenal.framework.data_backfill.BackfillProgressTracker")
    def test_backfill_gameweeks(self, mock_tracker_class, mock_session_scope):
        """Test backfilling specific gameweeks"""
        # Mock successful XG manager
        mock_xg_manager = MagicMock()
        mock_xg_manager.sync_xg_data_for_gameweek.return_value = 25

        # Mock validator
        mock_validator = MagicMock()
        mock_validator.validate_backfilled_data.return_value = []

        # Mock tracker
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.progress = BackfillProgress()

        backfiller = HistoricalDataBackfiller(mock_xg_manager)
        backfiller.validator = mock_validator

        with patch.object(backfiller, "_generate_reconciliation_report") as mock_report:
            mock_report.return_value = {"test": "report"}

            result = backfiller.backfill_gameweeks("2324", [1, 2, 3])

        assert result.status == BackfillStatus.COMPLETED
        mock_xg_manager.sync_xg_data_for_gameweek.assert_has_calls(
            [call("2324", 1), call("2324", 2), call("2324", 3)]
        )

    @patch("airsenal.framework.data_backfill.session_scope")
    @patch("airsenal.framework.data_backfill.BackfillProgressTracker")
    def test_backfill_season_no_fixtures(self, mock_tracker_class, mock_session_scope):
        """Test season backfill with no fixtures found"""
        # Mock empty fixtures query
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.distinct.return_value.order_by.return_value.all.return_value = []

        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        result = backfiller.backfill_season("2324")

        assert result.status == BackfillStatus.FAILED
        assert result.has_critical_issues
        assert any(
            "No fixtures found" in issue.description
            for issue in result.validation_issues
        )

    @patch("airsenal.framework.data_backfill.session_scope")
    def test_generate_reconciliation_report(self, mock_session_scope):
        """Test reconciliation report generation"""
        # Mock database session and queries
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock counts for coverage calculation
        mock_session.query.return_value.filter.return_value.count.return_value = 100
        mock_session.query.return_value.filter.return_value.filter.return_value.count.return_value = 85

        # Mock xG statistics
        mock_stats = MagicMock()
        mock_stats.avg_xg = 0.5
        mock_stats.max_xg = 2.5
        mock_stats.min_xg = 0.0
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_stats
        )

        # Mock gameweeks
        mock_session.query.return_value.filter.return_value.distinct.return_value.all.return_value = [
            (1,),
            (2,),
            (3,),
        ]

        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        report = backfiller._generate_reconciliation_report("2324")

        assert report["season"] == "2324"
        assert "data_coverage" in report
        assert "quality_metrics" in report
        assert report["data_coverage"]["xg_coverage_percentage"] == 85.0

    @patch("airsenal.framework.data_backfill.session_scope")
    def test_incremental_backfill(self, mock_session_scope):
        """Test incremental backfill functionality"""
        # Mock database session
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock recent fixtures
        recent_date = datetime.now() - timedelta(days=3)
        mock_fixture = MagicMock()
        mock_fixture.date = recent_date.isoformat()
        mock_fixture.gameweek = 10

        mock_session.query.return_value.filter.return_value.all.return_value = [
            mock_fixture
        ]

        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        with patch.object(backfiller, "backfill_gameweeks") as mock_backfill:
            mock_result = BackfillResult(BackfillStatus.COMPLETED, BackfillProgress())
            mock_backfill.return_value = mock_result

            result = backfiller.incremental_backfill("2324", days_back=7)

        assert result.status == BackfillStatus.COMPLETED
        mock_backfill.assert_called_once_with("2324", [10])

    def test_missing_data_report_generation(self):
        """Test missing data report generation"""
        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        # Mock validator
        mock_validator = MagicMock()
        mock_issue = ValidationIssue(
            ValidationSeverity.WARNING, "data_coverage", "Low coverage", "season"
        )
        mock_validator.validate_backfilled_data.return_value = [mock_issue]
        backfiller.validator = mock_validator

        # Mock reconciliation report
        mock_reconciliation = {
            "data_coverage": {"xg_coverage_percentage": 75.5},
            "gaps_identified": [{"gameweek": 5, "severity": "high"}],
        }

        with patch.object(backfiller, "_generate_reconciliation_report") as mock_report:
            mock_report.return_value = mock_reconciliation

            report = backfiller.get_missing_data_report("2324")

        assert report["season"] == "2324"
        assert len(report["missing_data_issues"]) == 1
        assert report["summary"]["overall_coverage"] == 75.5
        assert report["summary"]["high_priority_gaps"] == 1


class TestIntegration:
    """Integration tests for the backfill system"""

    @pytest.mark.integration
    def test_end_to_end_backfill_workflow(self):
        """Test complete backfill workflow"""
        # This would require actual database setup and mock data
        # For now, we'll test the workflow with mocks

        mock_xg_manager = MagicMock()
        mock_xg_manager.sync_xg_data_for_gameweek.return_value = 25

        backfiller = HistoricalDataBackfiller(mock_xg_manager)
        backfiller.configure(batch_size=2, max_workers=2)

        with patch("airsenal.framework.data_backfill.session_scope"):
            with patch.object(
                backfiller.validator, "validate_backfilled_data"
            ) as mock_validate:
                mock_validate.return_value = []

                with patch.object(
                    backfiller, "_generate_reconciliation_report"
                ) as mock_report:
                    mock_report.return_value = {"test": "report"}

                    result = backfiller.backfill_gameweeks("2324", [1, 2, 3, 4])

        assert result.status == BackfillStatus.COMPLETED
        # Should have called sync for each gameweek
        assert mock_xg_manager.sync_xg_data_for_gameweek.call_count == 4

    @pytest.mark.performance
    def test_parallel_processing_performance(self):
        """Test that parallel processing improves performance"""
        mock_xg_manager = MagicMock()

        # Simulate slow operation
        def slow_sync(season, gameweek):
            time.sleep(0.1)  # 100ms delay
            return 10

        mock_xg_manager.sync_xg_data_for_gameweek.side_effect = slow_sync

        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        # Test with 1 worker (sequential)
        backfiller.configure(max_workers=1)
        start_time = time.time()

        with patch("airsenal.framework.data_backfill.session_scope"):
            with patch.object(
                backfiller.validator, "validate_backfilled_data", return_value=[]
            ):
                with patch.object(
                    backfiller, "_generate_reconciliation_report", return_value={}
                ):
                    result1 = backfiller.backfill_gameweeks("2324", [1, 2, 3, 4])

        sequential_time = time.time() - start_time

        # Test with 4 workers (parallel)
        backfiller.configure(max_workers=4)
        start_time = time.time()

        with patch("airsenal.framework.data_backfill.session_scope"):
            with patch.object(
                backfiller.validator, "validate_backfilled_data", return_value=[]
            ):
                with patch.object(
                    backfiller, "_generate_reconciliation_report", return_value={}
                ):
                    result2 = backfiller.backfill_gameweeks("2324", [1, 2, 3, 4])

        parallel_time = time.time() - start_time

        # Parallel should be faster (though not 4x due to overhead)
        assert parallel_time < sequential_time
        assert result1.status == BackfillStatus.COMPLETED
        assert result2.status == BackfillStatus.COMPLETED


# Fixtures and utilities for testing
@pytest.fixture
def mock_xg_data_point():
    """Create a mock XGDataPoint for testing"""
    return XGDataPoint(
        player_id=123,
        fixture_id=456,
        match_date=datetime.now(),
        expected_goals=1.5,
        expected_assists=0.5,
        expected_goal_involvements=2.0,
        provider="test_provider",
    )


@pytest.fixture
def sample_validation_issues():
    """Create sample validation issues for testing"""
    return [
        ValidationIssue(
            ValidationSeverity.ERROR,
            "data_integrity",
            "Missing player reference",
            "player_score",
            entity_id=123,
        ),
        ValidationIssue(
            ValidationSeverity.WARNING,
            "data_coverage",
            "Low xG coverage",
            "gameweek",
            entity_id=5,
        ),
        ValidationIssue(
            ValidationSeverity.CRITICAL,
            "system_error",
            "Database connection failed",
            "system",
        ),
    ]


@pytest.fixture
def mock_database_session():
    """Create a mock database session for testing"""
    with patch("airsenal.framework.data_backfill.session_scope") as mock_session_scope:
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        yield mock_session


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_database_connection_failure(self):
        """Test handling of database connection failures"""
        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        with patch("airsenal.framework.data_backfill.session_scope") as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            result = backfiller.backfill_gameweeks("2324", [1])

        assert result.status == BackfillStatus.FAILED
        assert result.has_critical_issues

    def test_api_rate_limit_handling(self):
        """Test handling of API rate limiting"""
        mock_xg_manager = MagicMock()
        mock_xg_manager.sync_xg_data_for_gameweek.side_effect = Exception(
            "Rate limit exceeded"
        )

        backfiller = HistoricalDataBackfiller(mock_xg_manager)
        backfiller.configure(retry_attempts=2, retry_delay=0.1)

        with patch("airsenal.framework.data_backfill.session_scope"):
            result = backfiller._backfill_single_gameweek("2324", 1)

        assert result.status == BackfillStatus.FAILED

    def test_invalid_season_format(self):
        """Test handling of invalid season formats"""
        mock_xg_manager = MagicMock()
        backfiller = HistoricalDataBackfiller(mock_xg_manager)

        with patch(
            "airsenal.framework.data_backfill.session_scope"
        ) as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.distinct.return_value.order_by.return_value.all.return_value = []

            result = backfiller.backfill_season("invalid_season")

        assert result.status == BackfillStatus.FAILED


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
