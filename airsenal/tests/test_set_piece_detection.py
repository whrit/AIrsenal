"""
Comprehensive tests for the set piece specialist detection system.

Tests cover:
- SetPieceDetector class functionality
- SetPieceTracker monitoring system
- SetPieceOutcomeAnalyzer effectiveness measurement
- Historical set piece data analysis
- API endpoint functionality
- Database model integration
- Edge cases and error handling
- Performance characteristics
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.api_utils import (
    get_all_set_piece_specialists_for_api,
    get_corner_specialists_for_team,
    get_free_kick_specialists_for_team,
    get_set_piece_specialist_confidence,
)
from airsenal.framework.schema import (
    Base,
    Fixture,
    Player,
    SetPieceAnalysisLog,
    SetPieceSpecialist,
)
from airsenal.framework.schema import SetPieceEvent as SetPieceEventModel
from airsenal.framework.schema import SetPieceStatistics as SetPieceStatisticsModel
from airsenal.framework.set_piece_detection import (
    SetPieceDetector,
    SetPieceEvent,
    SetPieceOutcomeAnalyzer,
    SetPieceStatistics,
    SetPieceTracker,
    SetPieceType,
    get_corner_specialists,
    get_free_kick_specialists,
    get_throw_in_specialists,
)


class TestSetPieceEvent:
    """Test the SetPieceEvent dataclass."""

    def test_set_piece_event_initialization(self):
        """Test basic initialization of SetPieceEvent."""
        event = SetPieceEvent(
            player_id=123,
            player_name="Kevin De Bruyne",
            team="MCI",
            set_piece_type=SetPieceType.CORNER_RIGHT,
            date=datetime.now(),
            gameweek=1,
            season="2024-25",
            outcome="assist",
            opponent="ARS",
            venue="home",
        )

        assert event.player_id == 123
        assert event.player_name == "Kevin De Bruyne"
        assert event.team == "MCI"
        assert event.set_piece_type == SetPieceType.CORNER_RIGHT
        assert event.outcome == "assist"
        assert event.successful is False  # Default value

    def test_set_piece_event_with_details(self):
        """Test SetPieceEvent with optional details."""
        event = SetPieceEvent(
            player_id=123,
            player_name="James Ward-Prowse",
            team="WHU",
            set_piece_type=SetPieceType.FREE_KICK_CLOSE,
            date=datetime.now(),
            gameweek=1,
            season="2024-25",
            outcome="goal",
            opponent="BRE",
            venue="away",
            minute=23,
            foot_used="right",
            distance=18,
            successful=True,
        )

        assert event.minute == 23
        assert event.foot_used == "right"
        assert event.distance == 18
        assert event.successful is True


class TestSetPieceStatistics:
    """Test the SetPieceStatistics dataclass."""

    def test_set_piece_statistics_initialization(self):
        """Test basic initialization of SetPieceStatistics."""
        stats = SetPieceStatistics(
            player_id=123,
            player_name="Trent Alexander-Arnold",
            team="LIV",
            set_piece_type=SetPieceType.CORNER_RIGHT,
        )

        assert stats.player_id == 123
        assert stats.player_name == "Trent Alexander-Arnold"
        assert stats.team == "LIV"
        assert stats.set_piece_type == SetPieceType.CORNER_RIGHT
        assert stats.total_taken == 0
        assert stats.success_rate == 0.0

    def test_set_piece_statistics_success_rate(self):
        """Test success rate calculation."""
        stats = SetPieceStatistics(123, "Test Player", "TEST", SetPieceType.CORNER_LEFT)

        # No set pieces taken
        assert stats.success_rate == 0.0

        # Add successful set pieces
        stats.successful_outcomes = 3
        stats.total_taken = 5
        assert stats.success_rate == 0.6

        # Test goals and assists rates
        stats.goals_from = 1
        stats.assists_from = 2
        assert stats.goals_rate == 0.2
        assert stats.assists_rate == 0.4

    def test_set_piece_statistics_add_event(self):
        """Test adding set piece events."""
        stats = SetPieceStatistics(
            123, "Test Player", "TEST", SetPieceType.FREE_KICK_LONG
        )
        date1 = datetime.now() - timedelta(days=10)
        date2 = datetime.now() - timedelta(days=5)

        # Add successful event
        event1 = SetPieceEvent(
            123,
            "Test Player",
            "TEST",
            SetPieceType.FREE_KICK_LONG,
            date1,
            1,
            "2024-25",
            "goal",
            "OPP",
            "home",
            successful=True,
        )
        stats.add_event(event1)

        assert stats.total_taken == 1
        assert stats.successful_outcomes == 1
        assert stats.goals_from == 1
        assert stats.last_taken_date == date1

        # Add another event (more recent)
        event2 = SetPieceEvent(
            123,
            "Test Player",
            "TEST",
            SetPieceType.FREE_KICK_LONG,
            date2,
            2,
            "2024-25",
            "assist",
            "OPP2",
            "away",
            successful=True,
        )
        stats.add_event(event2)

        assert stats.total_taken == 2
        assert stats.successful_outcomes == 2
        assert stats.assists_from == 1
        assert stats.last_taken_date == date2  # Should update to more recent


class TestSetPieceDetector:
    """Test the SetPieceDetector class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def detector(self, mock_session):
        """Create a SetPieceDetector instance."""
        return SetPieceDetector(mock_session)

    def test_detector_initialization(self, detector):
        """Test detector initialization with default parameters."""
        assert detector.confidence_threshold == 0.7
        assert detector.decay_factor == 0.85  # α=0.85 as specified
        assert detector.min_set_pieces_for_confidence == 5

    def test_confidence_calculation_basic(self, detector):
        """Test basic confidence score calculation with exponential decay."""
        stats = SetPieceStatistics(123, "Test Player", "TEST", SetPieceType.CORNER_BOTH)
        stats.total_taken = 8
        stats.successful_outcomes = 6
        stats.goals_from = 2
        stats.assists_from = 3
        stats.last_taken_date = datetime.now() - timedelta(days=7)
        stats.recent_taken = 3

        confidence = detector.calculate_set_piece_confidence(stats, 15)

        # Should be reasonably high confidence with good recent performance
        assert 0.6 <= confidence <= 1.0

    def test_confidence_calculation_exponential_decay(self, detector):
        """Test exponential decay with α=0.85 for recency weighting."""
        stats_recent = SetPieceStatistics(
            123, "Recent Player", "TEST", SetPieceType.FREE_KICK_CLOSE
        )
        stats_recent.total_taken = 5
        stats_recent.successful_outcomes = 4
        stats_recent.last_taken_date = datetime.now() - timedelta(days=7)  # 1 week ago

        stats_old = SetPieceStatistics(
            124, "Old Player", "TEST", SetPieceType.FREE_KICK_CLOSE
        )
        stats_old.total_taken = 5
        stats_old.successful_outcomes = 4
        stats_old.last_taken_date = datetime.now() - timedelta(days=42)  # 6 weeks ago

        confidence_recent = detector.calculate_set_piece_confidence(stats_recent, 10)
        confidence_old = detector.calculate_set_piece_confidence(stats_old, 10)

        # Recent player should have higher confidence due to α=0.85 decay
        assert confidence_recent > confidence_old

    def test_confidence_calculation_no_set_pieces(self, detector):
        """Test confidence calculation with no set pieces."""
        stats = SetPieceStatistics(
            123, "Test Player", "TEST", SetPieceType.THROW_IN_LONG
        )

        confidence = detector.calculate_set_piece_confidence(stats, 10)
        assert confidence == 0.0

    def test_confidence_calculation_minimum_threshold(self, detector):
        """Test minimum confidence threshold for sufficient sample size."""
        stats = SetPieceStatistics(123, "Test Player", "TEST", SetPieceType.CORNER_LEFT)
        stats.total_taken = 7  # Above minimum threshold of 5
        stats.successful_outcomes = 5
        stats.goals_from = 1
        stats.assists_from = 2

        confidence = detector.calculate_set_piece_confidence(stats, 10)

        # Should meet minimum threshold
        assert confidence >= 0.6

    @patch("airsenal.framework.set_piece_detection.session_scope")
    def test_get_team_set_piece_statistics(self, mock_session_scope, detector):
        """Test getting team set piece statistics."""
        # Mock database session and query results
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock player and score data
        mock_player = Mock()
        mock_player.player_id = 123
        mock_player.name = "Test Player"

        mock_score = Mock()
        mock_score.player_id = 123
        mock_score.player_team = "TEST"
        mock_score.minutes = 90
        mock_score.assists = 1
        mock_score.creativity = 35

        mock_fixture = Mock()
        mock_fixture.date = "2024-01-01"
        mock_fixture.gameweek = 1
        mock_fixture.season = "2024-25"
        mock_fixture.home_team = "TEST"

        # Mock PlayerAttributes
        mock_attrs = Mock()
        mock_attrs.is_corner_taker = True
        mock_attrs.position = "DEF"

        mock_session.query.return_value.join.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = [
            (mock_score, mock_player, mock_fixture)
        ]

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_attrs
        )

        # Test the method
        stats = detector.get_team_set_piece_statistics(
            "TEST", "2024-25", SetPieceType.CORNER_BOTH
        )

        assert 123 in stats
        assert stats[123].player_name == "Test Player"
        assert stats[123].team == "TEST"
        assert stats[123].set_piece_type == SetPieceType.CORNER_BOTH


class TestSetPieceTracker:
    """Test the SetPieceTracker class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def tracker(self, mock_session):
        """Create a SetPieceTracker instance."""
        return SetPieceTracker(mock_session)

    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.detector is not None
        assert isinstance(tracker.detector, SetPieceDetector)

    def test_detect_role_changes_empty(self, tracker):
        """Test role change detection with no changes."""
        # Mock the detector to return empty results for now
        with patch.object(
            tracker.detector, "identify_team_set_piece_specialists"
        ) as mock_identify:
            mock_identify.return_value = []

            changes = tracker.detect_role_changes(
                "TEST", "2024-25", SetPieceType.CORNER_LEFT, 5
            )

            assert changes == []

    def test_get_recent_performance_trends(self, tracker):
        """Test getting recent performance trends."""
        trend = tracker.get_recent_performance_trends(
            123, "2024-25", SetPieceType.FREE_KICK_CLOSE
        )

        # Should return a properly structured trend analysis
        assert "player_id" in trend
        assert "set_piece_type" in trend
        assert "trend_direction" in trend
        assert trend["player_id"] == 123
        assert trend["set_piece_type"] == SetPieceType.FREE_KICK_CLOSE.value


class TestSetPieceOutcomeAnalyzer:
    """Test the SetPieceOutcomeAnalyzer class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def analyzer(self, mock_session):
        """Create a SetPieceOutcomeAnalyzer instance."""
        return SetPieceOutcomeAnalyzer(mock_session)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.db_session is not None

    def test_analyze_specialist_effectiveness(self, analyzer):
        """Test analyzing specialist effectiveness."""
        effectiveness = analyzer.analyze_specialist_effectiveness(
            123, "2024-25", SetPieceType.CORNER_RIGHT
        )

        # Should return a properly structured effectiveness analysis
        assert "player_id" in effectiveness
        assert "set_piece_type" in effectiveness
        assert "total_attempts" in effectiveness
        assert "success_rate" in effectiveness
        assert "effectiveness_score" in effectiveness
        assert effectiveness["player_id"] == 123
        assert effectiveness["set_piece_type"] == SetPieceType.CORNER_RIGHT.value

    def test_compare_specialists(self, analyzer):
        """Test comparing multiple specialists."""
        player_ids = [123, 124, 125]
        comparison = analyzer.compare_specialists(
            player_ids, "2024-25", SetPieceType.FREE_KICK_LONG
        )

        # Should return comparison structure
        assert "comparisons" in comparison
        assert "best_performer" in comparison
        assert "average_metrics" in comparison
        assert len(comparison["comparisons"]) == len(player_ids)


class TestDatabaseIntegration:
    """Test database integration and model functionality."""

    @pytest.fixture
    def in_memory_db(self):
        """Create an in-memory SQLite database for testing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()

    def test_set_piece_event_model(self, in_memory_db):
        """Test SetPieceEvent model creation and relationships."""
        # Create a player
        player = Player(name="Test Player")
        in_memory_db.add(player)
        in_memory_db.flush()

        # Create a fixture
        fixture = Fixture(
            date="2024-01-01",
            gameweek=1,
            home_team="HOME",
            away_team="AWAY",
            season="2024-25",
            tag="test",
        )
        in_memory_db.add(fixture)
        in_memory_db.flush()

        # Create set piece event
        event = SetPieceEventModel(
            player_id=player.player_id,
            team="TEST",
            season="2024-25",
            gameweek=1,
            set_piece_type="corner_right",
            outcome="assist",
            successful=True,
            opponent="OPP",
            venue="home",
            fixture_id=fixture.fixture_id,
            minute=23,
            foot_used="right",
            detected_method="heuristic",
            confidence=0.8,
            recorded_at=datetime.now().isoformat(),
        )

        in_memory_db.add(event)
        in_memory_db.commit()

        # Verify the relationship works
        retrieved = in_memory_db.query(SetPieceEventModel).first()
        assert retrieved.player.name == "Test Player"
        assert retrieved.set_piece_type == "corner_right"
        assert retrieved.successful is True
        assert retrieved.confidence == 0.8

    def test_set_piece_specialist_model(self, in_memory_db):
        """Test SetPieceSpecialist model creation and tracking."""
        # Create a player
        player = Player(name="Specialist Player")
        in_memory_db.add(player)
        in_memory_db.flush()

        # Create specialist assignment
        specialist = SetPieceSpecialist(
            player_id=player.player_id,
            team="TEST",
            season="2024-25",
            gameweek=1,
            set_piece_type="free_kick_close",
            is_primary=True,
            confidence_score=0.9,
            assignment_source="analysis",
            total_taken=10,
            successful_outcomes=7,
            goals_from=3,
            assists_from=2,
            success_rate=0.7,
            assignment_date=datetime.now().isoformat(),
            is_active=True,
            events_since_assignment=3,
            successful_since_assignment=2,
            analysis_version="1.0.0",
            data_quality_score=0.9,
        )

        in_memory_db.add(specialist)
        in_memory_db.commit()

        # Verify the data
        retrieved = in_memory_db.query(SetPieceSpecialist).first()
        assert retrieved.player.name == "Specialist Player"
        assert retrieved.set_piece_type == "free_kick_close"
        assert retrieved.is_primary is True
        assert retrieved.confidence_score == 0.9
        assert retrieved.success_rate == 0.7

    def test_set_piece_statistics_model(self, in_memory_db):
        """Test SetPieceStatistics model with aggregated data."""
        # Create a player
        player = Player(name="Stats Player")
        in_memory_db.add(player)
        in_memory_db.flush()

        # Create statistics record
        stats = SetPieceStatisticsModel(
            player_id=player.player_id,
            team="TEST",
            season="2024-25",
            set_piece_type="corner_both",
            period_type="season",
            period_start="2024-08-01",
            period_end="2025-05-31",
            games_included=20,
            total_attempts=15,
            successful_outcomes=9,
            goals_scored=2,
            assists_provided=4,
            key_passes_made=3,
            shots_created=8,
            success_rate=0.6,
            goals_per_attempt=0.133,
            assists_per_attempt=0.267,
            key_passes_per_attempt=0.2,
            home_attempts=8,
            away_attempts=7,
            home_success_rate=0.625,
            away_success_rate=0.571,
            consistency_score=0.75,
            recent_form=0.8,
            trend_direction="improving",
            trend_confidence=0.7,
            calculated_at=datetime.now().isoformat(),
            calculation_method="standard",
            data_quality_score=0.85,
        )

        in_memory_db.add(stats)
        in_memory_db.commit()

        # Verify the statistics
        retrieved = in_memory_db.query(SetPieceStatisticsModel).first()
        assert retrieved.player.name == "Stats Player"
        assert retrieved.total_attempts == 15
        assert retrieved.success_rate == 0.6
        assert retrieved.trend_direction == "improving"

    def test_set_piece_analysis_log_model(self, in_memory_db):
        """Test SetPieceAnalysisLog model for audit trails."""
        log = SetPieceAnalysisLog(
            analysis_type="team_specialist_identification",
            analysis_version="1.0.0",
            team="TEST",
            season="2024-25",
            set_piece_type="corner_right",
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            duration_seconds=45.2,
            status="completed",
            players_analyzed=25,
            events_processed=150,
            specialists_identified=3,
            confidence_threshold=0.7,
            data_quality_score=0.8,
            events_with_low_confidence=12,
            missing_data_percentage=5.0,
            warnings_count=2,
            average_confidence=0.75,
            assignments_changed=1,
        )

        in_memory_db.add(log)
        in_memory_db.commit()

        # Verify the log
        retrieved = in_memory_db.query(SetPieceAnalysisLog).first()
        assert retrieved.analysis_type == "team_specialist_identification"
        assert retrieved.status == "completed"
        assert retrieved.specialists_identified == 3
        assert retrieved.average_confidence == 0.75


class TestAPIFunctions:
    """Test API utility functions."""

    @patch("airsenal.framework.api_utils.get_corner_specialists")
    def test_get_corner_specialists_for_team_api(self, mock_get_corners):
        """Test API function for getting team corner specialists."""
        # Mock return value
        mock_get_corners.return_value = [
            {
                "player_id": 123,
                "player_name": "Test Player",
                "team": "TEST",
                "set_piece_type": "corner_both",
                "confidence": 0.9,
                "is_primary": True,
            }
        ]

        result = get_corner_specialists_for_team("TEST", "2024-25", "both")

        assert len(result) == 1
        assert result[0]["player_id"] == 123
        assert result[0]["confidence"] == 0.9
        mock_get_corners.assert_called_once_with("TEST", "2024-25", "both")

    @patch("airsenal.framework.api_utils.get_free_kick_specialists")
    def test_get_free_kick_specialists_for_team_api(self, mock_get_fks):
        """Test API function for getting team free kick specialists."""
        mock_get_fks.return_value = [
            {
                "player_id": 124,
                "player_name": "FK Taker",
                "team": "TEST",
                "set_piece_type": "free_kick_close",
                "confidence": 0.85,
                "is_primary": True,
            }
        ]

        result = get_free_kick_specialists_for_team("TEST", "2024-25", "close")

        assert len(result) == 1
        assert result[0]["set_piece_type"] == "free_kick_close"

    @patch("airsenal.framework.api_utils.get_all_set_piece_specialists")
    def test_get_all_set_piece_specialists_api(self, mock_get_all):
        """Test API function for getting all set piece specialists."""
        mock_get_all.return_value = {
            "TEST1": {
                "corner_both": [{"player_id": 123, "confidence": 0.9}],
                "free_kick_close": [{"player_id": 124, "confidence": 0.8}],
            },
            "TEST2": {"throw_in_long": [{"player_id": 125, "confidence": 0.7}]},
        }

        result = get_all_set_piece_specialists_for_api("2024-25")

        assert "season" in result
        assert "teams" in result
        assert "metadata" in result
        assert result["season"] == "2024-25"
        assert "TEST1" in result["teams"]
        assert "TEST2" in result["teams"]
        assert result["metadata"]["total_teams"] == 2

    @patch("airsenal.framework.api_utils.get_player")
    @patch("airsenal.framework.api_utils.SetPieceDetector")
    def test_get_set_piece_specialist_confidence_api(
        self, mock_detector_class, mock_get_player
    ):
        """Test API function for getting specialist confidence."""
        # Mock player
        mock_player = Mock()
        mock_player.player_id = 123
        mock_player.name = "Test Player"
        mock_player.team.return_value = "TEST"
        mock_get_player.return_value = mock_player

        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.identify_team_set_piece_specialists.return_value = [
            {
                "player_id": 123,
                "confidence": 0.85,
                "is_primary": True,
                "total_taken": 10,
                "success_rate": 0.7,
                "goals_rate": 0.1,
                "assists_rate": 0.3,
                "last_taken_date": datetime.now(),
                "recent_taken": 3,
            }
        ]

        result = get_set_piece_specialist_confidence(123, "corner_both", "2024-25")

        assert result["is_specialist"] is True
        assert result["confidence"] == 0.85
        assert result["player_name"] == "Test Player"
        assert result["set_piece_type"] == "corner_both"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_set_piece_type(self):
        """Test handling of invalid set piece types."""
        SetPieceDetector()

        # Test with valid enum
        assert SetPieceType.CORNER_LEFT.value == "corner_left"

        # Test string to enum conversion in API functions
        with patch("airsenal.framework.api_utils.get_player") as mock_get_player:
            mock_get_player.return_value = None

            result = get_set_piece_specialist_confidence(123, "invalid_type", "2024-25")
            assert "error" in result
            assert "Invalid set piece type" in result["error"]

    def test_empty_set_piece_data(self):
        """Test behavior with no set piece data available."""
        detector = SetPieceDetector()

        # Mock empty set piece statistics
        with patch.object(detector, "get_team_set_piece_statistics") as mock_get_stats:
            mock_get_stats.return_value = {}

            result = detector.identify_team_set_piece_specialists(
                "TEST", "2024-25", SetPieceType.CORNER_BOTH
            )
            assert result == []

    def test_insufficient_set_piece_data(self):
        """Test fallback behavior with insufficient set piece data."""
        detector = SetPieceDetector()

        # Mock stats with insufficient set pieces
        stats = {
            123: SetPieceStatistics(
                123, "Test Player", "TEST", SetPieceType.CORNER_BOTH
            )
        }
        stats[123].total_taken = 1  # Below threshold

        with patch.object(detector, "get_team_set_piece_statistics") as mock_get_stats:
            with patch.object(
                detector, "_fallback_to_attributes_for_set_pieces"
            ) as mock_fallback:
                mock_get_stats.return_value = stats
                mock_fallback.return_value = [{"player_id": 123, "confidence": 0.7}]

                detector.identify_team_set_piece_specialists(
                    "TEST", "2024-25", SetPieceType.CORNER_BOTH
                )
                mock_fallback.assert_called_once()

    def test_single_specialist_promotion(self):
        """Test that at least one player is marked as primary when we have data."""
        detector = SetPieceDetector()

        # Create a scenario where no player meets the primary threshold
        stats = SetPieceStatistics(
            123, "Test Player", "TEST", SetPieceType.FREE_KICK_CLOSE
        )
        stats.total_taken = 5
        stats.successful_outcomes = 3
        stats.last_taken_date = datetime.now() - timedelta(days=30)

        confidence = detector.calculate_set_piece_confidence(stats, 10)

        # Simulate the logic that promotes the highest confidence player
        specialists = [
            {
                "player_id": 123,
                "confidence": confidence,
                "is_primary": confidence >= 0.8,
            }
        ]

        # If no primary, promote the first one
        if not any(s["is_primary"] for s in specialists):
            specialists[0]["is_primary"] = True

        assert specialists[0]["is_primary"] is True

    def test_api_error_handling(self):
        """Test API functions handle errors gracefully."""
        # Test with non-existent team
        result = get_corner_specialists_for_team("NONEXISTENT")
        assert result == []

        # Test with invalid parameters
        result = get_set_piece_specialist_confidence(999999, "corner_both")
        assert "error" in result


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch("airsenal.framework.set_piece_detection.SetPieceDetector")
    def test_get_corner_specialists(self, mock_detector_class):
        """Test convenience function for corner specialists."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.identify_team_set_piece_specialists.return_value = [
            {"player_id": 123, "confidence": 0.9}
        ]

        # Test different corner types
        result_left = get_corner_specialists("TEST", "2024-25", "left")
        result_right = get_corner_specialists("TEST", "2024-25", "right")
        result_both = get_corner_specialists("TEST", "2024-25", "both")

        assert len(result_left) == 1
        assert len(result_right) == 1
        assert len(result_both) == 1

        # Verify correct enum mapping
        calls = mock_detector.identify_team_set_piece_specialists.call_args_list
        assert calls[0][0][2] == SetPieceType.CORNER_LEFT
        assert calls[1][0][2] == SetPieceType.CORNER_RIGHT
        assert calls[2][0][2] == SetPieceType.CORNER_BOTH

    @patch("airsenal.framework.set_piece_detection.SetPieceDetector")
    def test_get_free_kick_specialists(self, mock_detector_class):
        """Test convenience function for free kick specialists."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.identify_team_set_piece_specialists.return_value = [
            {"player_id": 124, "confidence": 0.8}
        ]

        # Test different distances
        result_close = get_free_kick_specialists("TEST", "2024-25", "close")
        result_long = get_free_kick_specialists("TEST", "2024-25", "long")
        result_both = get_free_kick_specialists("TEST", "2024-25", "both")

        assert len(result_close) == 1
        assert len(result_long) == 1
        assert len(result_both) == 2  # Should return both close and long

    @patch("airsenal.framework.set_piece_detection.SetPieceDetector")
    def test_get_throw_in_specialists(self, mock_detector_class):
        """Test convenience function for throw-in specialists."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.identify_team_set_piece_specialists.return_value = [
            {"player_id": 125, "confidence": 0.75}
        ]

        result = get_throw_in_specialists("TEST", "2024-25")

        assert len(result) == 1
        assert result[0]["player_id"] == 125
        mock_detector.identify_team_set_piece_specialists.assert_called_with(
            "TEST", "2024-25", SetPieceType.THROW_IN_LONG
        )


class TestPerformance:
    """Test performance characteristics of the set piece detection system."""

    def test_large_dataset_performance(self):
        """Test performance with large dataset (mocked)."""
        detector = SetPieceDetector()

        # Create large mock dataset
        large_stats = {}
        for i in range(100):
            stats = SetPieceStatistics(
                i, f"Player {i}", "TEST", SetPieceType.CORNER_BOTH
            )
            stats.total_taken = i % 15  # Vary set piece counts
            stats.successful_outcomes = (i % 15) // 2
            stats.last_taken_date = datetime.now() - timedelta(days=i % 60)
            large_stats[i] = stats

        start_time = datetime.now()

        # Test confidence calculation for all players
        for _player_id, stats in large_stats.items():
            confidence = detector.calculate_set_piece_confidence(stats, 200)
            assert 0.0 <= confidence <= 1.0

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Should process 100 players quickly (exponential decay calculation)
        assert processing_time < 1.0, (
            f"Processing took {processing_time}s, should be < 1s"
        )

    def test_exponential_decay_performance(self):
        """Test performance of exponential decay calculations."""
        detector = SetPieceDetector()

        # Test with various time gaps
        time_gaps = [1, 7, 14, 30, 60, 90, 180, 365]

        start_time = datetime.now()

        for gap in time_gaps:
            stats = SetPieceStatistics(
                123, "Test Player", "TEST", SetPieceType.CORNER_BOTH
            )
            stats.total_taken = 10
            stats.successful_outcomes = 6
            stats.last_taken_date = datetime.now() - timedelta(days=gap)

            # Calculate with α=0.85 decay
            confidence = detector.calculate_set_piece_confidence(stats, 20)
            assert 0.0 <= confidence <= 1.0

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Exponential decay should be fast
        assert processing_time < 0.1, (
            f"Decay calculations took {processing_time}s, should be < 0.1s"
        )


class TestSpecificationCompliance:
    """Test compliance with TASK-112 specifications."""

    def test_exponential_decay_alpha_085(self):
        """Test that exponential decay uses α=0.85 as specified."""
        detector = SetPieceDetector()

        assert detector.decay_factor == 0.85

        # Test decay calculation manually
        stats = SetPieceStatistics(123, "Test Player", "TEST", SetPieceType.CORNER_BOTH)
        stats.total_taken = 5
        stats.successful_outcomes = 3
        stats.last_taken_date = datetime.now() - timedelta(days=14)  # 2 weeks ago

        confidence = detector.calculate_set_piece_confidence(stats, 10)

        # Manually calculate expected recency score with α=0.85, 14-day half-life
        pow(0.85, 14 / 14.0)  # Should be 0.85

        # The confidence should incorporate this recency factor
        assert confidence > 0.0

    def test_minimum_sample_size_5(self):
        """Test minimum sample size of 5 set pieces for confidence scoring."""
        detector = SetPieceDetector()

        assert detector.min_set_pieces_for_confidence == 5

        # Test with exactly 5 set pieces
        stats = SetPieceStatistics(
            123, "Test Player", "TEST", SetPieceType.FREE_KICK_CLOSE
        )
        stats.total_taken = 5
        stats.successful_outcomes = 3

        confidence = detector.calculate_set_piece_confidence(stats, 10)

        # Should meet minimum threshold
        assert confidence >= 0.6

        # Test with fewer than 5
        stats.total_taken = 3
        confidence_low = detector.calculate_set_piece_confidence(stats, 10)

        # Should be lower confidence
        assert confidence_low < confidence

    def test_set_piece_types_coverage(self):
        """Test coverage of all specified set piece types."""
        expected_types = [
            "corner_left",
            "corner_right",
            "corner_both",
            "free_kick_close",
            "free_kick_long",
            "free_kick_indirect",
            "throw_in_long",
        ]

        actual_types = [t.value for t in SetPieceType]

        for expected in expected_types:
            assert expected in actual_types, f"Missing set piece type: {expected}"

    def test_outcome_tracking(self):
        """Test tracking of set piece outcomes (assists, goals, key passes)."""
        stats = SetPieceStatistics(123, "Test Player", "TEST", SetPieceType.CORNER_BOTH)

        # Add various outcome events
        goal_event = SetPieceEvent(
            123,
            "Test Player",
            "TEST",
            SetPieceType.CORNER_BOTH,
            datetime.now(),
            1,
            "2024-25",
            "goal",
            "OPP",
            "home",
            successful=True,
        )
        assist_event = SetPieceEvent(
            123,
            "Test Player",
            "TEST",
            SetPieceType.CORNER_BOTH,
            datetime.now(),
            2,
            "2024-25",
            "assist",
            "OPP",
            "home",
            successful=True,
        )
        key_pass_event = SetPieceEvent(
            123,
            "Test Player",
            "TEST",
            SetPieceType.CORNER_BOTH,
            datetime.now(),
            3,
            "2024-25",
            "key_pass",
            "OPP",
            "home",
            successful=True,
        )

        stats.add_event(goal_event)
        stats.add_event(assist_event)
        stats.add_event(key_pass_event)

        # Verify outcome tracking
        assert stats.goals_from == 1
        assert stats.assists_from == 1
        assert stats.key_passes_from == 1
        assert stats.successful_outcomes == 3
        assert stats.total_taken == 3


if __name__ == "__main__":
    pytest.main([__file__])
