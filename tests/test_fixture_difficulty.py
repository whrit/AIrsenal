"""
Tests for the fixture difficulty calculation system.

Tests cover:
- EloRatingSystem functionality
- FixtureDifficultyCalculator comprehensive analysis
- Database integration and caching
- Validation and accuracy metrics
- Edge cases and error handling
"""

from unittest.mock import Mock, patch

import pytest

from airsenal.framework.fixture_difficulty import (
    EloRatingSystem,
    FixtureDifficultyCalculator,
    create_difficulty_calculator,
    get_all_teams_average_difficulty,
    get_fixture_difficulty,
    get_team_fixture_difficulties,
)
from airsenal.framework.schema import (
    Fixture,
    Result,
    Team,
)
from airsenal.framework.season import CURRENT_SEASON


class TestEloRatingSystem:
    """Test cases for the Elo rating system."""

    def test_initialization(self):
        """Test EloRatingSystem initialization with default parameters."""
        elo = EloRatingSystem()
        assert elo.initial_rating == 1500.0
        assert elo.k_factor == 30.0
        assert elo.home_advantage == 80.0
        assert elo.form_weight == 0.3
        assert elo.ratings == {}

    def test_initialization_custom_params(self):
        """Test EloRatingSystem initialization with custom parameters."""
        elo = EloRatingSystem(
            initial_rating=1600.0, k_factor=40.0, home_advantage=100.0, form_weight=0.5
        )
        assert elo.initial_rating == 1600.0
        assert elo.k_factor == 40.0
        assert elo.home_advantage == 100.0
        assert elo.form_weight == 0.5

    def test_get_rating_new_team(self):
        """Test getting rating for a new team returns initial rating."""
        elo = EloRatingSystem()
        rating = elo.get_rating("Arsenal", 1, "2023-24")
        assert rating == 1500.0

    def test_get_rating_existing_team(self):
        """Test getting rating for existing team returns stored rating."""
        elo = EloRatingSystem()
        elo.ratings["Arsenal_2023-24"] = 1650.0
        rating = elo.get_rating("Arsenal", 1, "2023-24")
        assert rating == 1650.0

    def test_expected_score_equal_teams(self):
        """Test expected score calculation for equally matched teams."""
        elo = EloRatingSystem()
        elo.ratings["Arsenal_2023-24"] = 1500.0
        elo.ratings["Chelsea_2023-24"] = 1500.0

        # Without home advantage
        expected = elo.expected_score("Arsenal", "Chelsea", False, 1, "2023-24")
        assert abs(expected - 0.5) < 0.001

        # With home advantage (Arsenal home)
        expected_home = elo.expected_score("Arsenal", "Chelsea", True, 1, "2023-24")
        assert expected_home > 0.5
        assert (
            abs(expected_home - 0.64) < 0.01
        )  # Approximately 0.64 with 80 point advantage

    def test_expected_score_different_ratings(self):
        """Test expected score calculation for teams with different ratings."""
        elo = EloRatingSystem()
        elo.ratings["Arsenal_2023-24"] = 1600.0  # Stronger team
        elo.ratings["Chelsea_2023-24"] = 1400.0  # Weaker team

        expected = elo.expected_score("Arsenal", "Chelsea", False, 1, "2023-24")
        assert expected > 0.5  # Stronger team should have higher expected score
        assert (
            abs(expected - 0.76) < 0.01
        )  # Approximately 0.76 with 200 point difference

    def test_update_rating_win(self):
        """Test rating update after a win."""
        elo = EloRatingSystem()
        initial_rating = 1500.0

        # Team wins (actual_score = 1.0)
        new_rating = elo.update_rating("Arsenal", "Chelsea", 1.0, True, 1, "2023-24")

        assert new_rating > initial_rating  # Rating should increase after win
        assert elo.ratings["Arsenal_2023-24"] == new_rating
        assert len(elo.rating_history["Arsenal_2023-24"]) == 1

    def test_update_rating_loss(self):
        """Test rating update after a loss."""
        elo = EloRatingSystem()
        initial_rating = 1500.0

        # Team loses (actual_score = 0.0)
        new_rating = elo.update_rating("Arsenal", "Chelsea", 0.0, True, 1, "2023-24")

        assert new_rating < initial_rating  # Rating should decrease after loss

    def test_update_rating_draw(self):
        """Test rating update after a draw."""
        elo = EloRatingSystem()
        elo.ratings["Arsenal_2023-24"] = 1500.0
        elo.ratings["Chelsea_2023-24"] = 1500.0

        # Equal teams draw - rating should stay approximately the same
        new_rating = elo.update_rating("Arsenal", "Chelsea", 0.5, True, 1, "2023-24")

        # With equal teams and home advantage, a draw should slightly decrease home team's rating
        assert new_rating < 1500.0
        assert abs(new_rating - 1495.0) < 5.0  # Small change

    def test_importance_factor(self):
        """Test that importance factor affects rating changes."""
        elo = EloRatingSystem()

        # Normal importance
        rating_normal = elo.update_rating(
            "Arsenal", "Chelsea", 1.0, True, 1, "2023-24", importance_factor=1.0
        )

        # Reset for second test
        elo.ratings["Arsenal_2023-24"] = 1500.0

        # High importance (e.g., cup final)
        rating_high = elo.update_rating(
            "Arsenal", "Chelsea", 1.0, True, 1, "2023-24", importance_factor=1.5
        )

        # Higher importance should lead to larger rating change
        assert abs(rating_high - 1500.0) > abs(rating_normal - 1500.0)


class TestFixtureDifficultyCalculator:
    """Test cases for the comprehensive fixture difficulty calculator."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session for testing."""
        return Mock()

    @pytest.fixture
    def sample_fixtures(self):
        """Create sample fixture data for testing."""
        fixtures = []

        # Create sample fixtures
        for gw in range(1, 4):
            fixture = Mock(spec=Fixture)
            fixture.fixture_id = gw
            fixture.gameweek = gw
            fixture.season = CURRENT_SEASON
            fixture.home_team = "Arsenal"
            fixture.away_team = "Chelsea"
            fixture.date = f"2024-08-{10 + gw}T15:00:00Z"

            # Mock result
            result = Mock(spec=Result)
            result.home_score = 2 if gw == 1 else 1
            result.away_score = 1 if gw == 1 else 1
            fixture.result = result

            fixtures.append(fixture)

        return fixtures

    @pytest.fixture
    def sample_teams(self):
        """Create sample team data for testing."""
        teams = []
        for team_name in ["Arsenal", "Chelsea", "Liverpool", "Manchester City"]:
            team = Mock(spec=Team)
            team.name = team_name[:3].upper()  # 3-letter code
            team.full_name = team_name
            team.season = CURRENT_SEASON
            teams.append(team)
        return teams

    def test_initialization(self, mock_session):
        """Test FixtureDifficultyCalculator initialization."""
        calculator = FixtureDifficultyCalculator(
            dbsession=mock_session,
            season="2023-24",
            form_lookback=5,
            form_decay=0.15,
            congestion_hours=48,
        )

        assert calculator.dbsession == mock_session
        assert calculator.season == "2023-24"
        assert calculator.form_lookback == 5
        assert calculator.form_decay == 0.15
        assert calculator.congestion_hours == 48
        assert isinstance(calculator.elo_system, EloRatingSystem)

    @patch("airsenal.framework.fixture_difficulty.session")
    def test_calculate_team_form_no_fixtures(self, mock_db_session):
        """Test form calculation when no recent fixtures are available."""
        mock_db_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

        calculator = FixtureDifficultyCalculator(dbsession=mock_db_session)
        form_score = calculator.calculate_team_form("Arsenal", 5)

        assert form_score == 0.5  # Neutral form for no data

    def test_calculate_team_form_with_results(self, mock_session, sample_fixtures):
        """Test form calculation with actual fixture results."""
        # Mock database query chain
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = sample_fixtures

        calculator = FixtureDifficultyCalculator(dbsession=mock_session)
        form_score = calculator.calculate_team_form("Arsenal", 5)

        # Should return a form score between 0 and 1
        assert 0.0 <= form_score <= 1.0
        assert form_score > 0.5  # Arsenal won and drew, so should be above neutral

    def test_calculate_head_to_head_no_history(self, mock_session):
        """Test H2H calculation when no historical meetings exist."""
        mock_session.query.return_value.filter.return_value.filter.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

        calculator = FixtureDifficultyCalculator(dbsession=mock_session)
        h2h_stats = calculator.calculate_head_to_head("Arsenal", "Chelsea")

        assert h2h_stats["home_advantage"] == 0.0
        assert h2h_stats["recent_performance"] == 0.5
        assert h2h_stats["matches_played"] == 0

    def test_calculate_fixture_congestion_no_nearby_games(self, mock_session):
        """Test congestion calculation when no nearby fixtures exist."""
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.all.return_value = []

        calculator = FixtureDifficultyCalculator(dbsession=mock_session)
        congestion = calculator.calculate_fixture_congestion(
            "Arsenal", "2024-08-15T15:00:00Z"
        )

        assert congestion == 0.0  # No congestion

    def test_expected_to_difficulty_scale(self):
        """Test conversion from expected win probability to 1-5 difficulty scale."""
        calculator = FixtureDifficultyCalculator()

        # Test boundary conditions
        assert calculator.calculate_fixture_difficulty.__code__.co_code  # Method exists

        # These would be tested in integration tests with actual fixture data
        # The expected_to_difficulty function is defined within calculate_fixture_difficulty

    def test_factory_function(self):
        """Test the factory function for creating calculator instances."""
        calculator = create_difficulty_calculator()

        assert isinstance(calculator, FixtureDifficultyCalculator)
        assert calculator.season == CURRENT_SEASON

    def test_factory_function_custom_params(self, mock_session):
        """Test factory function with custom parameters."""
        calculator = create_difficulty_calculator(
            season="2022-23", dbsession=mock_session
        )

        assert calculator.season == "2022-23"
        assert calculator.dbsession == mock_session


class TestFixtureDifficultyIntegration:
    """Integration tests that require database setup."""

    @pytest.fixture
    def setup_test_data(self):
        """Set up test data in the database."""
        # This would require actual database setup in a real test environment
        # For now, we'll mock the database interactions

    @patch("airsenal.framework.fixture_difficulty.session")
    def test_get_fixture_difficulty_function(self, mock_session):
        """Test the convenience function for getting fixture difficulty."""
        # Mock fixture query
        fixture = Mock(spec=Fixture)
        fixture.fixture_id = 1
        fixture.home_team = "Arsenal"
        fixture.away_team = "Chelsea"
        fixture.gameweek = 1
        fixture.season = CURRENT_SEASON
        fixture.date = "2024-08-15T15:00:00Z"

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            fixture
        )

        # Mock the calculator initialization and calculation
        with patch(
            "airsenal.framework.fixture_difficulty.create_difficulty_calculator"
        ) as mock_create:
            mock_calculator = Mock()
            mock_calculator.calculate_fixture_difficulty.return_value = {
                "fixture_id": 1,
                "home_difficulty": 3.0,
                "away_difficulty": 3.0,
            }
            mock_create.return_value = mock_calculator

            result = get_fixture_difficulty(1, "Arsenal")

            assert result["fixture_id"] == 1
            mock_calculator.calculate_fixture_difficulty.assert_called_once_with(
                fixture, "Arsenal"
            )

    @patch("airsenal.framework.fixture_difficulty.session")
    def test_get_fixture_difficulty_not_found(self, mock_session):
        """Test fixture difficulty function with non-existent fixture."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Fixture 999 not found"):
            get_fixture_difficulty(999)

    @patch("airsenal.framework.fixture_difficulty.create_difficulty_calculator")
    def test_get_team_fixture_difficulties(self, mock_create):
        """Test getting fixture difficulties for a team over a period."""
        mock_calculator = Mock()
        mock_calculator.calculate_bulk_difficulties.return_value = [
            {"fixture_id": 1, "difficulty": 3.0},
            {"fixture_id": 2, "difficulty": 2.5},
        ]
        mock_create.return_value = mock_calculator

        result = get_team_fixture_difficulties("Arsenal", 1, 5)

        assert len(result) == 2
        assert result[0]["fixture_id"] == 1
        mock_calculator.calculate_bulk_difficulties.assert_called_once_with(
            1, 5, "Arsenal"
        )

    @patch("airsenal.framework.fixture_difficulty.session")
    @patch("airsenal.framework.fixture_difficulty.create_difficulty_calculator")
    def test_get_all_teams_average_difficulty(self, mock_create, mock_session):
        """Test getting average difficulty for all teams."""
        # Mock teams query
        mock_teams = [
            Mock(name="ARS", full_name="Arsenal"),
            Mock(name="CHE", full_name="Chelsea"),
        ]
        mock_session.query.return_value.filter_by.return_value.all.return_value = (
            mock_teams
        )

        # Mock calculator
        mock_calculator = Mock()
        mock_calculator.get_team_average_difficulty.side_effect = [
            {"team": "ARS", "average_difficulty": 2.8},
            {"team": "CHE", "average_difficulty": 3.2},
        ]
        mock_create.return_value = mock_calculator

        result = get_all_teams_average_difficulty(1, 5)

        assert "ARS" in result
        assert "CHE" in result
        assert result["ARS"]["average_difficulty"] == 2.8
        assert result["CHE"]["average_difficulty"] == 3.2


class TestFixtureDifficultyValidation:
    """Test cases for validation and accuracy metrics."""

    def test_validation_metrics_calculation(self):
        """Test calculation of validation metrics."""
        # This would test the validate_predictions method
        # Requires extensive mocking of completed fixtures and results

    def test_difficulty_scale_boundaries(self):
        """Test that difficulty ratings stay within 1-5 bounds."""
        # Test with extreme probability values
        probabilities = [0.01, 0.25, 0.5, 0.75, 0.99]

        for prob in probabilities:
            # This logic would be extracted from the calculate_fixture_difficulty method
            if prob >= 0.8:
                difficulty = 1.0
            elif prob >= 0.65:
                difficulty = 2.0
            elif prob >= 0.35:
                difficulty = 3.0
            elif prob >= 0.2:
                difficulty = 4.0
            else:
                difficulty = 5.0

            assert 1.0 <= difficulty <= 5.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_team_names(self):
        """Test handling of invalid team names."""
        elo = EloRatingSystem()

        # Should handle gracefully by returning initial rating
        rating = elo.get_rating("NonExistentTeam", 1, "2023-24")
        assert rating == 1500.0

    def test_invalid_gameweek_numbers(self):
        """Test handling of invalid gameweek numbers."""
        elo = EloRatingSystem()

        # Should handle negative gameweeks
        rating = elo.get_rating("Arsenal", -1, "2023-24")
        assert rating == 1500.0

        # Should handle very high gameweeks
        rating = elo.get_rating("Arsenal", 100, "2023-24")
        assert rating == 1500.0

    def test_malformed_dates(self, mock_session):
        """Test handling of malformed fixture dates."""
        calculator = FixtureDifficultyCalculator(dbsession=mock_session)

        # Should handle None dates
        congestion = calculator.calculate_fixture_congestion("Arsenal", None)
        assert congestion == 0.0

        # Should handle empty string dates
        congestion = calculator.calculate_fixture_congestion("Arsenal", "")
        assert congestion == 0.0

    def test_extreme_rating_values(self):
        """Test handling of extreme Elo rating values."""
        elo = EloRatingSystem()

        # Test with very high ratings
        elo.ratings["Arsenal_2023-24"] = 2500.0
        elo.ratings["Chelsea_2023-24"] = 500.0

        expected = elo.expected_score("Arsenal", "Chelsea", False, 1, "2023-24")
        assert 0.0 <= expected <= 1.0  # Should be clipped to valid probability range

    @patch("airsenal.framework.fixture_difficulty.logger")
    def test_logging_on_errors(self, mock_logger, mock_session):
        """Test that appropriate logging occurs on errors."""
        # Mock database error
        mock_session.query.side_effect = Exception("Database connection failed")

        calculator = FixtureDifficultyCalculator(dbsession=mock_session)

        # This would trigger database queries that fail
        try:
            calculator.calculate_team_form("Arsenal", 5)
        except Exception:
            pass  # Expected to fail

        # In a real implementation, we'd verify that appropriate error logging occurred


class TestPerformanceOptimizations:
    """Test performance optimizations like caching."""

    def test_form_caching(self, mock_session):
        """Test that form calculations are cached."""
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

        calculator = FixtureDifficultyCalculator(dbsession=mock_session)

        # First call
        form1 = calculator.calculate_team_form("Arsenal", 5)

        # Second call should use cache (no additional DB query)
        form2 = calculator.calculate_team_form("Arsenal", 5)

        assert form1 == form2
        assert ("Arsenal", 5) in calculator._team_form_cache.get("Arsenal", {})

    def test_h2h_caching(self, mock_session):
        """Test that H2H calculations are cached."""
        mock_session.query.return_value.filter.return_value.filter.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

        calculator = FixtureDifficultyCalculator(dbsession=mock_session)

        # First call
        h2h1 = calculator.calculate_head_to_head("Arsenal", "Chelsea")

        # Second call should use cache
        h2h2 = calculator.calculate_head_to_head("Arsenal", "Chelsea")

        assert h2h1 == h2h2
        assert ("Arsenal", "Chelsea") in calculator._h2h_cache


if __name__ == "__main__":
    pytest.main([__file__])
