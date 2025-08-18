"""
Test Suite for Home/Away Adjustment System

This test suite comprehensively validates the home/away adjustment system including:
- Team home advantage calculations with Bayesian shrinkage
- Player-specific home/away performance adjustments
- Statistical significance testing
- Database model integrity
- Integration with prediction systems
- Validation framework accuracy

The tests are organized into:
1. Unit tests for core calculation methods
2. Database model and persistence tests
3. Integration tests with existing systems
4. Statistical validation and accuracy tests
5. Error handling and edge cases
"""

import datetime
import unittest
from unittest.mock import patch

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.home_away_adjustment import (
    HomeAwayAdjuster,
    PlayerHomeAwayPerformance,
    TeamHomeAdvantage,
    apply_home_away_adjustment,
    create_home_away_adjuster,
    get_player_home_away_adjustment,
    get_team_home_advantage,
)
from airsenal.framework.schema import (
    Base,
    Fixture,
    HomeAwayAdjustmentHistory,
    Player,
    PlayerHomeAwayData,
    PlayerScore,
    Result,
    Team,
    TeamHomeAdvantageData,
    VenuePerformanceAnalysis,
)


class TestTeamHomeAdvantageCalculation(unittest.TestCase):
    """Test team home advantage calculation with Bayesian shrinkage."""

    def setUp(self):
        """Set up test database and sample data."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        # Create sample teams
        self.teams = ["ARS", "CHE", "LIV", "MCI"]
        for i, team_name in enumerate(self.teams):
            team = Team(
                id=i + 1,
                name=team_name,
                full_name=f"{team_name} FC",
                season="2425",
                team_id=i + 1,
            )
            self.session.add(team)

        # Create sample fixtures and results for home advantage testing
        self._create_sample_matches()
        self.session.commit()

        self.calculator = TeamHomeAdvantage(self.session)

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def _create_sample_matches(self):
        """Create sample match data with known home advantage patterns."""
        fixtures_data = [
            # ARS has strong home advantage (wins most home games, struggles away)
            ("ARS", "CHE", 1, 2, 1, "2425"),  # Home win
            ("ARS", "LIV", 2, 3, 0, "2425"),  # Home win
            ("ARS", "MCI", 3, 1, 1, "2425"),  # Home draw
            ("CHE", "ARS", 4, 0, 2, "2425"),  # Away loss
            ("LIV", "ARS", 5, 1, 2, "2425"),  # Away loss
            # CHE has moderate home advantage
            ("CHE", "LIV", 6, 2, 1, "2425"),  # Home win
            ("CHE", "MCI", 7, 1, 1, "2425"),  # Home draw
            ("LIV", "CHE", 8, 1, 1, "2425"),  # Away draw
            ("MCI", "CHE", 9, 2, 1, "2425"),  # Away win
            # LIV has minimal home advantage
            ("LIV", "MCI", 10, 1, 1, "2425"),  # Home draw
            ("MCI", "LIV", 11, 1, 2, "2425"),  # Away win
        ]

        for home_team, away_team, gw, home_score, away_score, season in fixtures_data:
            fixture = Fixture(
                home_team=home_team,
                away_team=away_team,
                season=season,
                gameweek=gw,
                date=f"2024-08-{gw + 10}",
                tag="test",
            )
            self.session.add(fixture)
            self.session.flush()

            result = Result(
                fixture_id=fixture.fixture_id,
                home_score=home_score,
                away_score=away_score,
            )
            self.session.add(result)

    def test_league_baseline_calculation(self):
        """Test calculation of league-wide baseline home advantage."""
        baseline = self.calculator.calculate_league_baseline_advantage("2425")

        # Should be positive (home teams score more on average)
        assert baseline > 0

        # Should be reasonable value (typically 0.2-0.5 goals per game)
        assert baseline < 1.0

        # Test caching
        baseline2 = self.calculator.calculate_league_baseline_advantage("2425")
        assert baseline == baseline2

    def test_team_home_advantage_calculation(self):
        """Test team-specific home advantage calculation."""
        # Test ARS (should have strong home advantage)
        ars_data = self.calculator.calculate_team_home_advantage("ARS", "2425")

        assert isinstance(ars_data, dict)
        assert ars_data["team"] == "ARS"
        assert "raw_home_advantage" in ars_data
        assert "adjusted_home_advantage" in ars_data
        assert "shrinkage_factor" in ars_data
        assert "confidence" in ars_data

        # ARS should have positive home advantage
        assert ars_data["adjusted_home_advantage"] > 0

        # Shrinkage factor should be between 0 and 1
        assert ars_data["shrinkage_factor"] >= 0
        assert ars_data["shrinkage_factor"] <= 1

    def test_bayesian_shrinkage_effect(self):
        """Test that Bayesian shrinkage appropriately moderates extreme values."""
        # Team with very few games should be heavily shrunk toward baseline
        few_games_data = self.calculator.calculate_team_home_advantage("MCI", "2425")

        # Should have low confidence due to small sample
        assert few_games_data["confidence"] < 0.7

        # Adjusted advantage should be closer to baseline than raw advantage
        raw_adv = few_games_data["raw_home_advantage"]
        adj_adv = few_games_data["adjusted_home_advantage"]
        baseline = few_games_data["league_baseline"]

        # If raw is very different from baseline, adjusted should be between them
        if abs(raw_adv - baseline) > 0.1:
            assert abs(adj_adv - baseline) < abs(raw_adv - baseline)

    def test_stadium_factor_integration(self):
        """Test stadium capacity factor integration."""
        # Should apply stadium factors
        ars_data = self.calculator.calculate_team_home_advantage("ARS", "2425")
        assert "stadium_factor" in ars_data
        assert ars_data["stadium_factor"] > 0.8
        assert ars_data["stadium_factor"] < 1.2

    def test_empty_stadium_adjustment(self):
        """Test COVID-era empty stadium adjustments."""
        # Test COVID season (2021)
        calculator_covid = TeamHomeAdvantage(self.session)
        # Mock COVID season data
        with patch.object(
            calculator_covid, "_get_empty_stadium_factor", return_value=0.8
        ):
            covid_data = calculator_covid.calculate_team_home_advantage("ARS", "2021")
            assert "empty_stadium_adjustment" in covid_data
            assert covid_data["empty_stadium_adjustment"] == 0.8

    def test_insufficient_data_handling(self):
        """Test handling of teams with insufficient data."""
        # Test team with no matches
        no_data = self.calculator.calculate_team_home_advantage("NEW", "2425")

        assert no_data["confidence"] == 0.0
        assert no_data["home_matches"] == 0
        assert no_data["away_matches"] == 0

        # Should fall back to league baseline
        baseline = self.calculator.calculate_league_baseline_advantage("2425")
        self.assertAlmostEqual(no_data["adjusted_home_advantage"], baseline, places=2)


class TestPlayerHomeAwayPerformance(unittest.TestCase):
    """Test player-specific home/away performance calculations."""

    def setUp(self):
        """Set up test database and sample data."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        # Create sample players
        self.players = []
        for i in range(3):
            player = Player(player_id=i + 1, name=f"Player_{i + 1}", fpl_api_id=i + 100)
            self.session.add(player)
            self.players.append(player)

        # Create sample fixtures and player scores
        self._create_sample_player_data()
        self.session.commit()

        self.calculator = PlayerHomeAwayPerformance(
            self.session, min_games_threshold=10
        )

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def _create_sample_player_data(self):
        """Create sample player performance data with home/away differences."""
        # Player 1: Strong home advantage (better at home)
        # Player 2: Away specialist (better away)
        # Player 3: No significant difference

        performance_patterns = [
            # Player 1: Home specialist (6 pts home, 3 pts away average)
            [
                (6, 7, 8, 5, 4, 6, 7, 8, 6, 5, 7, 6, 8, 5, 6),
                True,
            ],  # Home games (15 games)
            [
                (3, 2, 4, 3, 4, 2, 3, 4, 2, 3, 4, 3, 2, 4, 3),
                False,
            ],  # Away games (15 games)
            # Player 2: Away specialist (4 pts home, 7 pts away average)
            [(4, 3, 5, 4, 3, 4, 5, 3, 4, 5, 4, 3, 5, 4, 3), True],  # Home games
            [(7, 8, 6, 7, 8, 7, 6, 8, 7, 6, 8, 7, 6, 8, 7), False],  # Away games
            # Player 3: No difference (5 pts both home and away)
            [(5, 4, 6, 5, 4, 5, 6, 4, 5, 6, 5, 4, 6, 5, 4), True],  # Home games
            [(5, 6, 4, 5, 6, 5, 4, 6, 5, 4, 6, 5, 4, 6, 5), False],  # Away games
        ]

        score_id = 1
        fixture_id = 1

        for player_idx, player in enumerate(self.players):
            home_pattern, _ = performance_patterns[player_idx * 2]
            away_pattern, _ = performance_patterns[player_idx * 2 + 1]

            # Create home games
            for i, points in enumerate(home_pattern):
                fixture = Fixture(
                    fixture_id=fixture_id,
                    home_team="ARS",
                    away_team="CHE",
                    season="2425",
                    gameweek=i + 1,
                    date=f"2024-08-{i + 1:02d}",
                    tag="test",
                )
                self.session.add(fixture)

                result = Result(
                    result_id=score_id,
                    fixture_id=fixture_id,
                    home_score=2,
                    away_score=1,
                )
                self.session.add(result)

                score = PlayerScore(
                    id=score_id,
                    player_id=player.player_id,
                    player_team="ARS",
                    opponent="CHE",
                    points=points,
                    goals=1 if points > 6 else 0,
                    assists=1 if points > 5 else 0,
                    minutes=90,
                    bonus=0,
                    conceded=1,
                    fixture_id=fixture_id,
                    result_id=score_id,
                )
                self.session.add(score)

                score_id += 1
                fixture_id += 1

            # Create away games
            for i, points in enumerate(away_pattern):
                fixture = Fixture(
                    fixture_id=fixture_id,
                    home_team="CHE",
                    away_team="ARS",
                    season="2425",
                    gameweek=i + 16,
                    date=f"2024-09-{i + 1:02d}",
                    tag="test",
                )
                self.session.add(fixture)

                result = Result(
                    result_id=score_id,
                    fixture_id=fixture_id,
                    home_score=1,
                    away_score=2,
                )
                self.session.add(result)

                score = PlayerScore(
                    id=score_id,
                    player_id=player.player_id,
                    player_team="ARS",
                    opponent="CHE",
                    points=points,
                    goals=1 if points > 6 else 0,
                    assists=1 if points > 5 else 0,
                    minutes=90,
                    bonus=0,
                    conceded=1,
                    fixture_id=fixture_id,
                    result_id=score_id,
                )
                self.session.add(score)

                score_id += 1
                fixture_id += 1

    def test_player_adjustment_calculation(self):
        """Test calculation of player-specific home/away adjustments."""
        # Test Player 1 (home specialist)
        player1_data = self.calculator.calculate_player_home_away_adjustment(1, "2425")

        assert isinstance(player1_data, dict)
        assert player1_data["player_id"] == 1
        assert player1_data["total_games"] > 20

        # Should have positive home adjustment, negative away adjustment
        assert player1_data["home_adjustment"] > 0
        assert player1_data["away_adjustment"] < 0

        # Should be statistically significant
        assert player1_data["is_significant"]
        assert player1_data["p_value"] < 0.05

    def test_away_specialist_detection(self):
        """Test detection of players who perform better away."""
        # Test Player 2 (away specialist)
        player2_data = self.calculator.calculate_player_home_away_adjustment(2, "2425")

        # Should have negative home adjustment, positive away adjustment
        assert player2_data["home_adjustment"] < 0
        assert player2_data["away_adjustment"] > 0

        # Should be statistically significant
        assert player2_data["is_significant"]

    def test_no_difference_detection(self):
        """Test handling of players with no significant home/away difference."""
        # Test Player 3 (no difference)
        player3_data = self.calculator.calculate_player_home_away_adjustment(3, "2425")

        # Should have minimal adjustments
        assert abs(player3_data["home_adjustment"]) < 0.1
        assert abs(player3_data["away_adjustment"]) < 0.1

        # Should not be statistically significant
        assert not player3_data["is_significant"]

    def test_insufficient_games_handling(self):
        """Test handling of players with insufficient games."""
        # Test with very high threshold
        calculator_strict = PlayerHomeAwayPerformance(
            self.session, min_games_threshold=100
        )

        insufficient_data = calculator_strict.calculate_player_home_away_adjustment(
            1, "2425"
        )

        assert insufficient_data["home_adjustment"] == 0.0
        assert insufficient_data["away_adjustment"] == 0.0
        assert not insufficient_data["is_significant"]
        assert insufficient_data["confidence"] == 0.0

    def test_nonexistent_player_handling(self):
        """Test handling of nonexistent players."""
        nonexistent_data = self.calculator.calculate_player_home_away_adjustment(
            999, "2425"
        )

        assert nonexistent_data["player_id"] is None
        assert nonexistent_data["home_adjustment"] == 0.0
        assert nonexistent_data["away_adjustment"] == 0.0


class TestHomeAwayAdjuster(unittest.TestCase):
    """Test the main HomeAwayAdjuster class integration."""

    def setUp(self):
        """Set up test database and sample data."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        # Create minimal test data
        self._create_test_data()
        self.session.commit()

        self.adjuster = HomeAwayAdjuster(self.session)

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def _create_test_data(self):
        """Create minimal test data for integration testing."""
        # Create a team
        team = Team(id=1, name="ARS", full_name="Arsenal FC", season="2425", team_id=1)
        self.session.add(team)

        # Create a player
        player = Player(player_id=1, name="Test Player", fpl_api_id=100)
        self.session.add(player)

        # Create some fixture/result data for calculations
        for i in range(5):
            fixture = Fixture(
                fixture_id=i + 1,
                home_team="ARS" if i % 2 == 0 else "CHE",
                away_team="CHE" if i % 2 == 0 else "ARS",
                season="2425",
                gameweek=i + 1,
                date=f"2024-08-{i + 10:02d}",
                tag="test",
            )
            self.session.add(fixture)

            result = Result(
                result_id=i + 1, fixture_id=i + 1, home_score=2, away_score=1
            )
            self.session.add(result)

    def test_team_adjustment_application(self):
        """Test application of team-level adjustments."""
        base_prediction = 5.0

        # Test home adjustment
        home_adjusted = self.adjuster.apply_team_adjustment(
            base_prediction, "ARS", True, "2425", 5
        )

        # Home adjustment should modify the prediction
        assert home_adjusted != base_prediction
        assert home_adjusted > 0

        # Test away adjustment
        away_adjusted = self.adjuster.apply_team_adjustment(
            base_prediction, "ARS", False, "2425", 5
        )

        # Away adjustment should be different from home
        assert away_adjusted != home_adjusted

    def test_player_adjustment_application(self):
        """Test application of player-level adjustments."""
        base_prediction = 5.0

        # Test player adjustment (will likely return base prediction due to insufficient data)
        adjusted = self.adjuster.apply_player_adjustment(
            base_prediction, 1, True, "2425"
        )

        # Should return a valid number
        assert isinstance(adjusted, float)
        assert adjusted > 0

    def test_combined_adjustment(self):
        """Test combined team and player adjustments."""
        base_prediction = 5.0

        result = self.adjuster.apply_combined_adjustment(
            base_prediction, "ARS", 1, True, "2425", 5
        )

        assert isinstance(result, dict)
        assert "base_prediction" in result
        assert "team_adjusted" in result
        assert "player_adjusted" in result
        assert "final_prediction" in result
        assert "total_impact" in result

        # Final prediction should be positive
        assert result["final_prediction"] > 0

        # Should have reasonable bounds (0.5x to 2x base prediction)
        assert result["final_prediction"] >= base_prediction * 0.5
        assert result["final_prediction"] <= base_prediction * 2.0

    def test_venue_multipliers(self):
        """Test venue multiplier calculations."""
        multipliers = self.adjuster.get_venue_multipliers("ARS", "2425", 5)

        assert isinstance(multipliers, dict)
        assert "home_multiplier" in multipliers
        assert "away_multiplier" in multipliers
        assert "advantage" in multipliers
        assert "confidence" in multipliers

        # Multipliers should be positive and reasonable
        assert multipliers["home_multiplier"] > 0.5
        assert multipliers["home_multiplier"] < 1.5
        assert multipliers["away_multiplier"] > 0.5
        assert multipliers["away_multiplier"] < 1.5

    def test_adjustment_caching(self):
        """Test that adjustments are properly cached."""
        base_prediction = 5.0

        # First call
        result1 = self.adjuster.apply_combined_adjustment(
            base_prediction, "ARS", 1, True, "2425", 5
        )

        # Second call with same parameters
        result2 = self.adjuster.apply_combined_adjustment(
            base_prediction, "ARS", 1, True, "2425", 5
        )

        # Results should be identical (cached)
        assert result1["final_prediction"] == result2["final_prediction"]


class TestDatabaseModels(unittest.TestCase):
    """Test database models for home/away adjustment data."""

    def setUp(self):
        """Set up test database."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def test_team_home_advantage_data_model(self):
        """Test TeamHomeAdvantageData model creation and constraints."""
        advantage_data = TeamHomeAdvantageData(
            team="ARS",
            season="2425",
            gameweek=5,
            raw_home_advantage=0.4,
            adjusted_home_advantage=0.3,
            league_baseline=0.37,
            shrinkage_factor=0.6,
            confidence=0.75,
            home_matches=10,
            away_matches=8,
            home_goals_per_game=1.8,
            away_goals_per_game=1.2,
            home_goals_against_per_game=0.9,
            away_goals_against_per_game=1.3,
            stadium_factor=1.1,
            empty_stadium_adjustment=1.0,
            calculated_at=datetime.datetime.now().isoformat(),
        )

        self.session.add(advantage_data)
        self.session.commit()

        # Retrieve and verify
        retrieved = self.session.query(TeamHomeAdvantageData).first()
        assert retrieved is not None
        assert retrieved.team == "ARS"
        assert retrieved.season == "2425"
        self.assertAlmostEqual(retrieved.adjusted_home_advantage, 0.3)

    def test_player_home_away_data_model(self):
        """Test PlayerHomeAwayData model creation."""
        # Create a player first
        player = Player(player_id=1, name="Test Player", fpl_api_id=100)
        self.session.add(player)
        self.session.commit()

        player_data = PlayerHomeAwayData(
            player_id=1,
            season="2425",
            home_adjustment=0.15,
            away_adjustment=-0.15,
            is_significant=True,
            p_value=0.02,
            confidence=0.8,
            home_games=15,
            away_games=15,
            total_games=30,
            home_points_per_game=6.2,
            away_points_per_game=4.8,
            analyzed_at=datetime.datetime.now().isoformat(),
        )

        self.session.add(player_data)
        self.session.commit()

        # Retrieve and verify
        retrieved = self.session.query(PlayerHomeAwayData).first()
        assert retrieved is not None
        assert retrieved.player_id == 1
        assert retrieved.is_significant
        self.assertAlmostEqual(retrieved.home_adjustment, 0.15)

    def test_home_away_adjustment_history_model(self):
        """Test HomeAwayAdjustmentHistory model creation."""
        history = HomeAwayAdjustmentHistory(
            entity_type="player",
            entity_id=1,
            entity_name="Test Player",
            season="2425",
            gameweek=5,
            home_adjustment=0.1,
            away_adjustment=-0.1,
            prediction_type="points",
            base_prediction=5.0,
            adjusted_prediction=5.5,
            adjustment_impact=0.5,
            adjustment_method="combined",
            adjustment_version="1.0.0",
            applied_at=datetime.datetime.now().isoformat(),
        )

        self.session.add(history)
        self.session.commit()

        # Retrieve and verify
        retrieved = self.session.query(HomeAwayAdjustmentHistory).first()
        assert retrieved is not None
        assert retrieved.entity_type == "player"
        assert retrieved.adjustment_impact == 0.5

    def test_venue_performance_analysis_model(self):
        """Test VenuePerformanceAnalysis model creation."""
        performance = VenuePerformanceAnalysis(
            entity_type="player",
            entity_id=1,
            entity_name="Test Player",
            season="2425",
            venue="home",
            games_played=15,
            total_points=90,
            points_per_game=6.0,
            goals_scored=8,
            assists=4,
            recent_form=0.7,
            trend=0.1,
            performance_variance=2.5,
            consistency_score=0.75,
            analysis_period_start="2024-08-01",
            analysis_period_end="2024-12-01",
            analyzed_at=datetime.datetime.now().isoformat(),
        )

        self.session.add(performance)
        self.session.commit()

        # Retrieve and verify
        retrieved = self.session.query(VenuePerformanceAnalysis).first()
        assert retrieved is not None
        assert retrieved.venue == "home"
        assert retrieved.points_per_game == 6.0


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions and API integration."""

    def setUp(self):
        """Set up test database with minimal data."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        # Create minimal test data
        team = Team(id=1, name="ARS", full_name="Arsenal FC", season="2425", team_id=1)
        self.session.add(team)

        player = Player(player_id=1, name="Test Player", fpl_api_id=100)
        self.session.add(player)

        self.session.commit()

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def test_get_team_home_advantage_function(self):
        """Test get_team_home_advantage convenience function."""
        with patch("airsenal.framework.home_away_adjustment.session", self.session):
            result = get_team_home_advantage("ARS", "2425", 5)

            assert isinstance(result, dict)
            assert "team" in result
            assert result["team"] == "ARS"

    def test_get_player_home_away_adjustment_function(self):
        """Test get_player_home_away_adjustment convenience function."""
        with patch("airsenal.framework.home_away_adjustment.session", self.session):
            result = get_player_home_away_adjustment(1, "2425")

            assert isinstance(result, dict)
            assert "player_id" in result

    def test_apply_home_away_adjustment_function(self):
        """Test apply_home_away_adjustment convenience function."""
        with patch("airsenal.framework.home_away_adjustment.session", self.session):
            with patch(
                "airsenal.framework.home_away_adjustment.get_last_finished_gameweek",
                return_value=5,
            ):
                result = apply_home_away_adjustment(5.0, "ARS", 1, True, "2425")

                assert isinstance(result, float)
                assert result > 0

    def test_create_home_away_adjuster_function(self):
        """Test create_home_away_adjuster factory function."""
        adjuster = create_home_away_adjuster(self.session)

        assert isinstance(adjuster, HomeAwayAdjuster)
        assert adjuster.dbsession == self.session


class TestStatisticalValidation(unittest.TestCase):
    """Test statistical validation and accuracy of the adjustment system."""

    def setUp(self):
        """Set up test with known statistical patterns."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        # Create test data with known statistical properties
        self._create_statistical_test_data()
        self.session.commit()

        self.adjuster = HomeAwayAdjuster(self.session)

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def _create_statistical_test_data(self):
        """Create test data with known statistical properties for validation."""
        # Create teams with different home advantage patterns
        teams = ["STRONG_HOME", "WEAK_HOME", "NEUTRAL"]
        for i, team_name in enumerate(teams):
            team = Team(
                id=i + 1,
                name=team_name,
                full_name=f"{team_name} FC",
                season="2425",
                team_id=i + 1,
            )
            self.session.add(team)

        # Create fixtures with controlled outcomes
        # STRONG_HOME: wins 80% at home, 30% away
        # WEAK_HOME: wins 60% at home, 50% away
        # NEUTRAL: wins 55% at home, 45% away

        patterns = {
            "STRONG_HOME": [(0.8, 0.3), (2.2, 1.1)],  # (win_rate, avg_goals)
            "WEAK_HOME": [(0.6, 0.5), (1.6, 1.4)],
            "NEUTRAL": [(0.55, 0.45), (1.55, 1.45)],
        }

        fixture_id = 1
        for team_name, (
            (home_win_rate, away_win_rate),
            (home_goals, away_goals),
        ) in patterns.items():
            # Create 20 home games and 20 away games for each team
            for is_home in [True, False]:
                win_rate = home_win_rate if is_home else away_win_rate
                avg_goals = home_goals if is_home else away_goals

                for game in range(20):
                    # Simulate match outcome
                    home_team = team_name if is_home else "OPP"
                    away_team = "OPP" if is_home else team_name

                    # Simple outcome simulation
                    if np.random.random() < win_rate:
                        # Win
                        if is_home:
                            home_score = int(avg_goals + np.random.normal(0, 0.5))
                            away_score = max(
                                0, int(home_score - 1 - np.random.exponential(0.5))
                            )
                        else:
                            away_score = int(avg_goals + np.random.normal(0, 0.5))
                            home_score = max(
                                0, int(away_score - 1 - np.random.exponential(0.5))
                            )
                    else:
                        # Loss or draw
                        if is_home:
                            home_score = max(
                                0, int(avg_goals - 0.5 + np.random.normal(0, 0.5))
                            )
                            away_score = int(home_score + np.random.exponential(0.5))
                        else:
                            away_score = max(
                                0, int(avg_goals - 0.5 + np.random.normal(0, 0.5))
                            )
                            home_score = int(away_score + np.random.exponential(0.5))

                    fixture = Fixture(
                        fixture_id=fixture_id,
                        home_team=home_team,
                        away_team=away_team,
                        season="2425",
                        gameweek=game + 1,
                        date=f"2024-08-{(game % 28) + 1:02d}",
                        tag="test",
                    )
                    self.session.add(fixture)

                    result = Result(
                        result_id=fixture_id,
                        fixture_id=fixture_id,
                        home_score=home_score,
                        away_score=away_score,
                    )
                    self.session.add(result)

                    fixture_id += 1

    def test_advantage_ranking_accuracy(self):
        """Test that teams are correctly ranked by home advantage."""
        # Calculate home advantages
        strong_adv = self.adjuster.team_advantage.calculate_team_home_advantage(
            "STRONG_HOME", "2425"
        )
        weak_adv = self.adjuster.team_advantage.calculate_team_home_advantage(
            "WEAK_HOME", "2425"
        )
        neutral_adv = self.adjuster.team_advantage.calculate_team_home_advantage(
            "NEUTRAL", "2425"
        )

        # Verify ranking order
        assert (
            strong_adv["adjusted_home_advantage"]
            > neutral_adv["adjusted_home_advantage"]
        )
        assert (
            neutral_adv["adjusted_home_advantage"] > weak_adv["adjusted_home_advantage"]
        )

    def test_confidence_correlation_with_sample_size(self):
        """Test that confidence correlates with sample size."""
        # Teams with 40 total games should have high confidence
        strong_adv = self.adjuster.team_advantage.calculate_team_home_advantage(
            "STRONG_HOME", "2425"
        )

        # Should have reasonable confidence with 40 games
        assert strong_adv["confidence"] > 0.5

        # Test with limited data (mock smaller sample)
        with patch.object(
            self.adjuster.team_advantage, "_get_team_venue_stats"
        ) as mock_stats:
            mock_stats.return_value = {
                "matches": 2,
                "goals_per_game": 1.5,
                "goals_against_per_game": 1.0,
            }

            small_sample = self.adjuster.team_advantage.calculate_team_home_advantage(
                "TEST", "2425"
            )
            assert small_sample["confidence"] < strong_adv["confidence"]

    def test_adjustment_validation(self):
        """Test validation framework for adjustments."""
        validation_results = self.adjuster.validate_adjustments("2425", min_gameweek=1)

        assert isinstance(validation_results, dict)
        assert "correlation" in validation_results
        assert "mae" in validation_results
        assert "rmse" in validation_results

        # Should have processed some matches
        assert validation_results["matches_analyzed"] > 0


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up empty test database."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        self.adjuster = HomeAwayAdjuster(self.session)

    def tearDown(self):
        """Clean up test database."""
        self.session.close()

    def test_empty_database_handling(self):
        """Test handling of empty database."""
        # Should not crash with empty database
        result = self.adjuster.team_advantage.calculate_team_home_advantage(
            "ARS", "2425"
        )

        assert isinstance(result, dict)
        assert result["confidence"] == 0.0

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Invalid player ID
        result = self.adjuster.player_performance.calculate_player_home_away_adjustment(
            -1, "2425"
        )
        assert result["player_id"] is None

        # Invalid season format - should still work
        result = self.adjuster.team_advantage.calculate_team_home_advantage(
            "ARS", "invalid"
        )
        assert isinstance(result, dict)

    def test_extreme_adjustment_bounds(self):
        """Test that extreme adjustments are properly bounded."""
        # Test with extreme base prediction
        extreme_prediction = 100.0

        result = self.adjuster.apply_combined_adjustment(
            extreme_prediction, "ARS", 1, True, "2425", 5
        )

        # Should be bounded to reasonable range
        assert result["final_prediction"] <= extreme_prediction * 2.0
        assert result["final_prediction"] >= extreme_prediction * 0.5


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Run tests
    unittest.main(verbosity=2)
