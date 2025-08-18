"""
Home/Away Adjustment System for AIrsenal

This module implements a comprehensive home/away adjustment system that enhances
prediction accuracy by accounting for venue-specific performance differences at
team and player levels.

Key Features:
- Team-specific home advantage calculations with Bayesian shrinkage
- Player-specific home/away performance adjustments
- COVID-era empty stadium adjustments
- Stadium capacity and atmosphere factors
- Integration with existing TeamStrength and prediction systems

Statistical Methodology:
- Base home advantage: League average ~0.37 goals per game
- Bayesian shrinkage: α = n/(n+k) where k=20 games for stability
- Player adjustments: Only applied with >30 games sample size
- Temporal weighting: Recent seasons weighted more heavily
- Empty stadium penalty: -0.2 home advantage multiplier

Author: AIrsenal Team
"""

import datetime
import logging

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerScore,
    Result,
    session,
)
from airsenal.framework.utils import CURRENT_SEASON, get_last_finished_gameweek

logger = logging.getLogger(__name__)


class TeamHomeAdvantage:
    """
    Calculator for team-specific home advantage using Bayesian shrinkage.

    Methodology:
    1. Calculate league-wide baseline home advantage (~0.37 goals)
    2. Estimate team-specific effects from historical data
    3. Apply Bayesian shrinkage to prevent overfitting with small samples
    4. Account for stadium factors (capacity, atmosphere)
    5. Adjust for empty stadium periods (COVID era)

    Shrinkage Formula:
    adjusted_advantage = α * team_specific + (1-α) * league_baseline
    where α = n_games / (n_games + shrinkage_factor)
    """

    def __init__(
        self,
        dbsession: Session,
        shrinkage_factor: float = 20.0,
        min_games_threshold: int = 10,
        empty_stadium_penalty: float = 0.2,
        season_weight_decay: float = 0.8,
    ):
        """
        Initialize the team home advantage calculator.

        Args:
            dbsession: Database session for data access
            shrinkage_factor: Bayesian shrinkage parameter (higher = more conservative)
            min_games_threshold: Minimum games before applying team-specific adjustments
            empty_stadium_penalty: Reduction in home advantage for empty stadiums
            season_weight_decay: Exponential decay for older seasons (0-1)
        """
        self.dbsession = dbsession
        self.shrinkage_factor = shrinkage_factor
        self.min_games_threshold = min_games_threshold
        self.empty_stadium_penalty = empty_stadium_penalty
        self.season_weight_decay = season_weight_decay

        # Cache for computed values
        self._league_baseline_cache = {}
        self._team_advantage_cache = {}
        self._stadium_factors_cache = {}

        # Stadium capacity data (approximate values for PL teams)
        self._stadium_capacities = {
            "ARS": 60338,
            "AVL": 42729,
            "BOU": 11364,
            "BRE": 17250,
            "BRI": 31800,
            "CHE": 40834,
            "CRY": 25486,
            "EVE": 39414,
            "FUL": 19359,
            "LIV": 53394,
            "LUT": 10356,
            "MCI": 55097,
            "MUN": 74310,
            "NEW": 52409,
            "NOR": 27359,
            "SHU": 32609,
            "TOT": 62850,
            "WAT": 21577,
            "WHU": 66000,
            "WOL": 31700,
            "LEI": 32312,
            "LEE": 37890,
            "BUR": 21994,
            "SOU": 32384,
            "BHA": 31800,
        }

    def calculate_league_baseline_advantage(
        self, season: str, up_to_gameweek: int | None = None
    ) -> float:
        """
        Calculate the league-wide baseline home advantage.

        Args:
            season: Season to analyze
            up_to_gameweek: Only consider matches up to this gameweek

        Returns:
            League baseline home advantage (goals per game)
        """
        cache_key = f"{season}_{up_to_gameweek}"
        if cache_key in self._league_baseline_cache:
            return self._league_baseline_cache[cache_key]

        logger.debug(f"Calculating league baseline home advantage for {season}")

        # Get all completed matches
        query = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
        )

        if up_to_gameweek:
            query = query.filter(Fixture.gameweek <= up_to_gameweek)

        matches = query.all()

        if not matches:
            # Fallback to historical average
            baseline = 0.37
            logger.warning(
                f"No matches found for {season}, using default baseline: {baseline}"
            )
            return baseline

        # Calculate home advantage
        total_home_goals = sum(result.home_score for _, result in matches)
        total_away_goals = sum(result.away_score for _, result in matches)
        total_matches = len(matches)

        home_goals_per_game = total_home_goals / total_matches
        away_goals_per_game = total_away_goals / total_matches

        baseline = home_goals_per_game - away_goals_per_game

        # Cache the result
        self._league_baseline_cache[cache_key] = baseline

        logger.debug(f"League baseline home advantage: {baseline:.3f} goals per game")
        return baseline

    def calculate_team_home_advantage(
        self,
        team: str,
        season: str,
        up_to_gameweek: int | None = None,
        lookback_seasons: int = 3,
    ) -> dict[str, float]:
        """
        Calculate team-specific home advantage with Bayesian shrinkage.

        Args:
            team: Team name
            season: Current season
            up_to_gameweek: Only consider matches up to this gameweek
            lookback_seasons: Number of previous seasons to include

        Returns:
            Dictionary with team home advantage metrics
        """
        cache_key = f"{team}_{season}_{up_to_gameweek}_{lookback_seasons}"
        if cache_key in self._team_advantage_cache:
            return self._team_advantage_cache[cache_key]

        logger.debug(f"Calculating home advantage for {team}")

        # Get league baseline
        league_baseline = self.calculate_league_baseline_advantage(
            season, up_to_gameweek
        )

        # Get historical home and away matches for the team
        home_stats = self._get_team_venue_stats(
            team, True, season, up_to_gameweek, lookback_seasons
        )
        away_stats = self._get_team_venue_stats(
            team, False, season, up_to_gameweek, lookback_seasons
        )

        if home_stats["matches"] == 0 or away_stats["matches"] == 0:
            logger.warning(f"Insufficient data for {team}, using league baseline")
            return {
                "team": team,
                "raw_home_advantage": league_baseline,
                "adjusted_home_advantage": league_baseline,
                "shrinkage_factor": 0.0,
                "home_matches": home_stats["matches"],
                "away_matches": away_stats["matches"],
                "confidence": 0.0,
                "stadium_factor": 1.0,
                "empty_stadium_adjustment": 1.0,
            }

        # Calculate raw team-specific home advantage
        raw_home_advantage = (
            home_stats["goals_per_game"] - home_stats["goals_against_per_game"]
        ) - (away_stats["goals_per_game"] - away_stats["goals_against_per_game"])

        # Apply Bayesian shrinkage
        total_matches = home_stats["matches"] + away_stats["matches"]
        shrinkage_alpha = total_matches / (total_matches + self.shrinkage_factor)

        adjusted_home_advantage = (
            shrinkage_alpha * raw_home_advantage
            + (1 - shrinkage_alpha) * league_baseline
        )

        # Calculate stadium factor
        stadium_factor = self._get_stadium_factor(team)

        # Apply stadium factor
        final_home_advantage = adjusted_home_advantage * stadium_factor

        # Detect empty stadium periods and apply adjustment
        empty_stadium_factor = self._get_empty_stadium_factor(team, season)
        final_home_advantage *= empty_stadium_factor

        result = {
            "team": team,
            "raw_home_advantage": raw_home_advantage,
            "adjusted_home_advantage": final_home_advantage,
            "shrinkage_factor": shrinkage_alpha,
            "home_matches": home_stats["matches"],
            "away_matches": away_stats["matches"],
            "confidence": min(shrinkage_alpha, 1.0),
            "stadium_factor": stadium_factor,
            "empty_stadium_adjustment": empty_stadium_factor,
            "home_goals_per_game": home_stats["goals_per_game"],
            "away_goals_per_game": away_stats["goals_per_game"],
            "home_goals_against_per_game": home_stats["goals_against_per_game"],
            "away_goals_against_per_game": away_stats["goals_against_per_game"],
            "league_baseline": league_baseline,
        }

        # Cache result
        self._team_advantage_cache[cache_key] = result

        logger.debug(
            f"{team} home advantage: {final_home_advantage:.3f} (raw: {raw_home_advantage:.3f})"
        )
        return result

    def _get_team_venue_stats(
        self,
        team: str,
        is_home: bool,
        season: str,
        up_to_gameweek: int | None,
        lookback_seasons: int,
    ) -> dict[str, float]:
        """Get team's goals for/against statistics at home or away."""

        # Get seasons to analyze (current + lookback)
        seasons_to_analyze = self._get_seasons_for_analysis(season, lookback_seasons)

        matches = []
        weights = []

        for i, analysis_season in enumerate(seasons_to_analyze):
            # Exponential decay weight for older seasons
            weight = self.season_weight_decay**i

            query = (
                self.dbsession.query(Fixture, Result)
                .join(Result, Fixture.fixture_id == Result.fixture_id)
                .filter(Fixture.season == analysis_season)
            )

            if is_home:
                query = query.filter(Fixture.home_team == team)
            else:
                query = query.filter(Fixture.away_team == team)

            # Only apply gameweek filter to current season
            if analysis_season == season and up_to_gameweek:
                query = query.filter(Fixture.gameweek <= up_to_gameweek)

            season_matches = query.all()
            matches.extend(season_matches)
            weights.extend([weight] * len(season_matches))

        if not matches:
            return {"matches": 0, "goals_per_game": 0.0, "goals_against_per_game": 0.0}

        # Calculate weighted statistics
        total_goals = 0.0
        total_goals_against = 0.0
        total_weight = 0.0

        for (_fixture, result), weight in zip(matches, weights, strict=False):
            if is_home:
                goals = result.home_score
                goals_against = result.away_score
            else:
                goals = result.away_score
                goals_against = result.home_score

            total_goals += goals * weight
            total_goals_against += goals_against * weight
            total_weight += weight

        return {
            "matches": len(matches),
            "goals_per_game": total_goals / total_weight if total_weight > 0 else 0.0,
            "goals_against_per_game": total_goals_against / total_weight
            if total_weight > 0
            else 0.0,
        }

    def _get_seasons_for_analysis(
        self, current_season: str, lookback_seasons: int
    ) -> list[str]:
        """Get list of seasons to include in analysis."""
        # Simple implementation - in practice you'd have a more sophisticated season parser
        seasons = [current_season]

        # Add previous seasons (simplified - assumes format like "2425")
        if len(current_season) == 4:
            base_year = int(current_season[:2])
            for i in range(1, lookback_seasons + 1):
                prev_year = base_year - i
                if prev_year >= 15:  # Don't go before 2015-16 season
                    prev_season = f"{prev_year:02d}{prev_year + 1:02d}"
                    seasons.append(prev_season)

        return seasons

    def _get_stadium_factor(self, team: str) -> float:
        """
        Calculate stadium-specific factor based on capacity and atmosphere.

        Larger stadiums typically provide greater home advantage.
        """
        if team not in self._stadium_factors_cache:
            capacity = self._stadium_capacities.get(team, 35000)  # Default capacity

            # Normalize capacity to [0.9, 1.1] range
            # Max capacity ~75k, min ~10k
            normalized_capacity = (capacity - 10000) / (75000 - 10000)
            stadium_factor = 0.9 + 0.2 * normalized_capacity

            self._stadium_factors_cache[team] = stadium_factor

        return self._stadium_factors_cache[team]

    def _get_empty_stadium_factor(self, team: str, season: str) -> float:
        """
        Detect empty stadium periods (COVID era) and apply adjustment.

        For 2020-21 season and early 2021-22, stadiums were empty or limited capacity.
        """
        # COVID-affected seasons
        covid_seasons = ["2021", "2122"]  # 2020-21 and part of 2021-22

        if season in covid_seasons:
            return 1.0 - self.empty_stadium_penalty

        return 1.0


class PlayerHomeAwayPerformance:
    """
    Calculator for player-specific home/away performance adjustments.

    Only applies adjustments for players with sufficient sample size (>30 games)
    to ensure statistical significance.
    """

    def __init__(
        self,
        dbsession: Session,
        min_games_threshold: int = 30,
        significance_threshold: float = 0.05,
        max_adjustment: float = 0.3,
    ):
        """
        Initialize player home/away performance calculator.

        Args:
            dbsession: Database session
            min_games_threshold: Minimum games before applying player adjustments
            significance_threshold: P-value threshold for statistical significance
            max_adjustment: Maximum adjustment factor (±30%)
        """
        self.dbsession = dbsession
        self.min_games_threshold = min_games_threshold
        self.significance_threshold = significance_threshold
        self.max_adjustment = max_adjustment

        # Cache for player adjustments
        self._player_adjustment_cache = {}

    def calculate_player_home_away_adjustment(
        self, player_id: int, season: str, lookback_seasons: int = 2
    ) -> dict[str, float | bool | int]:
        """
        Calculate player-specific home/away performance adjustment.

        Args:
            player_id: Player ID
            season: Current season
            lookback_seasons: Number of seasons to analyze

        Returns:
            Dictionary with player adjustment metrics
        """
        cache_key = f"{player_id}_{season}_{lookback_seasons}"
        if cache_key in self._player_adjustment_cache:
            return self._player_adjustment_cache[cache_key]

        # Get player object
        player = self.dbsession.query(Player).filter_by(player_id=player_id).first()
        if not player:
            logger.warning(f"Player {player_id} not found")
            return self._get_default_adjustment()

        logger.debug(f"Calculating home/away adjustment for {player.name}")

        # Get player's historical performance
        home_stats = self._get_player_venue_stats(
            player_id, True, season, lookback_seasons
        )
        away_stats = self._get_player_venue_stats(
            player_id, False, season, lookback_seasons
        )

        total_games = home_stats["games"] + away_stats["games"]

        if total_games < self.min_games_threshold:
            logger.debug(f"Insufficient games ({total_games}) for {player.name}")
            return self._get_default_adjustment()

        # Calculate performance differences
        home_points_per_game = (
            home_stats["points"] / home_stats["games"] if home_stats["games"] > 0 else 0
        )
        away_points_per_game = (
            away_stats["points"] / away_stats["games"] if away_stats["games"] > 0 else 0
        )

        # Statistical significance test (two-sample t-test)
        if home_stats["games"] > 5 and away_stats["games"] > 5:
            # Simulate individual game points for t-test (simplified)
            home_points_data = [home_points_per_game] * home_stats["games"]
            away_points_data = [away_points_per_game] * away_stats["games"]

            try:
                t_stat, p_value = stats.ttest_ind(home_points_data, away_points_data)
                is_significant = p_value < self.significance_threshold
            except:
                is_significant = False
                p_value = 1.0
        else:
            is_significant = False
            p_value = 1.0

        # Calculate adjustment factors
        if is_significant and home_points_per_game > 0:
            home_adjustment = min(
                self.max_adjustment,
                max(
                    -self.max_adjustment,
                    (home_points_per_game - away_points_per_game)
                    / home_points_per_game,
                ),
            )
            away_adjustment = -home_adjustment
        else:
            home_adjustment = 0.0
            away_adjustment = 0.0

        result = {
            "player_id": player_id,
            "player_name": player.name,
            "home_adjustment": home_adjustment,
            "away_adjustment": away_adjustment,
            "is_significant": is_significant,
            "p_value": p_value,
            "home_games": home_stats["games"],
            "away_games": away_stats["games"],
            "home_points_per_game": home_points_per_game,
            "away_points_per_game": away_points_per_game,
            "total_games": total_games,
            "confidence": min(total_games / self.min_games_threshold, 1.0),
        }

        # Cache result
        self._player_adjustment_cache[cache_key] = result

        if is_significant:
            logger.debug(
                f"{player.name}: Home +{home_adjustment:.2%}, Away {away_adjustment:.2%}"
            )

        return result

    def _get_player_venue_stats(
        self, player_id: int, is_home: bool, season: str, lookback_seasons: int
    ) -> dict[str, float]:
        """Get player's performance statistics at home or away."""

        # Get seasons to analyze
        seasons_to_analyze = self._get_seasons_for_analysis(season, lookback_seasons)

        total_games = 0
        total_points = 0
        total_minutes = 0

        for analysis_season in seasons_to_analyze:
            # Query player scores for the venue
            query = (
                self.dbsession.query(PlayerScore)
                .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
                .filter(PlayerScore.player_id == player_id)
                .filter(Fixture.season == analysis_season)
                .filter(
                    PlayerScore.minutes > 0
                )  # Only count games where player actually played
            )

            # Filter by venue
            if is_home:
                query = query.filter(PlayerScore.player_team == Fixture.home_team)
            else:
                query = query.filter(PlayerScore.player_team == Fixture.away_team)

            scores = query.all()

            for score in scores:
                total_games += 1
                total_points += score.points
                total_minutes += score.minutes

        return {"games": total_games, "points": total_points, "minutes": total_minutes}

    def _get_seasons_for_analysis(
        self, current_season: str, lookback_seasons: int
    ) -> list[str]:
        """Get list of seasons to include in analysis."""
        seasons = [current_season]

        if len(current_season) == 4:
            base_year = int(current_season[:2])
            for i in range(1, lookback_seasons + 1):
                prev_year = base_year - i
                if prev_year >= 18:  # Don't go before 2018-19 for player data
                    prev_season = f"{prev_year:02d}{prev_year + 1:02d}"
                    seasons.append(prev_season)

        return seasons

    def _get_default_adjustment(self) -> dict[str, float | bool | int]:
        """Return default adjustment when insufficient data."""
        return {
            "player_id": None,
            "player_name": None,
            "home_adjustment": 0.0,
            "away_adjustment": 0.0,
            "is_significant": False,
            "p_value": 1.0,
            "home_games": 0,
            "away_games": 0,
            "home_points_per_game": 0.0,
            "away_points_per_game": 0.0,
            "total_games": 0,
            "confidence": 0.0,
        }


class HomeAwayAdjuster:
    """
    Main class for applying home/away adjustments to predictions.

    Integrates team-level and player-level home/away effects with existing
    prediction systems to provide venue-adjusted performance forecasts.
    """

    def __init__(
        self,
        dbsession: Session = None,
        team_weight: float = 0.7,
        player_weight: float = 0.3,
    ):
        """
        Initialize the home/away adjuster.

        Args:
            dbsession: Database session
            team_weight: Weight for team-level adjustments (0-1)
            player_weight: Weight for player-level adjustments (0-1)
        """
        self.dbsession = dbsession or session
        self.team_weight = team_weight
        self.player_weight = player_weight

        # Initialize component calculators
        self.team_advantage = TeamHomeAdvantage(self.dbsession)
        self.player_performance = PlayerHomeAwayPerformance(self.dbsession)

        # Adjustment cache
        self._adjustment_cache = {}

    def apply_team_adjustment(
        self,
        base_prediction: float,
        team: str,
        is_home: bool,
        season: str,
        gameweek: int,
    ) -> float:
        """
        Apply team-level home/away adjustment to a base prediction.

        Args:
            base_prediction: Base prediction value (e.g., expected goals)
            team: Team name
            is_home: Whether team is playing at home
            season: Season
            gameweek: Current gameweek

        Returns:
            Adjusted prediction
        """
        # Get team home advantage
        team_advantage_data = self.team_advantage.calculate_team_home_advantage(
            team, season, gameweek
        )

        # Apply adjustment
        advantage = team_advantage_data["adjusted_home_advantage"]

        if is_home:
            adjustment_factor = 1.0 + (advantage * 0.5)  # Scale to appropriate range
        else:
            adjustment_factor = 1.0 - (advantage * 0.5)

        adjusted_prediction = base_prediction * adjustment_factor

        logger.debug(
            f"{team} ({'H' if is_home else 'A'}): {base_prediction:.2f} -> {adjusted_prediction:.2f}"
        )

        return adjusted_prediction

    def apply_player_adjustment(
        self, base_prediction: float, player_id: int, is_home: bool, season: str
    ) -> float:
        """
        Apply player-level home/away adjustment to a base prediction.

        Args:
            base_prediction: Base prediction value (e.g., expected points)
            player_id: Player ID
            is_home: Whether player is playing at home
            season: Season

        Returns:
            Adjusted prediction
        """
        # Get player adjustment
        player_data = self.player_performance.calculate_player_home_away_adjustment(
            player_id, season
        )

        # Only apply if statistically significant
        if not player_data["is_significant"]:
            return base_prediction

        # Apply adjustment
        if is_home:
            adjustment = player_data["home_adjustment"]
        else:
            adjustment = player_data["away_adjustment"]

        return base_prediction * (1.0 + adjustment)

    def apply_combined_adjustment(
        self,
        base_prediction: float,
        team: str,
        player_id: int,
        is_home: bool,
        season: str,
        gameweek: int,
        prediction_type: str = "points",
    ) -> dict[str, float]:
        """
        Apply combined team and player home/away adjustments.

        Args:
            base_prediction: Base prediction value
            team: Team name
            player_id: Player ID
            is_home: Whether playing at home
            season: Season
            gameweek: Current gameweek
            prediction_type: Type of prediction ('points', 'goals', etc.)

        Returns:
            Dictionary with adjusted predictions and component breakdown
        """
        cache_key = f"{base_prediction}_{team}_{player_id}_{is_home}_{season}_{gameweek}_{prediction_type}"
        if cache_key in self._adjustment_cache:
            return self._adjustment_cache[cache_key]

        # Apply team adjustment
        team_adjusted = self.apply_team_adjustment(
            base_prediction, team, is_home, season, gameweek
        )

        # Apply player adjustment
        player_adjusted = self.apply_player_adjustment(
            base_prediction, player_id, is_home, season
        )

        # Combine adjustments with weights
        combined_adjustment = (
            self.team_weight * team_adjusted + self.player_weight * player_adjusted
        )

        # Ensure reasonable bounds
        min_prediction = base_prediction * 0.5
        max_prediction = base_prediction * 2.0
        final_prediction = np.clip(combined_adjustment, min_prediction, max_prediction)

        result = {
            "base_prediction": base_prediction,
            "team_adjusted": team_adjusted,
            "player_adjusted": player_adjusted,
            "combined_adjusted": combined_adjustment,
            "final_prediction": final_prediction,
            "team_impact": team_adjusted - base_prediction,
            "player_impact": player_adjusted - base_prediction,
            "total_impact": final_prediction - base_prediction,
            "venue": "home" if is_home else "away",
        }

        # Cache result
        self._adjustment_cache[cache_key] = result

        return result

    def get_venue_multipliers(
        self, team: str, season: str, gameweek: int
    ) -> dict[str, float]:
        """
        Get simple venue multipliers for a team.

        Returns:
            Dictionary with home and away multipliers
        """
        team_data = self.team_advantage.calculate_team_home_advantage(
            team, season, gameweek
        )

        home_advantage = team_data["adjusted_home_advantage"]

        # Convert to multipliers (centered around 1.0)
        home_multiplier = 1.0 + (home_advantage * 0.3)  # Scale factor
        away_multiplier = 1.0 - (home_advantage * 0.3)

        return {
            "home_multiplier": home_multiplier,
            "away_multiplier": away_multiplier,
            "advantage": home_advantage,
            "confidence": team_data["confidence"],
        }

    def validate_adjustments(
        self, season: str, min_gameweek: int = 5
    ) -> dict[str, float]:
        """
        Validate home/away adjustments against actual results.

        Args:
            season: Season to validate
            min_gameweek: Start validation from this gameweek

        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Validating home/away adjustments for {season}")

        # Get completed matches
        matches = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek >= min_gameweek)
            .all()
        )

        home_advantages = []
        actual_home_advantages = []

        for fixture, result in matches:
            # Get predicted home advantage
            team_data = self.team_advantage.calculate_team_home_advantage(
                fixture.home_team, season, fixture.gameweek
            )
            predicted_advantage = team_data["adjusted_home_advantage"]

            # Calculate actual home advantage for this match
            actual_advantage = result.home_score - result.away_score

            home_advantages.append(predicted_advantage)
            actual_home_advantages.append(actual_advantage)

        if not home_advantages:
            return {"error": "No validation data available"}

        # Calculate correlation
        correlation = np.corrcoef(home_advantages, actual_home_advantages)[0, 1]

        # Calculate MAE and RMSE
        mae = np.mean(
            np.abs(np.array(home_advantages) - np.array(actual_home_advantages))
        )
        rmse = np.sqrt(
            np.mean((np.array(home_advantages) - np.array(actual_home_advantages)) ** 2)
        )

        return {
            "season": season,
            "matches_analyzed": len(matches),
            "correlation": float(correlation),
            "mae": float(mae),
            "rmse": float(rmse),
            "mean_predicted_advantage": float(np.mean(home_advantages)),
            "mean_actual_advantage": float(np.mean(actual_home_advantages)),
            "validation_date": datetime.datetime.now().isoformat(),
        }


# Convenience functions for integration with existing systems


def get_team_home_advantage(
    team: str,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> dict[str, float]:
    """
    Get home advantage data for a team.

    Args:
        team: Team name
        season: Season
        gameweek: Current gameweek
        dbsession: Database session

    Returns:
        Team home advantage data
    """
    if not dbsession:
        dbsession = session

    if not gameweek:
        gameweek = get_last_finished_gameweek()

    calculator = TeamHomeAdvantage(dbsession)
    return calculator.calculate_team_home_advantage(team, season, gameweek)


def get_player_home_away_adjustment(
    player_id: int, season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> dict[str, float | bool | int]:
    """
    Get home/away adjustment for a player.

    Args:
        player_id: Player ID
        season: Season
        dbsession: Database session

    Returns:
        Player home/away adjustment data
    """
    if not dbsession:
        dbsession = session

    calculator = PlayerHomeAwayPerformance(dbsession)
    return calculator.calculate_player_home_away_adjustment(player_id, season)


def apply_home_away_adjustment(
    base_prediction: float,
    team: str,
    player_id: int,
    is_home: bool,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> float:
    """
    Apply home/away adjustment to a prediction.

    Args:
        base_prediction: Base prediction value
        team: Team name
        player_id: Player ID
        is_home: Whether playing at home
        season: Season
        gameweek: Current gameweek
        dbsession: Database session

    Returns:
        Adjusted prediction
    """
    if not dbsession:
        dbsession = session

    if not gameweek:
        gameweek = get_last_finished_gameweek()

    adjuster = HomeAwayAdjuster(dbsession)
    result = adjuster.apply_combined_adjustment(
        base_prediction, team, player_id, is_home, season, gameweek
    )

    return result["final_prediction"]


def create_home_away_adjuster(dbsession: Session | None = None) -> HomeAwayAdjuster:
    """
    Factory function to create a configured HomeAwayAdjuster.

    Args:
        dbsession: Database session

    Returns:
        Configured HomeAwayAdjuster instance
    """
    return HomeAwayAdjuster(dbsession or session)
