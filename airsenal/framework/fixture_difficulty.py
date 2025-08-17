"""
Fixture Difficulty Calculator for AIrsenal

This module implements a comprehensive fixture difficulty rating system based on:
1. Elo-based team strength ratings
2. Recent form analysis with exponential decay
3. Home/away factors
4. Historical head-to-head records
5. Fixture congestion effects

The system produces difficulty ratings scaled to 1-5 (1=easiest, 5=hardest) for all fixtures.

Methodology:
-----------
Elo Rating Formula:
    Expected Score = 1 / (1 + 10^((R_B - R_A) / 400))
    New Rating = Old Rating + K * (Actual Score - Expected Score)

Home Advantage: 80 Elo points (configurable)
Form Decay: exponential with λ = 0.1 per gameweek
Fixture Congestion: -50 to +50 Elo adjustment based on games within 72 hours

Difficulty Scale:
    5: Very Hard (Expected win probability < 20%)
    4: Hard (20% ≤ p < 35%)  
    3: Medium (35% ≤ p < 65%)
    2: Easy (65% ≤ p < 80%)
    1: Very Easy (Expected win probability ≥ 80%)
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from airsenal.framework.schema import Fixture, FixtureDifficulty, PlayerScore, Result, Team, session
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import NEXT_GAMEWEEK, parse_datetime

logger = logging.getLogger(__name__)


class EloRatingSystem:
    """
    Elo rating system for tracking team strength over time.
    
    Based on the standard Elo system with modifications for football:
    - K-factor varies by match importance (20-40)
    - Home advantage of 80 points
    - Form-based adjustments
    """
    
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 30.0,
        home_advantage: float = 80.0,
        form_weight: float = 0.3
    ):
        """
        Initialize Elo rating system.
        
        Args:
            initial_rating: Starting Elo rating for new teams
            k_factor: K-factor for rating updates (higher = more volatile)
            home_advantage: Home team advantage in Elo points
            form_weight: Weight for recent form (0-1, 0=no form adjustment)
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.form_weight = form_weight
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
        
    def get_rating(self, team: str, gameweek: int = None, season: str = CURRENT_SEASON) -> float:
        """Get current Elo rating for a team."""
        key = f"{team}_{season}"
        if key not in self.ratings:
            self.ratings[key] = self.initial_rating
        return self.ratings[key]
    
    def update_rating(
        self,
        team: str,
        opponent: str,
        actual_score: float,
        is_home: bool,
        gameweek: int,
        season: str = CURRENT_SEASON,
        importance_factor: float = 1.0
    ) -> float:
        """
        Update team's Elo rating based on match result.
        
        Args:
            team: Team name
            opponent: Opponent team name
            actual_score: Actual match result (1.0=win, 0.5=draw, 0.0=loss)
            is_home: Whether team played at home
            gameweek: Gameweek number
            season: Season
            importance_factor: Multiplier for K-factor (e.g., 1.5 for crucial matches)
            
        Returns:
            New Elo rating for the team
        """
        team_rating = self.get_rating(team, gameweek, season)
        opponent_rating = self.get_rating(opponent, gameweek, season)
        
        # Apply home advantage
        if is_home:
            team_rating += self.home_advantage
        
        # Calculate expected score
        expected_score = 1 / (1 + 10**((opponent_rating - team_rating) / 400))
        
        # Calculate rating change
        k_adjusted = self.k_factor * importance_factor
        rating_change = k_adjusted * (actual_score - expected_score)
        
        # Update rating
        key = f"{team}_{season}"
        new_rating = self.get_rating(team, gameweek, season) + rating_change
        self.ratings[key] = new_rating
        
        # Store in history
        self.rating_history[key].append((f"GW{gameweek}", gameweek, new_rating))
        
        return new_rating
    
    def expected_score(
        self,
        team_a: str,
        team_b: str,
        is_team_a_home: bool,
        gameweek: int = None,
        season: str = CURRENT_SEASON
    ) -> float:
        """
        Calculate expected score for team_a vs team_b.
        
        Returns probability that team_a wins (0-1 scale).
        """
        rating_a = self.get_rating(team_a, gameweek, season)
        rating_b = self.get_rating(team_b, gameweek, season)
        
        if is_team_a_home:
            rating_a += self.home_advantage
        
        return 1 / (1 + 10**((rating_b - rating_a) / 400))


class FixtureDifficultyCalculator:
    """
    Comprehensive fixture difficulty calculator combining multiple factors:
    - Elo-based team strength
    - Recent form analysis  
    - Home/away factors
    - Historical head-to-head
    - Fixture congestion
    """
    
    def __init__(
        self,
        dbsession: Optional[Session] = None,
        season: str = CURRENT_SEASON,
        form_lookback: int = 6,
        form_decay: float = 0.1,
        congestion_hours: int = 72
    ):
        """
        Initialize fixture difficulty calculator.
        
        Args:
            dbsession: Database session
            season: Season to analyze
            form_lookback: Number of recent games to consider for form
            form_decay: Exponential decay factor for form (per gameweek)
            congestion_hours: Hours threshold for fixture congestion
        """
        self.dbsession = dbsession or session
        self.season = season
        self.form_lookback = form_lookback
        self.form_decay = form_decay
        self.congestion_hours = congestion_hours
        
        # Initialize Elo system
        self.elo_system = EloRatingSystem()
        
        # Cache for computed values
        self._team_form_cache: Dict[str, Dict[int, float]] = {}
        self._h2h_cache: Dict[Tuple[str, str], Dict] = {}
        self._congestion_cache: Dict[Tuple[str, int], float] = {}
        
        # Initialize from historical data
        self._initialize_elo_ratings()
    
    def _initialize_elo_ratings(self) -> None:
        """Initialize Elo ratings from historical match results."""
        logger.info(f"Initializing Elo ratings for season {self.season}")
        
        # Get all completed fixtures in chronological order
        fixtures = (
            self.dbsession.query(Fixture)
            .filter(Fixture.season == self.season)
            .filter(Fixture.result.isnot(None))
            .join(Result)
            .order_by(Fixture.gameweek, Fixture.date)
            .all()
        )
        
        for fixture in fixtures:
            if not fixture.result:
                continue
                
            home_team = fixture.home_team
            away_team = fixture.away_team
            home_score = fixture.result.home_score
            away_score = fixture.result.away_score
            
            # Determine match result
            if home_score > away_score:
                home_result, away_result = 1.0, 0.0
            elif home_score < away_score:
                home_result, away_result = 0.0, 1.0
            else:
                home_result, away_result = 0.5, 0.5
            
            # Update Elo ratings
            self.elo_system.update_rating(
                home_team, away_team, home_result, True,
                fixture.gameweek, self.season
            )
            self.elo_system.update_rating(
                away_team, home_team, away_result, False,
                fixture.gameweek, self.season
            )
        
        logger.info(f"Initialized Elo ratings for {len(self.elo_system.ratings)} teams")
    
    def calculate_team_form(self, team: str, gameweek: int) -> float:
        """
        Calculate recent form score for a team using exponential decay.
        
        Returns form score (0-1 scale, higher = better form).
        """
        cache_key = (team, gameweek)
        if cache_key in self._team_form_cache.get(team, {}):
            return self._team_form_cache[team][cache_key]
        
        # Get recent results
        recent_fixtures = (
            self.dbsession.query(Fixture)
            .filter(Fixture.season == self.season)
            .filter(
                or_(
                    Fixture.home_team == team,
                    Fixture.away_team == team
                )
            )
            .filter(Fixture.gameweek < gameweek)
            .filter(Fixture.result.isnot(None))
            .join(Result)
            .order_by(desc(Fixture.gameweek))
            .limit(self.form_lookback)
            .all()
        )
        
        if not recent_fixtures:
            return 0.5  # Neutral form for teams with no recent games
        
        # Calculate weighted form score
        form_score = 0.0
        total_weight = 0.0
        
        for i, fixture in enumerate(recent_fixtures):
            # Determine result from team's perspective
            if fixture.home_team == team:
                team_score = fixture.result.home_score
                opp_score = fixture.result.away_score
            else:
                team_score = fixture.result.away_score
                opp_score = fixture.result.home_score
            
            # Convert to performance score (0-1)
            if team_score > opp_score:
                performance = 1.0
            elif team_score == opp_score:
                performance = 0.5
            else:
                performance = 0.0
            
            # Apply exponential decay based on recency
            weeks_ago = gameweek - fixture.gameweek
            weight = math.exp(-self.form_decay * weeks_ago)
            
            form_score += performance * weight
            total_weight += weight
        
        form_score = form_score / total_weight if total_weight > 0 else 0.5
        
        # Cache result
        if team not in self._team_form_cache:
            self._team_form_cache[team] = {}
        self._team_form_cache[team][cache_key] = form_score
        
        return form_score
    
    def calculate_head_to_head(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Calculate head-to-head statistics between two teams.
        
        Returns dict with home team advantage and recent h2h performance.
        """
        cache_key = (home_team, away_team)
        if cache_key in self._h2h_cache:
            return self._h2h_cache[cache_key]
        
        # Get historical h2h fixtures (last 10 meetings)
        h2h_fixtures = (
            self.dbsession.query(Fixture)
            .filter(
                or_(
                    and_(Fixture.home_team == home_team, Fixture.away_team == away_team),
                    and_(Fixture.home_team == away_team, Fixture.away_team == home_team)
                )
            )
            .filter(Fixture.result.isnot(None))
            .join(Result)
            .order_by(desc(Fixture.date))
            .limit(10)
            .all()
        )
        
        if not h2h_fixtures:
            h2h_stats = {
                'home_advantage': 0.0,
                'recent_performance': 0.5,
                'matches_played': 0
            }
        else:
            home_wins = 0
            away_wins = 0
            draws = 0
            home_team_wins = 0  # When current home team won (regardless of venue)
            
            for fixture in h2h_fixtures:
                home_score = fixture.result.home_score
                away_score = fixture.result.away_score
                
                if home_score > away_score:
                    home_wins += 1
                    if fixture.home_team == home_team:
                        home_team_wins += 1
                elif away_score > home_score:
                    away_wins += 1
                    if fixture.away_team == home_team:
                        home_team_wins += 1
                else:
                    draws += 1
            
            total_matches = len(h2h_fixtures)
            
            # Calculate home advantage (how often home team wins in this matchup)
            home_advantage = (home_wins / total_matches - 0.5) * 2 if total_matches > 0 else 0.0
            
            # Recent performance of current home team in this matchup
            recent_performance = (home_team_wins + 0.5 * draws) / total_matches if total_matches > 0 else 0.5
            
            h2h_stats = {
                'home_advantage': home_advantage,
                'recent_performance': recent_performance,
                'matches_played': total_matches
            }
        
        self._h2h_cache[cache_key] = h2h_stats
        return h2h_stats
    
    def calculate_fixture_congestion(self, team: str, fixture_date: str) -> float:
        """
        Calculate fixture congestion penalty/bonus for a team.
        
        Returns adjustment factor (-1 to +1, negative = congested schedule).
        """
        if not fixture_date:
            return 0.0
        
        fixture_datetime = parse_datetime(fixture_date)
        
        # Get fixtures within congestion window
        start_window = fixture_datetime - timedelta(hours=self.congestion_hours)
        end_window = fixture_datetime + timedelta(hours=self.congestion_hours)
        
        nearby_fixtures = (
            self.dbsession.query(Fixture)
            .filter(Fixture.season == self.season)
            .filter(
                or_(
                    Fixture.home_team == team,
                    Fixture.away_team == team
                )
            )
            .filter(Fixture.date.isnot(None))
            .all()
        )
        
        congestion_count = 0
        for fixture in nearby_fixtures:
            if not fixture.date:
                continue
            fdate = parse_datetime(fixture.date)
            if start_window <= fdate <= end_window and fdate != fixture_datetime:
                congestion_count += 1
        
        # Convert to adjustment factor
        # 0 nearby games = 0.0 (neutral)
        # 1 nearby game = -0.3 (slight penalty)
        # 2+ nearby games = -0.7 (high penalty)
        if congestion_count == 0:
            return 0.0
        elif congestion_count == 1:
            return -0.3
        else:
            return -0.7
    
    def calculate_fixture_difficulty(
        self,
        fixture: Fixture,
        perspective_team: Optional[str] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate comprehensive difficulty rating for a fixture.
        
        Args:
            fixture: Fixture object
            perspective_team: Calculate difficulty from this team's perspective
                            (None = calculate for both teams)
        
        Returns:
            Dictionary with difficulty ratings and component breakdown
        """
        home_team = fixture.home_team
        away_team = fixture.away_team
        gameweek = fixture.gameweek or NEXT_GAMEWEEK
        
        # Calculate base Elo expected scores
        home_expected = self.elo_system.expected_score(
            home_team, away_team, True, gameweek, self.season
        )
        away_expected = 1 - home_expected
        
        # Get recent form
        home_form = self.calculate_team_form(home_team, gameweek)
        away_form = self.calculate_team_form(away_team, gameweek)
        
        # Get head-to-head stats
        h2h_stats = self.calculate_head_to_head(home_team, away_team)
        
        # Get fixture congestion
        home_congestion = self.calculate_fixture_congestion(home_team, fixture.date)
        away_congestion = self.calculate_fixture_congestion(away_team, fixture.date)
        
        # Adjust expected scores based on form and congestion
        form_adjustment = 0.2 * (home_form - away_form)  # Max ±0.2 adjustment
        congestion_adjustment = 0.1 * (away_congestion - home_congestion)  # Max ±0.1
        h2h_adjustment = 0.1 * h2h_stats['home_advantage']  # Max ±0.1
        
        adjusted_home_expected = np.clip(
            home_expected + form_adjustment + congestion_adjustment + h2h_adjustment,
            0.05, 0.95
        )
        adjusted_away_expected = 1 - adjusted_home_expected
        
        # Convert to 1-5 difficulty scale
        def expected_to_difficulty(expected_win_prob: float) -> float:
            """Convert expected win probability to 1-5 difficulty scale."""
            if expected_win_prob >= 0.8:
                return 1.0  # Very Easy
            elif expected_win_prob >= 0.65:
                return 2.0  # Easy
            elif expected_win_prob >= 0.35:
                return 3.0  # Medium
            elif expected_win_prob >= 0.2:
                return 4.0  # Hard
            else:
                return 5.0  # Very Hard
        
        home_difficulty = expected_to_difficulty(adjusted_home_expected)
        away_difficulty = expected_to_difficulty(adjusted_away_expected)
        
        result = {
            'fixture_id': fixture.fixture_id,
            'gameweek': gameweek,
            'season': fixture.season,
            'home_team': home_team,
            'away_team': away_team,
            'home_difficulty': home_difficulty,
            'away_difficulty': away_difficulty,
            'home_expected_score': adjusted_home_expected,
            'away_expected_score': adjusted_away_expected,
            'components': {
                'elo_ratings': {
                    'home': self.elo_system.get_rating(home_team, gameweek, self.season),
                    'away': self.elo_system.get_rating(away_team, gameweek, self.season)
                },
                'form_scores': {
                    'home': home_form,
                    'away': away_form
                },
                'congestion_factors': {
                    'home': home_congestion,
                    'away': away_congestion
                },
                'h2h_stats': h2h_stats,
                'adjustments': {
                    'form': form_adjustment,
                    'congestion': congestion_adjustment,
                    'h2h': h2h_adjustment
                }
            }
        }
        
        # Filter by perspective team if specified
        if perspective_team:
            if perspective_team == home_team:
                result['difficulty'] = home_difficulty
                result['expected_score'] = adjusted_home_expected
            elif perspective_team == away_team:
                result['difficulty'] = away_difficulty  
                result['expected_score'] = adjusted_away_expected
            else:
                raise ValueError(f"Team {perspective_team} not found in fixture")
        
        return result
    
    def calculate_bulk_difficulties(
        self,
        gameweek_start: int,
        gameweek_end: int,
        team: Optional[str] = None
    ) -> List[Dict]:
        """
        Calculate fixture difficulties for a range of gameweeks.
        
        Args:
            gameweek_start: Starting gameweek
            gameweek_end: Ending gameweek (inclusive)
            team: Optional team filter
            
        Returns:
            List of difficulty dictionaries
        """
        query = (
            self.dbsession.query(Fixture)
            .filter(Fixture.season == self.season)
            .filter(Fixture.gameweek >= gameweek_start)
            .filter(Fixture.gameweek <= gameweek_end)
        )
        
        if team:
            query = query.filter(
                or_(
                    Fixture.home_team == team,
                    Fixture.away_team == team
                )
            )
        
        fixtures = query.order_by(Fixture.gameweek, Fixture.date).all()
        
        results = []
        for fixture in fixtures:
            try:
                difficulty = self.calculate_fixture_difficulty(fixture, team)
                results.append(difficulty)
            except Exception as e:
                logger.warning(f"Error calculating difficulty for fixture {fixture.fixture_id}: {e}")
                continue
        
        return results
    
    def get_team_average_difficulty(
        self,
        team: str,
        gameweek_start: int,
        gameweek_end: int
    ) -> Dict[str, float]:
        """
        Calculate average fixture difficulty for a team over a period.
        
        Returns:
            Dictionary with average difficulty and fixture count
        """
        difficulties = self.calculate_bulk_difficulties(
            gameweek_start, gameweek_end, team
        )
        
        if not difficulties:
            return {
                'team': team,
                'gameweek_start': gameweek_start,
                'gameweek_end': gameweek_end,
                'average_difficulty': 3.0,  # Neutral difficulty
                'fixture_count': 0,
                'min_difficulty': None,
                'max_difficulty': None
            }
        
        difficulty_values = [d['difficulty'] for d in difficulties]
        
        return {
            'team': team,
            'gameweek_start': gameweek_start,
            'gameweek_end': gameweek_end,
            'average_difficulty': np.mean(difficulty_values),
            'fixture_count': len(difficulty_values),
            'min_difficulty': min(difficulty_values),
            'max_difficulty': max(difficulty_values),
            'difficulties': difficulty_values
        }
    
    def validate_predictions(self, season: str = None) -> Dict[str, float]:
        """
        Validate fixture difficulty predictions against actual results.
        
        Returns:
            Dictionary with validation metrics
        """
        season = season or self.season
        
        # Get completed fixtures with predictions
        completed_fixtures = (
            self.dbsession.query(Fixture)
            .filter(Fixture.season == season)
            .filter(Fixture.result.isnot(None))
            .join(Result)
            .all()
        )
        
        predictions = []
        actuals = []
        
        for fixture in completed_fixtures:
            try:
                difficulty_data = self.calculate_fixture_difficulty(fixture)
                
                # Extract predictions
                home_expected = difficulty_data['home_expected_score']
                away_expected = difficulty_data['away_expected_score']
                
                # Extract actual results
                home_score = fixture.result.home_score
                away_score = fixture.result.away_score
                
                if home_score > away_score:
                    actual_home, actual_away = 1.0, 0.0
                elif away_score > home_score:
                    actual_home, actual_away = 0.0, 1.0
                else:
                    actual_home, actual_away = 0.5, 0.5
                
                predictions.extend([home_expected, away_expected])
                actuals.extend([actual_home, actual_away])
                
            except Exception as e:
                logger.warning(f"Error validating fixture {fixture.fixture_id}: {e}")
                continue
        
        if not predictions:
            return {'error': 'No valid predictions found'}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate validation metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        # Calculate log loss (Brier score for probability predictions)
        epsilon = 1e-15  # Avoid log(0)
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        log_loss = -np.mean(actuals * np.log(predictions_clipped) + 
                           (1 - actuals) * np.log(1 - predictions_clipped))
        
        return {
            'season': season,
            'fixtures_analyzed': len(completed_fixtures),
            'predictions_count': len(predictions),
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'log_loss': log_loss,
            'brier_score': np.mean((predictions - actuals) ** 2)
        }


def create_difficulty_calculator(
    season: str = CURRENT_SEASON,
    dbsession: Optional[Session] = None
) -> FixtureDifficultyCalculator:
    """
    Factory function to create a fixture difficulty calculator.
    
    Args:
        season: Season to analyze
        dbsession: Database session
        
    Returns:
        Configured FixtureDifficultyCalculator instance
    """
    return FixtureDifficultyCalculator(
        dbsession=dbsession or session,
        season=season
    )


# Convenience functions for common use cases
def get_fixture_difficulty(
    fixture_id: int,
    perspective_team: Optional[str] = None,
    season: str = CURRENT_SEASON,
    dbsession: Optional[Session] = None
) -> Dict:
    """Get difficulty rating for a specific fixture."""
    dbsession = dbsession or session
    fixture = dbsession.query(Fixture).filter_by(fixture_id=fixture_id).first()
    
    if not fixture:
        raise ValueError(f"Fixture {fixture_id} not found")
    
    calculator = create_difficulty_calculator(season, dbsession)
    return calculator.calculate_fixture_difficulty(fixture, perspective_team)


def get_team_fixture_difficulties(
    team: str,
    gameweek_start: int,
    gameweek_end: int,
    season: str = CURRENT_SEASON,
    dbsession: Optional[Session] = None
) -> List[Dict]:
    """Get fixture difficulties for a team over a period."""
    calculator = create_difficulty_calculator(season, dbsession)
    return calculator.calculate_bulk_difficulties(
        gameweek_start, gameweek_end, team
    )


def get_all_teams_average_difficulty(
    gameweek_start: int,
    gameweek_end: int,
    season: str = CURRENT_SEASON,
    dbsession: Optional[Session] = None
) -> Dict[str, Dict]:
    """Get average fixture difficulty for all teams over a period."""
    dbsession = dbsession or session
    calculator = create_difficulty_calculator(season, dbsession)
    
    # Get all teams
    teams = dbsession.query(Team).filter_by(season=season).all()
    team_names = [team.name for team in teams]
    
    results = {}
    for team in team_names:
        try:
            results[team] = calculator.get_team_average_difficulty(
                team, gameweek_start, gameweek_end
            )
        except Exception as e:
            logger.warning(f"Error calculating average difficulty for {team}: {e}")
            continue
    
    return results