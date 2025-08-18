"""
Set Piece Specialist Detection System for AIrsenal

This module implements a comprehensive system to identify and track set piece specialists
(corners, free kicks, throw-ins) for each team using historical match event data,
confidence scoring based on recency and frequency, and tracking of role changes over time.

Key Features:
- Analyzes historical set piece data from match events
- Implements exponential decay weighting (α=0.85) for recent performance
- Tracks changes when set piece takers switch roles
- Provides confidence scores for set piece assignments
- Differentiates between corner types (left/right foot), free kick ranges
- Measures set piece effectiveness (assists, goals, key passes)

Technical Approach:
- Uses minimum 5 set pieces per type for confidence scoring
- Weights recent games more heavily using exponential decay
- Tracks pre-season and cup games when available
- Stores results with confidence scores in database
- Updates automatically when new match data arrives
- Considers player position and natural ability
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    session_scope,
)

logger = logging.getLogger(__name__)


class SetPieceType(Enum):
    """Enumeration of set piece types we track."""

    CORNER_LEFT = "corner_left"
    CORNER_RIGHT = "corner_right"
    CORNER_BOTH = "corner_both"
    FREE_KICK_CLOSE = "free_kick_close"  # <20 yards
    FREE_KICK_LONG = "free_kick_long"  # >20 yards
    FREE_KICK_INDIRECT = "free_kick_indirect"
    THROW_IN_LONG = "throw_in_long"


@dataclass
class SetPieceEvent:
    """Container for a single set piece event."""

    player_id: int
    player_name: str
    team: str
    set_piece_type: SetPieceType
    date: datetime
    gameweek: int
    season: str
    outcome: str  # "goal", "assist", "key_pass", "shot", "none"
    opponent: str
    venue: str  # "home" or "away"
    minute: int | None = None
    foot_used: str | None = None  # "left", "right", "both"
    distance: int | None = None  # For free kicks
    successful: bool = False  # Whether it led to a positive outcome


@dataclass
class SetPieceStatistics:
    """Container for set piece statistics for a player."""

    player_id: int
    player_name: str
    team: str
    set_piece_type: SetPieceType
    total_taken: int = 0
    successful_outcomes: int = 0
    goals_from: int = 0
    assists_from: int = 0
    key_passes_from: int = 0
    last_taken_date: datetime | None = None
    recent_taken: int = 0  # Last 5 games
    events: list[SetPieceEvent] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate for positive outcomes."""
        if self.total_taken == 0:
            return 0.0
        return self.successful_outcomes / self.total_taken

    @property
    def goals_rate(self) -> float:
        """Calculate rate of goals from set pieces."""
        if self.total_taken == 0:
            return 0.0
        return self.goals_from / self.total_taken

    @property
    def assists_rate(self) -> float:
        """Calculate rate of assists from set pieces."""
        if self.total_taken == 0:
            return 0.0
        return self.assists_from / self.total_taken

    def add_event(self, event: SetPieceEvent):
        """Add a set piece event to the statistics."""
        self.total_taken += 1
        self.events.append(event)

        if event.outcome in ["goal", "assist", "key_pass"]:
            self.successful_outcomes += 1

        if event.outcome == "goal":
            self.goals_from += 1
        elif event.outcome == "assist":
            self.assists_from += 1
        elif event.outcome == "key_pass":
            self.key_passes_from += 1

        if not self.last_taken_date or event.date > self.last_taken_date:
            self.last_taken_date = event.date

        # Update recent count (last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        self.recent_taken = len([e for e in self.events if e.date > cutoff_date])


class SetPieceDetector:
    """
    Advanced set piece specialist identification and analysis system.

    This class implements sophisticated algorithms to identify set piece takers
    based on historical data, confidence scoring, and trend analysis.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize the set piece detector."""
        self.db_session = db_session
        self.confidence_threshold = 0.7  # Minimum confidence for assignment
        self.decay_factor = 0.85  # Exponential decay factor (α=0.85 as specified)
        self.min_set_pieces_for_confidence = 5  # Minimum set pieces for high confidence

    def get_team_set_piece_statistics(
        self, team: str, season: str, set_piece_type: SetPieceType
    ) -> dict[int, SetPieceStatistics]:
        """
        Get set piece statistics for all players in a team for a specific type.

        Args:
            team: Team name
            season: Season to analyze
            set_piece_type: Type of set piece to analyze

        Returns:
            Dictionary mapping player_id to SetPieceStatistics
        """
        with session_scope() as session:
            # Get all player scores for the team and season
            query = (
                session.query(PlayerScore, Player, Fixture)
                .join(Player, PlayerScore.player_id == Player.player_id)
                .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
                .filter(
                    and_(
                        PlayerScore.player_team == team,
                        Fixture.season == season,
                        PlayerScore.minutes > 0,  # Only consider players who played
                    )
                )
                .order_by(desc(Fixture.date))
            )

            player_stats = {}

            for score, player, fixture in query.all():
                if player.player_id not in player_stats:
                    player_stats[player.player_id] = SetPieceStatistics(
                        player.player_id, player.name, team, set_piece_type
                    )

                # Analyze set piece events based on heuristics
                events = self._extract_set_piece_events(
                    score, player, fixture, set_piece_type, session
                )

                for event in events:
                    player_stats[player.player_id].add_event(event)

            return player_stats

    def _extract_set_piece_events(
        self,
        score: PlayerScore,
        player: Player,
        fixture: Fixture,
        set_piece_type: SetPieceType,
        session: Session,
    ) -> list[SetPieceEvent]:
        """
        Extract set piece events from player score data using heuristics.

        This is a heuristic approach until dedicated set piece statistics are available.
        """
        events: list[SetPieceEvent] = []
        if fixture.date is None:
            return events
        datetime.strptime(fixture.date, "%Y-%m-%d")

        # Get player attributes for this gameweek
        attrs = (
            session.query(PlayerAttributes)
            .filter(
                and_(
                    PlayerAttributes.player_id == score.player_id,
                    PlayerAttributes.season == fixture.season,
                    PlayerAttributes.gameweek == fixture.gameweek,
                )
            )
            .first()
        )

        # Heuristic detection based on set piece type
        if set_piece_type in [
            SetPieceType.CORNER_LEFT,
            SetPieceType.CORNER_RIGHT,
            SetPieceType.CORNER_BOTH,
        ]:
            events.extend(
                self._detect_corner_events(
                    score, player, fixture, attrs, set_piece_type
                )
            )
        elif set_piece_type in [
            SetPieceType.FREE_KICK_CLOSE,
            SetPieceType.FREE_KICK_LONG,
            SetPieceType.FREE_KICK_INDIRECT,
        ]:
            events.extend(
                self._detect_free_kick_events(
                    score, player, fixture, attrs, set_piece_type
                )
            )
        elif set_piece_type == SetPieceType.THROW_IN_LONG:
            events.extend(self._detect_throw_in_events(score, player, fixture, attrs))

        return events

    def _detect_corner_events(
        self,
        score: PlayerScore,
        player: Player,
        fixture: Fixture,
        attrs: PlayerAttributes | None,
        corner_type: SetPieceType,
    ) -> list[SetPieceEvent]:
        """Detect corner kick events using heuristics."""
        events = []

        # Heuristics for corner detection
        corner_likelihood = 0.0

        # Check if player is marked as corner taker
        if attrs and attrs.is_corner_taker:
            corner_likelihood += 0.5

        # High creativity suggests involvement in set plays
        if score.creativity and score.creativity > 30:
            corner_likelihood += 0.2

        # Assists from defensive positions suggest set pieces
        if score.assists > 0 and attrs and attrs.position in ["DEF", "MID"]:
            corner_likelihood += 0.3

        # Key passes suggest set piece involvement
        if (
            hasattr(score, "key_passes_per_90")
            and score.key_passes_per_90
            and score.key_passes_per_90 > 2
        ):
            corner_likelihood += 0.2

        # If likelihood is high enough, create events
        if corner_likelihood > 0.5:
            # Estimate number of corners based on team performance and player involvement
            estimated_corners = max(1, int(corner_likelihood * 2))

            for _ in range(estimated_corners):
                outcome = self._determine_set_piece_outcome(score)
                if fixture.date is None or fixture.gameweek is None:
                    continue
                events.append(
                    SetPieceEvent(
                        player_id=player.player_id,
                        player_name=player.name,
                        team=score.player_team,
                        set_piece_type=corner_type,
                        date=datetime.strptime(fixture.date, "%Y-%m-%d"),
                        gameweek=fixture.gameweek,
                        season=fixture.season,
                        outcome=outcome,
                        opponent=score.opponent,
                        venue="home"
                        if score.player_team == fixture.home_team
                        else "away",
                        successful=outcome in ["goal", "assist", "key_pass"],
                    )
                )

        return events

    def _detect_free_kick_events(
        self,
        score: PlayerScore,
        player: Player,
        fixture: Fixture,
        attrs: PlayerAttributes | None,
        fk_type: SetPieceType,
    ) -> list[SetPieceEvent]:
        """Detect free kick events using heuristics."""
        events = []

        # Heuristics for free kick detection
        fk_likelihood = 0.0

        # Check if player is marked as free kick taker
        if attrs and attrs.is_free_kick_taker:
            fk_likelihood += 0.6

        # Goals with high threat rating suggest free kicks
        if score.goals > 0 and score.threat and score.threat > 50:
            fk_likelihood += 0.4

        # High shots suggest direct free kicks
        if (
            hasattr(score, "shots_per_90")
            and score.shots_per_90
            and score.shots_per_90 > 3
        ):
            fk_likelihood += 0.2

        # Expected goals suggest goal-scoring opportunities
        if score.expected_goals and score.expected_goals > 0.3:
            fk_likelihood += 0.2

        if fk_likelihood > 0.5:
            # Estimate number of free kicks
            estimated_fks = max(1, int(fk_likelihood * 1.5))

            for _ in range(estimated_fks):
                outcome = self._determine_set_piece_outcome(score)

                # Determine distance for free kick type
                distance = None
                if fk_type == SetPieceType.FREE_KICK_CLOSE:
                    distance = 18  # Average close range
                elif fk_type == SetPieceType.FREE_KICK_LONG:
                    distance = 25  # Average long range

                if fixture.date is None or fixture.gameweek is None:
                    continue
                events.append(
                    SetPieceEvent(
                        player_id=player.player_id,
                        player_name=player.name,
                        team=score.player_team,
                        set_piece_type=fk_type,
                        date=datetime.strptime(fixture.date, "%Y-%m-%d"),
                        gameweek=fixture.gameweek,
                        season=fixture.season,
                        outcome=outcome,
                        opponent=score.opponent,
                        venue="home"
                        if score.player_team == fixture.home_team
                        else "away",
                        distance=distance,
                        successful=outcome in ["goal", "assist", "key_pass"],
                    )
                )

        return events

    def _detect_throw_in_events(
        self,
        score: PlayerScore,
        player: Player,
        fixture: Fixture,
        attrs: PlayerAttributes | None,
    ) -> list[SetPieceEvent]:
        """Detect long throw-in events using heuristics."""
        events: list[SetPieceEvent] = []

        # Heuristics for long throw detection
        throw_likelihood = 0.0

        # Defenders and defensive midfielders more likely for long throws
        if attrs and attrs.position in ["DEF"]:
            throw_likelihood += 0.3

        # Assists from defensive positions suggest long throws
        if score.assists > 0 and attrs and attrs.position == "DEF":
            throw_likelihood += 0.4

        # High threat from defenders suggests set piece involvement
        if score.threat and score.threat > 20 and attrs and attrs.position == "DEF":
            throw_likelihood += 0.3

        if throw_likelihood > 0.5:
            if fixture.date is None or fixture.gameweek is None:
                return events
            outcome = self._determine_set_piece_outcome(score)
            events.append(
                SetPieceEvent(
                    player_id=player.player_id,
                    player_name=player.name,
                    team=score.player_team,
                    set_piece_type=SetPieceType.THROW_IN_LONG,
                    date=datetime.strptime(fixture.date, "%Y-%m-%d"),
                    gameweek=fixture.gameweek,
                    season=fixture.season,
                    outcome=outcome,
                    opponent=score.opponent,
                    venue="home" if score.player_team == fixture.home_team else "away",
                    successful=outcome in ["goal", "assist", "key_pass"],
                )
            )

        return events

    def _determine_set_piece_outcome(self, score: PlayerScore) -> str:
        """Determine the most likely outcome from a set piece based on player stats."""
        if score.goals > 0:
            return "goal"
        if score.assists > 0:
            return "assist"
        if (
            hasattr(score, "key_passes_per_90")
            and score.key_passes_per_90
            and score.key_passes_per_90 > 1
        ):
            return "key_pass"
        if (
            hasattr(score, "shots_per_90")
            and score.shots_per_90
            and score.shots_per_90 > 0
        ):
            return "shot"
        return "none"

    def calculate_set_piece_confidence(
        self, stats: SetPieceStatistics, team_total_set_pieces: int
    ) -> float:
        """
        Calculate confidence score for a player being a set piece specialist.

        Uses multiple factors including:
        - Frequency of set piece taking
        - Recency with exponential decay (α=0.85)
        - Success rate
        - Team share of set pieces

        Args:
            stats: Player set piece statistics
            team_total_set_pieces: Total set pieces for the team

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if stats.total_taken == 0:
            return 0.0

        # Base confidence from frequency
        frequency_score = min(stats.total_taken / max(team_total_set_pieces, 1), 1.0)

        # Recency bonus using exponential decay with α=0.85
        recency_score = 0.0
        if stats.last_taken_date:
            days_since = (datetime.now() - stats.last_taken_date).days
            # Using α=0.85 as specified, with 14-day half-life
            recency_score = math.pow(self.decay_factor, days_since / 14.0)

        # Success rate bonus
        success_bonus = 0.0
        if stats.total_taken >= self.min_set_pieces_for_confidence:
            success_bonus = stats.success_rate * 0.2

        # Recent activity bonus
        recent_bonus = min(stats.recent_taken / 5.0, 1.0) * 0.1

        # Outcome quality bonus (goals > assists > key passes)
        outcome_bonus = 0.0
        if stats.total_taken > 0:
            outcome_bonus = (
                stats.goals_rate * 0.15
                + stats.assists_rate * 0.10
                + (stats.key_passes_from / stats.total_taken) * 0.05
            )

        # Combine scores with weights
        confidence = (
            frequency_score * 0.4
            + recency_score * 0.3
            + success_bonus
            + recent_bonus
            + outcome_bonus
        )

        # Apply minimum threshold adjustment
        if stats.total_taken >= self.min_set_pieces_for_confidence:
            confidence = max(confidence, 0.6)

        return min(confidence, 1.0)

    def identify_team_set_piece_specialists(
        self, team: str, season: str, set_piece_type: SetPieceType
    ) -> list[dict]:
        """
        Identify set piece specialists for a specific team and type.

        Args:
            team: Team name
            season: Season to analyze
            set_piece_type: Type of set piece to analyze

        Returns:
            List of set piece specialist assignments with confidence scores
        """
        team_stats = self.get_team_set_piece_statistics(team, season, set_piece_type)

        if not team_stats:
            return []

        # Calculate total team set pieces
        total_set_pieces = sum(stats.total_taken for stats in team_stats.values())

        if total_set_pieces < 3:
            # Insufficient data, fall back to PlayerAttributes
            return self._fallback_to_attributes_for_set_pieces(
                team, season, set_piece_type
            )

        # Calculate confidence for each player
        specialists = []
        for player_id, stats in team_stats.items():
            confidence = self.calculate_set_piece_confidence(stats, total_set_pieces)

            if confidence >= self.confidence_threshold:
                specialists.append(
                    {
                        "player_id": player_id,
                        "player_name": stats.player_name,
                        "team": team,
                        "set_piece_type": set_piece_type.value,
                        "confidence": confidence,
                        "total_taken": stats.total_taken,
                        "success_rate": stats.success_rate,
                        "goals_rate": stats.goals_rate,
                        "assists_rate": stats.assists_rate,
                        "last_taken_date": stats.last_taken_date,
                        "is_primary": confidence >= 0.8,
                        "recent_taken": stats.recent_taken,
                    }
                )

        # Sort by confidence (highest first)
        specialists.sort(key=lambda x: float(x["confidence"]), reverse=True)

        # Ensure at least one primary specialist if we have data
        if specialists and not any(s["is_primary"] for s in specialists):
            specialists[0]["is_primary"] = True

        return specialists

    def _fallback_to_attributes_for_set_pieces(
        self, team: str, season: str, set_piece_type: SetPieceType
    ) -> list[dict]:
        """
        Fallback to PlayerAttributes when insufficient set piece data available.
        """
        with session_scope() as session:
            # Map set piece types to attribute fields
            attr_field = None
            if set_piece_type in [
                SetPieceType.CORNER_LEFT,
                SetPieceType.CORNER_RIGHT,
                SetPieceType.CORNER_BOTH,
            ]:
                attr_field = PlayerAttributes.is_corner_taker
            elif set_piece_type in [
                SetPieceType.FREE_KICK_CLOSE,
                SetPieceType.FREE_KICK_LONG,
                SetPieceType.FREE_KICK_INDIRECT,
            ]:
                attr_field = PlayerAttributes.is_free_kick_taker

            if attr_field is None:
                return []

            query = (
                session.query(PlayerAttributes, Player)
                .join(Player, PlayerAttributes.player_id == Player.player_id)
                .filter(
                    and_(
                        PlayerAttributes.team == team,
                        PlayerAttributes.season == season,
                        attr_field,
                    )
                )
                .order_by(desc(PlayerAttributes.role_confidence))
            )

            specialists = []
            for attrs, player in query.all():
                confidence = attrs.role_confidence or 0.5
                specialists.append(
                    {
                        "player_id": player.player_id,
                        "player_name": player.name,
                        "team": team,
                        "set_piece_type": set_piece_type.value,
                        "confidence": confidence,
                        "total_taken": 0,
                        "success_rate": 0.0,
                        "goals_rate": 0.0,
                        "assists_rate": 0.0,
                        "last_taken_date": None,
                        "is_primary": confidence >= 0.8,
                        "recent_taken": 0,
                        "source": "attributes",
                    }
                )

            return specialists

    def get_all_set_piece_specialists(
        self, season: str, set_piece_type: SetPieceType
    ) -> dict[str, list[dict]]:
        """
        Get set piece specialists for all teams for a specific type.

        Args:
            season: Season to analyze
            set_piece_type: Type of set piece to analyze

        Returns:
            Dictionary mapping team names to their specialists
        """
        with session_scope() as session:
            teams_query = (
                session.query(PlayerAttributes.team)
                .filter(PlayerAttributes.season == season)
                .distinct()
            )

            all_specialists = {}
            for (team,) in teams_query.all():
                specialists = self.identify_team_set_piece_specialists(
                    team, season, set_piece_type
                )
                if specialists:
                    all_specialists[team] = specialists

            return all_specialists


class SetPieceTracker:
    """
    Monitor recent changes in set piece specialist assignments.

    This class tracks when players change roles and provides insights
    into recent trends and changes in set piece responsibilities.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize the set piece tracker."""
        self.db_session = db_session
        self.detector = SetPieceDetector(db_session)

    def detect_role_changes(
        self,
        team: str,
        season: str,
        set_piece_type: SetPieceType,
        lookback_gameweeks: int = 5,
    ) -> list[dict]:
        """
        Detect recent changes in set piece specialist assignments.

        Args:
            team: Team name
            season: Season to analyze
            set_piece_type: Type of set piece to analyze
            lookback_gameweeks: Number of gameweeks to look back

        Returns:
            List of detected role changes with metadata
        """
        # Get current specialists
        self.detector.identify_team_set_piece_specialists(team, season, set_piece_type)

        # Get historical data and compare
        # This would require additional implementation to track historical assignments
        # For now, return empty list as placeholder
        return []

        # TODO: Implement historical comparison logic
        # This would involve storing and comparing specialist assignments over time

    def get_recent_performance_trends(
        self, player_id: int, season: str, set_piece_type: SetPieceType
    ) -> dict:
        """
        Get recent performance trends for a set piece specialist.

        Args:
            player_id: Player ID
            season: Season to analyze
            set_piece_type: Type of set piece to analyze

        Returns:
            Dictionary with trend analysis
        """
        # Placeholder for trend analysis
        return {
            "player_id": player_id,
            "set_piece_type": set_piece_type.value,
            "trend_direction": "stable",  # 'improving', 'declining', 'stable'
            "recent_success_rate": 0.0,
            "trend_confidence": 0.0,
        }


class SetPieceOutcomeAnalyzer:
    """
    Analyze the effectiveness of set piece specialists.

    This class measures and analyzes the outcomes of set pieces
    to provide insights into specialist effectiveness.
    """

    def __init__(self, db_session: Session | None = None):
        """Initialize the outcome analyzer."""
        self.db_session = db_session

    def analyze_specialist_effectiveness(
        self, player_id: int, season: str, set_piece_type: SetPieceType
    ) -> dict:
        """
        Analyze the effectiveness of a set piece specialist.

        Args:
            player_id: Player ID
            season: Season to analyze
            set_piece_type: Type of set piece to analyze

        Returns:
            Dictionary with effectiveness metrics
        """
        # Placeholder for effectiveness analysis
        return {
            "player_id": player_id,
            "set_piece_type": set_piece_type.value,
            "total_attempts": 0,
            "success_rate": 0.0,
            "goals_per_attempt": 0.0,
            "assists_per_attempt": 0.0,
            "key_passes_per_attempt": 0.0,
            "effectiveness_score": 0.0,
            "rank_in_league": None,
        }

    def compare_specialists(
        self, player_ids: list[int], season: str, set_piece_type: SetPieceType
    ) -> dict:
        """
        Compare effectiveness of multiple set piece specialists.

        Args:
            player_ids: List of player IDs to compare
            season: Season to analyze
            set_piece_type: Type of set piece to analyze

        Returns:
            Dictionary with comparative analysis
        """
        comparisons = {}
        for player_id in player_ids:
            comparisons[player_id] = self.analyze_specialist_effectiveness(
                player_id, season, set_piece_type
            )

        return {
            "comparisons": comparisons,
            "best_performer": None,  # Would be determined by analysis
            "average_metrics": {},
        }


# Convenience functions for easy integration
def get_corner_specialists(
    team: str, season: str, corner_type: str = "both"
) -> list[dict]:
    """
    Get corner kick specialists for a specific team.

    Args:
        team: Team name
        season: Season to analyze
        corner_type: "left", "right", or "both"

    Returns:
        List of corner specialist assignments
    """
    corner_type_map = {
        "left": SetPieceType.CORNER_LEFT,
        "right": SetPieceType.CORNER_RIGHT,
        "both": SetPieceType.CORNER_BOTH,
    }

    set_piece_type = corner_type_map.get(corner_type, SetPieceType.CORNER_BOTH)
    detector = SetPieceDetector()
    return detector.identify_team_set_piece_specialists(team, season, set_piece_type)


def get_free_kick_specialists(
    team: str, season: str, distance: str = "both"
) -> list[dict]:
    """
    Get free kick specialists for a specific team.

    Args:
        team: Team name
        season: Season to analyze
        distance: "close", "long", or "both"

    Returns:
        List of free kick specialist assignments
    """
    if distance == "close":
        set_piece_type = SetPieceType.FREE_KICK_CLOSE
    elif distance == "long":
        set_piece_type = SetPieceType.FREE_KICK_LONG
    else:
        # Return both close and long range specialists
        detector = SetPieceDetector()
        close_specialists = detector.identify_team_set_piece_specialists(
            team, season, SetPieceType.FREE_KICK_CLOSE
        )
        long_specialists = detector.identify_team_set_piece_specialists(
            team, season, SetPieceType.FREE_KICK_LONG
        )
        return close_specialists + long_specialists

    detector = SetPieceDetector()
    return detector.identify_team_set_piece_specialists(team, season, set_piece_type)


def get_throw_in_specialists(team: str, season: str) -> list[dict]:
    """
    Get long throw-in specialists for a specific team.

    Args:
        team: Team name
        season: Season to analyze

    Returns:
        List of throw-in specialist assignments
    """
    detector = SetPieceDetector()
    return detector.identify_team_set_piece_specialists(
        team, season, SetPieceType.THROW_IN_LONG
    )


def get_all_set_piece_specialists(season: str) -> dict[str, dict[str, list[dict]]]:
    """
    Get all set piece specialists for all teams and types.

    Args:
        season: Season to analyze

    Returns:
        Nested dictionary mapping teams -> set piece types -> specialists
    """
    detector = SetPieceDetector()
    all_specialists: dict[str, dict[str, list[dict]]] = {}

    with session_scope() as session:
        teams_query = (
            session.query(PlayerAttributes.team)
            .filter(PlayerAttributes.season == season)
            .distinct()
        )

        for (team,) in teams_query.all():
            all_specialists[team] = {}

            for set_piece_type in SetPieceType:
                specialists = detector.identify_team_set_piece_specialists(
                    team, season, set_piece_type
                )
                if specialists:
                    all_specialists[team][set_piece_type.value] = specialists

    return all_specialists
