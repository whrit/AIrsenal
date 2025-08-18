"""
Factory_boy factories for Feature Store components.

Provides realistic test data generation for feature definitions, computed features,
feature cache, and time-series data used in feature engineering pipelines.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Any

import factory

from airsenal.framework.schema import (
    ComputedFeature,
    FeatureCache,
    FeatureDefinition,
    FeatureTimeSeries,
)

from .fpl_data import PREMIER_LEAGUE_TEAMS, generate_season_gameweek


class FeatureDefinitionFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating FeatureDefinition instances with realistic feature metadata."""

    class Meta:
        model = FeatureDefinition
        sqlalchemy_session_persistence = "commit"

    name = factory.fuzzy.FuzzyChoice(
        [
            # Player features
            "rolling_goals_5",
            "rolling_assists_5",
            "rolling_points_5",
            "xg_form_3",
            "xa_form_3",
            "xgi_form_3",
            "minutes_per_game_5",
            "bonus_points_rate",
            "home_away_split",
            "big_team_performance",
            "fixture_difficulty_weighted_form",
            "penalty_conversion_rate",
            "set_piece_involvement",
            # Team features
            "team_attack_strength",
            "team_defense_strength",
            "team_goals_scored_5",
            "team_goals_conceded_5",
            "team_clean_sheet_rate",
            "team_home_advantage",
            "team_recent_form_5",
            "team_injury_impact",
            # Fixture features
            "fixture_importance",
            "derby_match_indicator",
            "rest_days_home",
            "rest_days_away",
            "historical_h2h_goals",
            "referee_card_tendency",
            # Advanced/External features
            "weather_impact",
            "crowd_atmosphere",
            "transfer_rumor_sentiment",
            "social_media_buzz",
            "betting_odds_implied_probability",
            "injury_risk_score",
            "fatigue_index",
        ]
    )

    version = factory.LazyFunction(
        lambda: f"v{random.randint(1, 10)}.{random.randint(0, 9)}"
    )

    feature_type = factory.LazyAttribute(lambda obj: _determine_feature_type(obj.name))

    data_type = factory.LazyAttribute(lambda obj: _determine_data_type(obj.name))

    description = factory.LazyAttribute(
        lambda obj: _generate_feature_description(obj.name)
    )

    computation_logic = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_computation_logic(obj.name, obj.feature_type))
    )

    dependencies = factory.LazyAttribute(lambda obj: _generate_dependencies(obj.name))

    is_active = factory.fuzzy.FuzzyChoice([True, False], weights=[9, 1])  # 90% active

    created_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
    )

    updated_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
    )


class ComputedFeatureFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating ComputedFeature instances with realistic computed values."""

    class Meta:
        model = ComputedFeature
        sqlalchemy_session_persistence = "commit"

    feature_definition = factory.SubFactory(FeatureDefinitionFactory)

    entity_type = factory.LazyAttribute(lambda obj: obj.feature_definition.feature_type)

    entity_id = factory.LazyAttribute(lambda obj: _generate_entity_id(obj.entity_type))

    gameweek = factory.LazyAttribute(
        lambda obj: random.randint(1, 38)
        if _needs_gameweek(obj.feature_definition.name)
        else None
    )

    season = factory.LazyFunction(lambda: generate_season_gameweek()[0])

    value = factory.LazyAttribute(
        lambda obj: _generate_feature_value(
            obj.feature_definition.name, obj.feature_definition.data_type
        )
    )

    string_value = factory.LazyAttribute(
        lambda obj: _generate_string_value(
            obj.feature_definition.name, obj.feature_definition.data_type
        )
    )

    computed_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(hours=random.randint(0, 48))).isoformat()
    )

    ttl_expires_at = factory.LazyFunction(
        lambda: (
            datetime.now() + timedelta(hours=random.randint(24, 168))
        ).isoformat()  # 1-7 days
    )


class FeatureCacheFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating FeatureCache instances for hot feature serving."""

    class Meta:
        model = FeatureCache
        sqlalchemy_session_persistence = "commit"

    feature_name = factory.fuzzy.FuzzyChoice(
        [
            "rolling_goals_5",
            "xg_form_3",
            "team_attack_strength",
            "fixture_difficulty_weighted_form",
            "penalty_conversion_rate",
            "team_recent_form_5",
            "rest_days_home",
            "injury_risk_score",
        ]
    )

    entity_type = factory.LazyAttribute(
        lambda obj: _determine_feature_type(obj.feature_name)
    )

    entity_id = factory.LazyAttribute(lambda obj: _generate_entity_id(obj.entity_type))

    cache_key = factory.LazyAttribute(
        lambda obj: f"{obj.feature_name}:{obj.entity_type}:{obj.entity_id}:gw{random.randint(1, 38)}"
    )

    value = factory.LazyAttribute(
        lambda obj: _generate_feature_value(obj.feature_name, "float")
    )

    string_value = factory.LazyAttribute(
        lambda obj: _generate_string_value(obj.feature_name, "string")
        if random.random() < 0.2
        else None
    )

    cached_at = factory.LazyFunction(
        lambda: (
            datetime.now() - timedelta(minutes=random.randint(0, 1440))
        ).isoformat()  # Last 24 hours
    )

    expires_at = factory.LazyFunction(
        lambda: (
            datetime.now() + timedelta(hours=random.randint(1, 72))
        ).isoformat()  # 1-72 hours
    )

    hit_count = factory.fuzzy.FuzzyInteger(0, 1000)


class FeatureTimeSeriesFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating FeatureTimeSeries instances for time-based feature computation."""

    class Meta:
        model = FeatureTimeSeries
        sqlalchemy_session_persistence = "commit"

    entity_type = factory.fuzzy.FuzzyChoice(["player", "team", "fixture"])

    entity_id = factory.LazyAttribute(lambda obj: _generate_entity_id(obj.entity_type))

    metric_name = factory.fuzzy.FuzzyChoice(
        [
            # Player metrics
            "goals",
            "assists",
            "minutes",
            "points",
            "bonus",
            "saves",
            "xg",
            "xa",
            "shots",
            "key_passes",
            "tackles",
            "interceptions",
            "clean_sheets",
            "yellow_cards",
            "red_cards",
            # Team metrics
            "goals_scored",
            "goals_conceded",
            "shots_on_target",
            "possession",
            "corners",
            "fouls",
            "offsides",
            # Fixture metrics
            "attendance",
            "total_goals",
            "total_cards",
            "total_corners",
        ]
    )

    season = factory.LazyFunction(lambda: generate_season_gameweek()[0])

    gameweek = factory.fuzzy.FuzzyInteger(1, 38)

    timestamp = factory.LazyAttribute(
        lambda obj: _generate_gameweek_timestamp(obj.season, obj.gameweek)
    )

    value = factory.LazyAttribute(
        lambda obj: _generate_timeseries_value(obj.metric_name, obj.entity_type)
    )

    context = factory.LazyAttribute(
        lambda obj: json.dumps(
            _generate_timeseries_context(obj.metric_name, obj.entity_type)
        )
    )


# Specialized factories for specific feature scenarios
class HighValueFeatureCacheFactory(FeatureCacheFactory):
    """Factory for frequently accessed, high-value features."""

    feature_name = factory.fuzzy.FuzzyChoice(
        ["rolling_goals_5", "xg_form_3", "fixture_difficulty_weighted_form"]
    )

    hit_count = factory.fuzzy.FuzzyInteger(500, 5000)  # High hit count

    expires_at = factory.LazyFunction(
        lambda: (datetime.now() + timedelta(hours=1)).isoformat()  # Short TTL
    )


class PlayerFormTimeSeriesFactory(FeatureTimeSeriesFactory):
    """Factory for player form-related time series data."""

    entity_type = "player"

    metric_name = factory.fuzzy.FuzzyChoice(
        ["goals", "assists", "points", "minutes", "xg", "xa", "bonus"]
    )

    @factory.post_generation
    def create_progression(obj, create, extracted, **kwargs):
        """Create a realistic progression of values over multiple gameweeks."""
        if not create:
            return

        # Create 5 more gameweeks of data for this player/metric
        base_value = obj.value
        for i in range(1, 6):
            if obj.gameweek + i <= 38:
                FeatureTimeSeriesFactory(
                    entity_type="player",
                    entity_id=obj.entity_id,
                    metric_name=obj.metric_name,
                    season=obj.season,
                    gameweek=obj.gameweek + i,
                    value=_evolve_timeseries_value(base_value, obj.metric_name, i),
                )


class TeamPerformanceTimeSeriesFactory(FeatureTimeSeriesFactory):
    """Factory for team performance time series data."""

    entity_type = "team"

    metric_name = factory.fuzzy.FuzzyChoice(
        ["goals_scored", "goals_conceded", "shots_on_target", "possession"]
    )


# Helper functions for realistic data generation
def _determine_feature_type(feature_name: str) -> str:
    """Determine feature type based on feature name."""
    if any(term in feature_name for term in ["team_", "defense", "attack"]):
        return "team"
    if any(term in feature_name for term in ["fixture_", "h2h", "referee"]):
        return "fixture"
    if any(term in feature_name for term in ["weather", "odds", "external"]):
        return "external"
    return "player"


def _determine_data_type(feature_name: str) -> str:
    """Determine data type based on feature name."""
    if any(
        term in feature_name for term in ["rate", "percentage", "ratio", "strength"]
    ):
        return "float"
    if any(term in feature_name for term in ["indicator", "match", "derby"]):
        return "boolean"
    if any(term in feature_name for term in ["sentiment", "tendency", "category"]):
        return "string"
    return "float"  # Default


def _generate_feature_description(feature_name: str) -> str:
    """Generate realistic feature description."""
    descriptions = {
        "rolling_goals_5": "Rolling average of goals scored over last 5 games",
        "xg_form_3": "Expected goals form over last 3 games",
        "team_attack_strength": "Team's attacking strength rating based on recent performance",
        "fixture_difficulty_weighted_form": "Player form weighted by fixture difficulty",
        "penalty_conversion_rate": "Historical penalty conversion rate for player",
        "rest_days_home": "Number of rest days for home team before fixture",
        "injury_risk_score": "Machine learning model prediction of injury risk",
        "derby_match_indicator": "Boolean flag indicating if fixture is a local derby",
        "weather_impact": "Expected impact of weather conditions on match",
        "betting_odds_implied_probability": "Implied probability from betting market odds",
    }

    if feature_name in descriptions:
        return descriptions[feature_name]

    # Generate generic description
    parts = feature_name.split("_")
    if len(parts) >= 2:
        return f"Computed feature for {' '.join(parts[:-1])} analysis"

    return f"Feature computation for {feature_name.replace('_', ' ')}"


def _generate_computation_logic(feature_name: str, feature_type: str) -> dict[str, Any]:
    """Generate realistic computation logic configuration."""
    base_logic = {
        "computation_type": "rolling_average"
        if "rolling" in feature_name
        else "aggregation",
        "update_frequency": "daily",
        "lookback_window": 5
        if "5" in feature_name
        else (3 if "3" in feature_name else 10),
    }

    if "rolling" in feature_name:
        base_logic.update(
            {"window_type": "games", "min_periods": 1, "aggregation_func": "mean"}
        )
    elif "form" in feature_name:
        base_logic.update(
            {"weight_recent": True, "decay_factor": 0.9, "normalize": True}
        )
    elif "strength" in feature_name:
        base_logic.update(
            {
                "components": ["attack", "defense", "midfield"],
                "weighting": {"recent": 0.6, "season": 0.4},
                "home_advantage": 0.1,
            }
        )

    return base_logic


def _generate_dependencies(feature_name: str) -> str:
    """Generate realistic feature dependencies."""
    dependency_map = {
        "rolling_goals_5": "goals,minutes",
        "xg_form_3": "xg,minutes,opponent_strength",
        "fixture_difficulty_weighted_form": "points,fixture_difficulty,team_strength",
        "team_attack_strength": "goals_scored,shots,key_passes,team_rating",
        "penalty_conversion_rate": "penalties_taken,penalties_scored",
        "injury_risk_score": "age,position,minutes,previous_injuries,workload",
        "rest_days_home": "fixture_date,previous_fixture_date",
    }

    if feature_name in dependency_map:
        return dependency_map[feature_name]

    # Generate generic dependencies
    if "team" in feature_name:
        return "team_stats,recent_results"
    if "player" in feature_name:
        return "player_stats,minutes,position"
    return "base_stats"


def _generate_entity_id(entity_type: str) -> int:
    """Generate realistic entity ID based on type."""
    if entity_type == "player":
        return random.randint(1, 500)  # Player IDs
    if entity_type == "team":
        return random.randint(1, 20)  # Team IDs
    if entity_type == "fixture":
        return random.randint(1, 380)  # Fixture IDs (38 gameweeks * 10 matches)
    return random.randint(1, 100)  # Generic entity ID


def _needs_gameweek(feature_name: str) -> bool:
    """Determine if feature needs gameweek context."""
    non_gameweek_features = [
        "penalty_conversion_rate",
        "historical_h2h",
        "team_home_advantage",
    ]
    return not any(feature in feature_name for feature in non_gameweek_features)


def _generate_feature_value(feature_name: str, data_type: str) -> float:
    """Generate realistic feature value based on name and type."""
    if data_type == "boolean":
        return float(random.choice([0, 1]))

    # Define realistic ranges for different feature types
    value_ranges = {
        "goals": (0.0, 2.0),
        "assists": (0.0, 1.5),
        "points": (0.0, 15.0),
        "xg": (0.0, 1.2),
        "xa": (0.0, 0.8),
        "form": (0.0, 10.0),
        "strength": (0.3, 1.2),
        "rate": (0.0, 1.0),
        "difficulty": (1.0, 5.0),
        "risk": (0.0, 1.0),
        "impact": (-0.5, 0.5),
        "probability": (0.0, 1.0),
        "days": (1, 10),
    }

    for key, (min_val, max_val) in value_ranges.items():
        if key in feature_name:
            return round(random.uniform(min_val, max_val), 2)

    # Default range
    return round(random.uniform(0.0, 5.0), 2)


def _generate_string_value(feature_name: str, data_type: str) -> str:
    """Generate string value for string-type features."""
    if data_type != "string":
        return None

    string_values = {
        "sentiment": ["positive", "negative", "neutral"],
        "tendency": ["aggressive", "lenient", "average"],
        "category": ["high", "medium", "low"],
        "weather": ["sunny", "rainy", "cloudy", "windy"],
        "atmosphere": ["excellent", "good", "average", "poor"],
    }

    for key, values in string_values.items():
        if key in feature_name:
            return random.choice(values)

    return random.choice(["A", "B", "C"])


def _generate_gameweek_timestamp(season: str, gameweek: int) -> str:
    """Generate realistic timestamp for a gameweek."""
    # Premier League season starts in mid-August
    season_start = datetime(2024, 8, 17)  # Approximate start
    gameweek_date = season_start + timedelta(weeks=gameweek - 1)

    # Add some randomness for match day (weekend)
    match_day = gameweek_date + timedelta(days=random.randint(0, 6))
    match_time = match_day.replace(
        hour=random.choice([12, 15, 17]),  # Typical kick-off times
        minute=random.choice([0, 30]),
        second=0,
        microsecond=0,
    )

    return match_time.isoformat()


def _generate_timeseries_value(metric_name: str, entity_type: str) -> float:
    """Generate realistic time series value based on metric and entity type."""
    value_ranges = {
        # Player metrics
        "goals": (0, 3),
        "assists": (0, 2),
        "minutes": (0, 90),
        "points": (0, 20),
        "bonus": (0, 3),
        "saves": (0, 10),
        "xg": (0.0, 2.0),
        "xa": (0.0, 1.5),
        "shots": (0, 8),
        "key_passes": (0, 10),
        "tackles": (0, 8),
        "interceptions": (0, 6),
        # Team metrics
        "goals_scored": (0, 5),
        "goals_conceded": (0, 4),
        "shots_on_target": (2, 12),
        "possession": (30, 80),
        "corners": (0, 15),
        "fouls": (5, 25),
        # Fixture metrics
        "attendance": (20000, 80000),
        "total_goals": (0, 8),
        "total_cards": (0, 8),
        "total_corners": (0, 20),
    }

    if metric_name in value_ranges:
        min_val, max_val = value_ranges[metric_name]
        if isinstance(min_val, int):
            return float(random.randint(min_val, max_val))
        return round(random.uniform(min_val, max_val), 2)

    return round(random.uniform(0.0, 5.0), 2)


def _generate_timeseries_context(metric_name: str, entity_type: str) -> dict[str, Any]:
    """Generate contextual metadata for time series data."""
    context = {
        "data_source": "fpl_api",
        "confidence": round(random.uniform(0.8, 1.0), 2),
    }

    if entity_type == "player":
        context.update(
            {
                "position": random.choice(["GK", "DEF", "MID", "FWD"]),
                "team": random.choice(list(PREMIER_LEAGUE_TEAMS.keys())),
                "home_away": random.choice(["home", "away"]),
            }
        )
    elif entity_type == "team":
        context.update(
            {
                "opponent": random.choice(list(PREMIER_LEAGUE_TEAMS.keys())),
                "venue": random.choice(["home", "away"]),
                "competition": "premier_league",
            }
        )
    elif entity_type == "fixture":
        context.update(
            {
                "kickoff_time": random.choice(["12:30", "15:00", "17:30", "20:00"]),
                "referee": f"ref_{random.randint(1, 20)}",
                "weather": random.choice(["sunny", "cloudy", "rainy"]),
            }
        )

    return context


def _evolve_timeseries_value(
    base_value: float, metric_name: str, week_offset: int
) -> float:
    """Evolve a time series value realistically over time."""
    # Add some trend and random variation
    trend = random.uniform(-0.1, 0.1) * week_offset
    noise = random.uniform(-0.5, 0.5)

    new_value = base_value + trend + noise

    # Apply realistic bounds
    if metric_name in ["goals", "assists", "points"]:
        new_value = max(0.0, min(20.0, new_value))
    elif metric_name in ["minutes"]:
        new_value = max(0.0, min(90.0, new_value))
    else:
        new_value = max(0.0, new_value)

    return round(new_value, 2)


# Batch creation functions for testing scenarios
def create_feature_computation_pipeline(session) -> dict[str, list]:
    """Create a complete feature computation pipeline scenario."""
    # Create feature definitions
    definitions = []
    for feature_name in ["rolling_goals_5", "xg_form_3", "team_attack_strength"]:
        definition = FeatureDefinitionFactory(session=session, name=feature_name)
        definitions.append(definition)

    # Create computed features
    computed_features = []
    for definition in definitions:
        for _ in range(10):  # 10 computed values per feature
            computed = ComputedFeatureFactory(
                session=session, feature_definition=definition
            )
            computed_features.append(computed)

    # Create cache entries
    cache_entries = []
    for definition in definitions:
        for _ in range(5):  # 5 cache entries per feature
            cache = FeatureCacheFactory(session=session, feature_name=definition.name)
            cache_entries.append(cache)

    return {
        "definitions": definitions,
        "computed_features": computed_features,
        "cache_entries": cache_entries,
    }


def create_player_feature_timeline(
    session, player_id: int, season: str
) -> list[FeatureTimeSeries]:
    """Create a complete season timeline for a player's features."""
    metrics = ["goals", "assists", "points", "minutes", "xg", "xa"]
    timeline = []

    for gameweek in range(1, 39):  # Full season
        for metric in metrics:
            entry = FeatureTimeSeriesFactory(
                session=session,
                entity_type="player",
                entity_id=player_id,
                metric_name=metric,
                season=season,
                gameweek=gameweek,
            )
            timeline.append(entry)

    return timeline


def create_feature_performance_test_data(session) -> dict[str, Any]:
    """Create data for testing feature computation performance."""
    # High-volume cache scenario
    cache_entries = []
    for _ in range(1000):  # 1000 cache entries
        cache = HighValueFeatureCacheFactory(session=session)
        cache_entries.append(cache)

    # Time series bulk data
    timeseries_entries = []
    for player_id in range(1, 51):  # 50 players
        for gameweek in range(1, 11):  # 10 gameweeks
            entry = PlayerFormTimeSeriesFactory(
                session=session,
                entity_id=player_id,
                gameweek=gameweek,
            )
            timeseries_entries.append(entry)

    return {
        "cache_entries": cache_entries,
        "timeseries_entries": timeseries_entries,
    }
