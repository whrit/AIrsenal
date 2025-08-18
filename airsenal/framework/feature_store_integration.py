"""
Feature Store Integration for AIrsenal Prediction Pipeline

This module provides integration between the new FeatureStore and the existing
prediction pipeline. It extends the current prediction utilities to leverage
cached features and provides backwards compatibility.

Key Integration Points:
1. Enhanced player data preparation using cached features
2. Feature-aware prediction computation
3. Automatic feature computation during DB updates
4. Migration utilities for existing features

Usage:
    # Enhanced prediction with feature store
    predictor = FeatureAwarePredictionUtils()
    predictions = predictor.calc_predicted_points_for_player_enhanced(
        player=player,
        fixture_goal_probs=fixture_probs,
        season=season,
        gw_range=[10, 11, 12]
    )

    # Migrate existing features to feature store
    migrator = FeatureMigrator()
    migrator.migrate_player_scores_to_features(season="2425")
"""

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.framework.feature_store import FeatureStore
from airsenal.framework.prediction_utils import (
    calc_predicted_points_for_player,
    fit_player_data,
    get_appearance_points,
    get_attacking_points,
    get_defending_points,
)
from airsenal.framework.schema import (
    ComputedFeature,
    FeatureTimeSeries,
    Fixture,
    Player,
    PlayerScore,
    session,
)
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_fixtures_for_player,
    get_recent_minutes_for_player,
    list_players,
)

logger = logging.getLogger(__name__)


class FeatureAwarePredictionUtils:
    """Enhanced prediction utilities that leverage the feature store."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession
        self.feature_store = FeatureStore(dbsession)

    def get_player_features_for_prediction(
        self,
        player_ids: list[int],
        season: str = CURRENT_SEASON,
        gameweek: int = NEXT_GAMEWEEK,
    ) -> dict[int, dict[str, Any]]:
        """Get all relevant features for player prediction."""

        # Define features needed for prediction
        feature_names = [
            "rolling_goals_5",
            "rolling_assists_5",
            "rolling_minutes_5",
            "rolling_points_5",
            "goals_form",
            "assists_form",
            "player_position",
            "player_team",
        ]

        # Get features from store
        features = self.feature_store.get_features(
            entity_type="player",
            entity_ids=player_ids,
            feature_names=feature_names,
            season=season,
            gameweek=gameweek,
        )

        # Enrich with traditional computed features for backwards compatibility
        for player_id in player_ids:
            if player_id not in features:
                features[player_id] = {}

            # Add traditional probability features if not available
            if "prob_score" not in features[player_id]:
                try:
                    # Get player position and fit traditional model
                    player = (
                        self.dbsession.query(Player)
                        .filter_by(player_id=player_id)
                        .first()
                    )
                    if player:
                        position = player.position(season)
                        if position:
                            df_player = fit_player_data(
                                position, season, gameweek, dbsession=self.dbsession
                            )
                            if player_id in df_player.index:
                                player_probs = df_player.loc[player_id]
                                features[player_id].update(
                                    {
                                        "prob_score": player_probs["prob_score"],
                                        "prob_assist": player_probs["prob_assist"],
                                        "prob_neither": player_probs["prob_neither"],
                                    }
                                )
                except Exception as e:
                    logger.warning(
                        f"Failed to get traditional features for player {player_id}: {e}"
                    )

        return features

    def calc_predicted_points_for_player_enhanced(
        self,
        player: Player,
        fixture_goal_probs: dict,
        season: str,
        gw_range: list[int],
        use_feature_store: bool = True,
        fallback_to_traditional: bool = True,
        tag: str = "",
    ) -> list:
        """Enhanced prediction calculation using feature store data."""

        if use_feature_store:
            try:
                return self._calc_with_feature_store(
                    player, fixture_goal_probs, season, gw_range, tag
                )
            except Exception as e:
                logger.warning(f"Feature store prediction failed for {player}: {e}")
                if not fallback_to_traditional:
                    raise

        # Fallback to traditional method
        logger.info(f"Using traditional prediction method for {player}")
        return calc_predicted_points_for_player(
            player=player,
            fixture_goal_probs=fixture_goal_probs,
            df_player={},  # Will be computed internally
            df_bonus=None,
            df_saves=None,
            df_cards=None,
            season=season,
            gw_range=gw_range,
            tag=tag,
            dbsession=self.dbsession,
        )

    def _calc_with_feature_store(
        self,
        player: Player,
        fixture_goal_probs: dict,
        season: str,
        gw_range: list[int],
        tag: str,
    ) -> list:
        """Calculate predictions using feature store data."""

        # Get player features
        features = self.get_player_features_for_prediction(
            player_ids=[player.player_id], season=season, gameweek=min(gw_range)
        )[player.player_id]

        position = features.get("player_position")
        team = features.get("player_team")

        if not position or not team:
            msg = f"Missing position or team for player {player}"
            raise ValueError(msg)

        # Get fixtures
        fixtures = get_fixtures_for_player(
            player, season, gw_range=gw_range, dbsession=self.dbsession
        )

        # Use recent minutes (traditional approach for now)
        recent_minutes = get_recent_minutes_for_player(
            player,
            num_match_to_use=len(gw_range),
            season=season,
            last_gw=min(gw_range) - 1,
            dbsession=self.dbsession,
        )

        if len(recent_minutes) == 0:
            msg = "No recent minutes data available"
            raise ValueError(msg)

        predictions = []

        for fixture in fixtures:
            gameweek = fixture.gameweek
            if gameweek is None:
                continue

            is_home = fixture.home_team == team
            opponent = fixture.away_team if is_home else fixture.home_team

            team_score_prob = fixture_goal_probs[fixture.fixture_id][team]
            team_concede_prob = fixture_goal_probs[fixture.fixture_id][opponent]

            # Enhanced point calculation using features
            points = 0.0

            if sum(recent_minutes) == 0 or player.is_injured_or_suspended(
                season, min(gw_range), gameweek
            ):
                points = 0.0
            else:
                # Calculate points for each recent minute scenario
                for mins in recent_minutes:
                    # Basic appearance points
                    points += get_appearance_points(mins)

                    # Attacking points using feature store probabilities or fallback
                    if all(
                        key in features
                        for key in ["prob_score", "prob_assist", "prob_neither"]
                    ):
                        # Use cached probabilities
                        player_prob = pd.Series(
                            {
                                "prob_score": features["prob_score"],
                                "prob_assist": features["prob_assist"],
                                "prob_neither": features["prob_neither"],
                            }
                        )
                        points += get_attacking_points(
                            position, mins, team_score_prob, player_prob
                        )
                    else:
                        # Use form-based estimation
                        goals_form = features.get("goals_form", 0.1)
                        assists_form = features.get("assists_form", 0.05)

                        # Convert form to probability approximation
                        estimated_prob = pd.Series(
                            {
                                "prob_score": goals_form,
                                "prob_assist": assists_form,
                                "prob_neither": max(0, 1 - goals_form - assists_form),
                            }
                        )
                        points += get_attacking_points(
                            position, mins, team_score_prob, estimated_prob
                        )

                    # Defending points
                    points += get_defending_points(position, mins, team_concede_prob)

                points /= len(recent_minutes)

            # Create prediction object (simplified - would need full PlayerPrediction)
            prediction_data = {
                "player": player,
                "fixture": fixture,
                "predicted_points": points,
                "tag": tag,
                "enhanced_features": features,
            }
            predictions.append(prediction_data)

        return predictions

    def batch_update_player_features(
        self,
        season: str = CURRENT_SEASON,
        gameweek_range: tuple[int, int] | None = None,
        position_filter: str | None = None,
    ) -> dict[str, int]:
        """Batch update all player features for a season."""

        if gameweek_range is None:
            gameweek_range = (1, NEXT_GAMEWEEK)

        # Get players to update
        if position_filter:
            players = list_players(
                position=position_filter,
                season=season,
                gameweek=gameweek_range[1],
                dbsession=self.dbsession,
            )
        else:
            players = list_players(
                season=season, gameweek=gameweek_range[1], dbsession=self.dbsession
            )

        player_ids = [p.player_id for p in players]

        # Update features using feature store batch computation
        feature_names = [
            "rolling_goals_5",
            "rolling_assists_5",
            "rolling_minutes_5",
            "rolling_points_5",
            "goals_form",
            "assists_form",
        ]

        stats = self.feature_store.compute_features_batch(
            feature_names=feature_names,
            entity_type="player",
            entity_ids=player_ids,
            season=season,
            gameweek_range=gameweek_range,
        )

        logger.info(
            f"Updated features for {len(player_ids)} players in season {season}"
        )
        return stats


class FeatureMigrator:
    """Migrates existing data to the feature store format."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession
        self.feature_store = FeatureStore(dbsession)

    def migrate_player_scores_to_timeseries(
        self, season: str = CURRENT_SEASON, metrics: list[str] | None = None
    ) -> int:
        """Migrate PlayerScore data to FeatureTimeSeries for rolling computations."""

        if metrics is None:
            metrics = ["goals", "assists", "minutes", "points", "bonus"]

        # Get all player scores for the season
        scores_query = (
            self.dbsession.query(PlayerScore)
            .join(Fixture)
            .filter(Fixture.season == season)
            .order_by(Fixture.gameweek)
        )

        migrated_count = 0

        for score in scores_query:
            # Check if already migrated
            existing = (
                self.dbsession.query(FeatureTimeSeries)
                .filter_by(
                    entity_type="player",
                    entity_id=score.player_id,
                    season=season,
                    gameweek=score.fixture.gameweek,
                )
                .first()
            )

            if existing:
                continue

            # Create time series entries for each metric
            for metric in metrics:
                value = getattr(score, metric, None)
                if value is not None:
                    ts_entry = FeatureTimeSeries(
                        entity_type="player",
                        entity_id=score.player_id,
                        metric_name=metric,
                        season=season,
                        gameweek=score.fixture.gameweek,
                        timestamp=score.fixture.date or datetime.now().isoformat(),
                        value=float(value),
                        context=json.dumps(
                            {
                                "opponent": score.opponent,
                                "player_team": score.player_team,
                                "fixture_id": score.fixture_id,
                            }
                        ),
                    )
                    self.dbsession.add(ts_entry)
                    migrated_count += 1

        self.dbsession.commit()
        logger.info(
            f"Migrated {migrated_count} time series entries for season {season}"
        )
        return migrated_count

    def compute_and_store_historical_features(
        self, seasons: list[str], feature_names: list[str] | None = None
    ) -> dict[str, int]:
        """Compute and store historical features for multiple seasons."""

        if feature_names is None:
            feature_names = [
                "rolling_goals_5",
                "rolling_assists_5",
                "rolling_minutes_5",
                "rolling_points_5",
            ]

        total_stats = {"total_computations": 0, "successful_computations": 0}

        for season in seasons:
            logger.info(f"Computing historical features for season {season}")

            # Get all players for the season
            players = list_players(season=season, gameweek=38, dbsession=self.dbsession)
            player_ids = [p.player_id for p in players]

            # Compute features for all gameweeks
            stats = self.feature_store.compute_features_batch(
                feature_names=feature_names,
                entity_type="player",
                entity_ids=player_ids,
                season=season,
                gameweek_range=(
                    6,
                    38,
                ),  # Start from GW6 to have enough history for rolling features
                chunk_size=50,
            )

            total_stats["total_computations"] += stats["total_computations"]
            total_stats["successful_computations"] += stats["successful_computations"]

        logger.info(f"Historical feature computation completed: {total_stats}")
        return total_stats


class FeatureStoreMonitor:
    """Monitoring and alerting for the feature store."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession
        self.feature_store = FeatureStore(dbsession)

    def check_feature_freshness(
        self, feature_names: list[str], max_age_hours: int = 24
    ) -> dict[str, dict[str, Any]]:
        """Check if features are fresh enough for prediction."""

        cutoff_time = datetime.now() - pd.Timedelta(hours=max_age_hours)
        cutoff_str = cutoff_time.isoformat()

        freshness_report = {}

        for feature_name in feature_names:
            feature_def = self.feature_store.registry.get_feature_definition(
                feature_name
            )

            # Count fresh vs stale features
            fresh_count = (
                self.dbsession.query(ComputedFeature)
                .filter(
                    ComputedFeature.feature_definition_id == feature_def.id,
                    ComputedFeature.computed_at > cutoff_str,
                )
                .count()
            )

            total_count = (
                self.dbsession.query(ComputedFeature)
                .filter(ComputedFeature.feature_definition_id == feature_def.id)
                .count()
            )

            freshness_report[feature_name] = {
                "fresh_count": fresh_count,
                "total_count": total_count,
                "freshness_ratio": fresh_count / total_count if total_count > 0 else 0,
                "is_fresh": fresh_count / total_count >= 0.8
                if total_count > 0
                else False,
            }

        return freshness_report

    def generate_feature_quality_report(self) -> dict[str, Any]:
        """Generate comprehensive feature quality report."""

        # Get all active features
        features = self.feature_store.registry.list_features()

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(features),
            "features": {},
            "cache_performance": self.feature_store.cache.get_stats(),
            "alerts": [],
        }

        for feature in features:
            try:
                stats = self.feature_store.get_feature_stats(feature.name)
                report["features"][feature.name] = stats

                # Check for potential issues
                if stats["value_stats"]["count"] == 0:
                    report["alerts"].append(
                        f"No computed values for feature {feature.name}"
                    )

                # Check for feature drift (comparing last two seasons)
                current_season = CURRENT_SEASON
                prev_season = str(int(current_season[:2]) - 1) + str(
                    int(current_season[2:]) - 1
                )

                drift_check = self.feature_store.validator.check_feature_drift(
                    feature_name=feature.name,
                    entity_type="player",
                    current_season=current_season,
                    reference_season=prev_season,
                )

                if drift_check["drift_detected"]:
                    report["alerts"].append(
                        f"Feature drift detected for {feature.name}: "
                        f"mean change {drift_check['mean_change']:.2%}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to generate stats for feature {feature.name}: {e}"
                )
                report["alerts"].append(
                    f"Failed to analyze feature {feature.name}: {e!s}"
                )

        return report


# Convenience functions for integration


def get_enhanced_player_features(
    player_ids: list[int], season: str = CURRENT_SEASON, gameweek: int = NEXT_GAMEWEEK
) -> pd.DataFrame:
    """Get player features as a pandas DataFrame for analysis."""

    predictor = FeatureAwarePredictionUtils()
    features_dict = predictor.get_player_features_for_prediction(
        player_ids=player_ids, season=season, gameweek=gameweek
    )

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(features_dict, orient="index")
    df.index.name = "player_id"
    return df


def migrate_historical_data(seasons: list[str] | None = None) -> None:
    """One-time migration of historical data to feature store."""

    if seasons is None:
        seasons = ["2223", "2324", "2425"]

    migrator = FeatureMigrator()

    # Migrate player scores to time series
    for season in seasons:
        migrator.migrate_player_scores_to_timeseries(season)

    # Compute historical features
    migrator.compute_and_store_historical_features(seasons)

    logger.info("Historical data migration completed")


def update_features_for_gameweek(
    season: str = CURRENT_SEASON, gameweek: int = NEXT_GAMEWEEK
) -> None:
    """Update features after a gameweek is completed."""

    predictor = FeatureAwarePredictionUtils()

    # Update features for the completed gameweek
    stats = predictor.batch_update_player_features(
        season=season, gameweek_range=(gameweek, gameweek)
    )

    logger.info(f"Updated features for gameweek {gameweek}: {stats}")
