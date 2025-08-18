"""
Enhanced Prediction Utils with Home/Away Adjustments

This module extends the existing prediction utilities to integrate home/away
adjustments for improved prediction accuracy.

Key enhancements:
- Home/away adjustments applied to player predictions
- Team-level venue effects integration
- Integration with existing prediction pipeline
- Backward compatibility with original prediction system

Author: AIrsenal Team
"""

import numpy as np
from sqlalchemy.orm.session import Session

from airsenal.framework.home_away_adjustment import (
    create_home_away_adjuster,
)
from airsenal.framework.logging_config import get_logger
from airsenal.framework.player_model import ConjugatePlayerModel, NumpyroPlayerModel
from airsenal.framework.prediction_utils import (
    calc_predicted_points_for_player as original_calc_predicted_points_for_player,
)
from airsenal.framework.prediction_utils import (
    fit_player_data,
)
from airsenal.framework.schema import Player, PlayerPrediction
from airsenal.framework.utils import (
    list_players,
    session,
)

logger = get_logger(__name__)


def calc_predicted_points_for_player_with_venue_adjustment(
    player: Player,
    fixture_goal_probs: dict,
    df_player: dict,
    df_bonus=None,
    df_saves=None,
    df_cards=None,
    season: str = "2425",
    gw_range: list | None = None,
    tag: str = "default",
    fixtures_behind: int = 5,
    dbsession: Session = session,
    apply_venue_adjustments: bool = True,
    venue_adjustment_weight: float = 1.0,
) -> list[PlayerPrediction]:
    """
    Enhanced version of calc_predicted_points_for_player with home/away adjustments.

    This function extends the original prediction calculation to include venue-specific
    adjustments based on team and player performance patterns.

    Args:
        player: Player object
        fixture_goal_probs: Goal probabilities for fixtures
        df_player: Player model data
        df_bonus: Bonus points data
        df_saves: Save points data
        df_cards: Card points data
        season: Season string
        gw_range: Range of gameweeks to predict
        tag: Prediction tag for database
        fixtures_behind: Number of previous fixtures to consider
        dbsession: Database session
        apply_venue_adjustments: Whether to apply home/away adjustments
        venue_adjustment_weight: Weight for venue adjustments (0-1)

    Returns:
        List of PlayerPrediction objects with venue adjustments applied
    """
    if gw_range is None:
        gw_range = [1]

    # Get base predictions using original method
    base_predictions = original_calc_predicted_points_for_player(
        player=player,
        fixture_goal_probs=fixture_goal_probs,
        df_player=df_player,
        df_bonus=df_bonus,
        df_saves=df_saves,
        df_cards=df_cards,
        season=season,
        gw_range=gw_range,
        tag=tag,
        fixtures_behind=fixtures_behind,
        dbsession=dbsession,
    )

    # If venue adjustments are disabled, return base predictions
    if not apply_venue_adjustments:
        return base_predictions

    # Initialize home/away adjuster
    try:
        adjuster = create_home_away_adjuster(dbsession)

        # Get player's team for the season
        team = player.team(season, min(gw_range))
        if not team:
            logger.warning(
                f"No team found for {player.name} in {season}, skipping venue adjustments"
            )
            return base_predictions

        # Apply venue adjustments to each prediction
        enhanced_predictions = []

        for prediction in base_predictions:
            try:
                # Determine if player is playing at home
                fixture = prediction.fixture
                is_home = fixture.home_team == team

                # Apply combined home/away adjustment
                adjustment_result = adjuster.apply_combined_adjustment(
                    base_prediction=prediction.predicted_points,
                    team=team,
                    player_id=player.player_id,
                    is_home=is_home,
                    season=season,
                    gameweek=fixture.gameweek,
                    prediction_type="points",
                )

                # Apply adjustment weight
                final_adjustment = (
                    venue_adjustment_weight * adjustment_result["final_prediction"]
                    + (1 - venue_adjustment_weight) * prediction.predicted_points
                )

                # Create new prediction with adjusted points
                enhanced_prediction = PlayerPrediction()
                enhanced_prediction.player = prediction.player
                enhanced_prediction.fixture = prediction.fixture
                enhanced_prediction.predicted_points = final_adjustment
                enhanced_prediction.tag = f"{tag}_venue_adjusted"

                enhanced_predictions.append(enhanced_prediction)

                # Log significant adjustments
                adjustment_impact = final_adjustment - prediction.predicted_points
                if abs(adjustment_impact) > 0.5:  # Log if adjustment > 0.5 points
                    logger.info(
                        f"Significant venue adjustment for {player.name}: "
                        f"{prediction.predicted_points:.2f} -> {final_adjustment:.2f} "
                        f"({'home' if is_home else 'away'}) vs {fixture.away_team if is_home else fixture.home_team}"
                    )

            except Exception as e:
                logger.error(f"Error applying venue adjustment for {player.name}: {e}")
                # Fall back to base prediction if adjustment fails
                enhanced_predictions.append(prediction)

        return enhanced_predictions

    except Exception as e:
        logger.error(f"Error initializing venue adjustments: {e}")
        return base_predictions


def calc_predicted_points_for_pos_with_venue_adjustment(
    pos: str,
    fixture_goal_probs: dict,
    df_bonus=None,
    df_saves=None,
    df_cards=None,
    season: str = "2425",
    gw_range: list | None = None,
    tag: str = "default",
    model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    dbsession: Session = session,
    apply_venue_adjustments: bool = True,
    venue_adjustment_weight: float = 1.0,
) -> dict[int, list[PlayerPrediction]]:
    """
    Calculate points predictions for all players in a position with venue adjustments.

    Args:
        pos: Position string ('GK', 'DEF', 'MID', 'FWD')
        fixture_goal_probs: Goal probabilities for fixtures
        df_bonus: Bonus points data
        df_saves: Save points data
        df_cards: Card points data
        season: Season string
        gw_range: Range of gameweeks to predict
        tag: Prediction tag
        model: Player model to use
        dbsession: Database session
        apply_venue_adjustments: Whether to apply venue adjustments
        venue_adjustment_weight: Weight for venue adjustments

    Returns:
        Dictionary mapping player_id to list of predictions
    """
    if gw_range is None:
        gw_range = [1]

    # Fit player data for position
    df_player = {pos: fit_player_data(pos, season, min(gw_range), model, dbsession)}

    # Get all players for position
    players = list_players(
        position=pos, season=season, gameweek=min(gw_range), dbsession=dbsession
    )

    results = {}

    for player in players:
        try:
            predictions = calc_predicted_points_for_player_with_venue_adjustment(
                player=player,
                fixture_goal_probs=fixture_goal_probs,
                df_player=df_player,
                df_bonus=df_bonus,
                df_saves=df_saves,
                df_cards=df_cards,
                season=season,
                gw_range=gw_range,
                tag=tag,
                dbsession=dbsession,
                apply_venue_adjustments=apply_venue_adjustments,
                venue_adjustment_weight=venue_adjustment_weight,
            )
            results[player.player_id] = predictions

        except Exception as e:
            logger.error(f"Error calculating predictions for {player.name}: {e}")
            # Fall back to original method
            try:
                results[player.player_id] = original_calc_predicted_points_for_player(
                    player=player,
                    fixture_goal_probs=fixture_goal_probs,
                    df_player=df_player,
                    df_bonus=df_bonus,
                    df_saves=df_saves,
                    df_cards=df_cards,
                    season=season,
                    gw_range=gw_range,
                    tag=tag,
                    dbsession=dbsession,
                )
            except Exception as fallback_error:
                logger.error(
                    f"Fallback prediction also failed for {player.name}: {fallback_error}"
                )
                results[player.player_id] = []

    return results


def get_venue_adjusted_team_multipliers(
    team: str, season: str, gameweek: int, dbsession: Session = session
) -> dict[str, float]:
    """
    Get home/away multipliers for a team's expected performance.

    Args:
        team: Team name
        season: Season string
        gameweek: Current gameweek
        dbsession: Database session

    Returns:
        Dictionary with home_multiplier and away_multiplier
    """
    try:
        adjuster = create_home_away_adjuster(dbsession)
        return adjuster.get_venue_multipliers(team, season, gameweek)
    except Exception as e:
        logger.error(f"Error getting venue multipliers for {team}: {e}")
        return {
            "home_multiplier": 1.0,
            "away_multiplier": 1.0,
            "advantage": 0.0,
            "confidence": 0.0,
        }


def apply_venue_adjustment_to_existing_prediction(
    prediction: PlayerPrediction,
    season: str,
    dbsession: Session = session,
    adjustment_weight: float = 1.0,
) -> float:
    """
    Apply venue adjustment to an existing prediction.

    Args:
        prediction: Existing PlayerPrediction object
        season: Season string
        dbsession: Database session
        adjustment_weight: Weight for the adjustment (0-1)

    Returns:
        Adjusted prediction value
    """
    try:
        # Get player's team
        player = prediction.player
        team = player.team(season, prediction.fixture.gameweek)

        if not team:
            return prediction.predicted_points

        # Determine venue
        is_home = prediction.fixture.home_team == team

        # Apply adjustment
        adjuster = create_home_away_adjuster(dbsession)
        adjustment_result = adjuster.apply_combined_adjustment(
            base_prediction=prediction.predicted_points,
            team=team,
            player_id=player.player_id,
            is_home=is_home,
            season=season,
            gameweek=prediction.fixture.gameweek,
            prediction_type="points",
        )

        # Apply adjustment weight
        return (
            adjustment_weight * adjustment_result["final_prediction"]
            + (1 - adjustment_weight) * prediction.predicted_points
        )

    except Exception as e:
        logger.error(f"Error applying venue adjustment to prediction: {e}")
        return prediction.predicted_points


def calculate_venue_impact_summary(
    predictions: list[PlayerPrediction], season: str, dbsession: Session = session
) -> dict[str, float]:
    """
    Calculate summary statistics for venue impact on predictions.

    Args:
        predictions: List of PlayerPrediction objects
        season: Season string
        dbsession: Database session

    Returns:
        Dictionary with venue impact statistics
    """
    try:
        adjuster = create_home_away_adjuster(dbsession)

        home_impacts = []
        away_impacts = []

        for prediction in predictions:
            player = prediction.player
            team = player.team(season, prediction.fixture.gameweek)

            if not team:
                continue

            is_home = prediction.fixture.home_team == team

            adjustment_result = adjuster.apply_combined_adjustment(
                base_prediction=prediction.predicted_points,
                team=team,
                player_id=player.player_id,
                is_home=is_home,
                season=season,
                gameweek=prediction.fixture.gameweek,
                prediction_type="points",
            )

            impact = adjustment_result["total_impact"]

            if is_home:
                home_impacts.append(impact)
            else:
                away_impacts.append(impact)

        return {
            "home_avg_impact": np.mean(home_impacts) if home_impacts else 0.0,
            "away_avg_impact": np.mean(away_impacts) if away_impacts else 0.0,
            "home_max_impact": max(home_impacts) if home_impacts else 0.0,
            "away_max_impact": max(away_impacts) if away_impacts else 0.0,
            "home_predictions": len(home_impacts),
            "away_predictions": len(away_impacts),
            "total_predictions": len(predictions),
        }

    except Exception as e:
        logger.error(f"Error calculating venue impact summary: {e}")
        return {
            "home_avg_impact": 0.0,
            "away_avg_impact": 0.0,
            "home_max_impact": 0.0,
            "away_max_impact": 0.0,
            "home_predictions": 0,
            "away_predictions": 0,
            "total_predictions": len(predictions),
        }


# Backward compatibility aliases
calc_predicted_points_for_player_enhanced = (
    calc_predicted_points_for_player_with_venue_adjustment
)
calc_predicted_points_for_pos_enhanced = (
    calc_predicted_points_for_pos_with_venue_adjustment
)
