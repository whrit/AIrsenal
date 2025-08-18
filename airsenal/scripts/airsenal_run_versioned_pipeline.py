"""
Enhanced AIrsenal Pipeline with Model Versioning Support

This script extends the standard AIrsenal pipeline to support model versioning,
A/B testing, and production model management.

Usage:
    airsenal_run_versioned_pipeline --weeks_ahead 3 --use-versioned-models
    airsenal_run_versioned_pipeline --ab-test control_v1.0.0 treatment_v1.1.0
"""

import multiprocessing
import sys
import warnings

import click
from sqlalchemy.orm.session import Session
from tqdm import TqdmWarning

from airsenal.framework.ab_testing import ABTestManager, ExperimentConfig
from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.schema import session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_past_seasons,
)
from airsenal.framework.versioned_prediction_utils import (
    VersionedPredictionManager,
)
from airsenal.scripts.fill_db_init import check_clean_db, make_init_db
from airsenal.scripts.fill_predictedscore_table import (
    get_top_predicted_points,
    make_predictedscore_table,
)
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.set_lineup import set_lineup
from airsenal.scripts.update_db import update_db

# Suppress warnings
warnings.filterwarnings("ignore", category=TqdmWarning)


@click.command("airsenal_run_versioned_pipeline")
@click.option("--num_thread", type=int, help="No. of threads to use for pipeline run")
@click.option(
    "--weeks_ahead", type=int, default=3, help="No of weeks to use for pipeline run"
)
@click.option(
    "--fpl_team_id", type=int, required=False, help="fpl team id for pipeline run"
)
@click.option(
    "--clean", is_flag=True, help="If set, delete and recreate the AIrsenal database"
)
@click.option(
    "--apply_transfers",
    is_flag=True,
    help="If set, go ahead and make the transfers via the API.",
)
# Model versioning options
@click.option(
    "--use-versioned-models", is_flag=True, help="Use model versioning system"
)
@click.option("--use-production-models", is_flag=True, help="Use production models")
@click.option(
    "--player-model-versions",
    help="JSON string of player model versions (e.g., {'fwd': 'v1.0.0'})",
)
@click.option("--team-model-version", help="Specific team model version to use")
@click.option(
    "--train-and-register", is_flag=True, help="Train new models and register them"
)
@click.option("--model-version", help="Version string for newly trained models")
# A/B testing options
@click.option(
    "--ab-test",
    nargs=2,
    metavar=("CONTROL_VERSION", "TREATMENT_VERSION"),
    help="Run A/B test with control and treatment versions",
)
@click.option("--experiment-name", help="Name for the A/B test experiment")
@click.option(
    "--traffic-split", type=float, default=0.5, help="Traffic split for A/B test"
)
# Standard pipeline options
@click.option(
    "--wildcard_week", type=int, help="Play wildcard in specified week", default=-1
)
@click.option(
    "--free_hit_week", type=int, help="Play free hit in specified week", default=-1
)
@click.option(
    "--triple_captain_week",
    type=int,
    help="Play triple captain in specified week",
    default=-1,
)
@click.option(
    "--bench_boost_week",
    type=int,
    help="Play bench boost in specified week",
    default=-1,
)
@click.option(
    "--n_previous", help="Number of past seasons to include", type=int, default=3
)
@click.option(
    "--no_current_season", help="Exclude current season from database", is_flag=True
)
@click.option("--team_model", help="Team model type", default="extended")
@click.option("--player_model", help="Player model type", default="conjugate")
@click.option("--season", help="Season to run pipeline for", default=CURRENT_SEASON)
def main(
    num_thread: int | None,
    weeks_ahead: int,
    fpl_team_id: int | None,
    clean: bool,
    apply_transfers: bool,
    # Model versioning options
    use_versioned_models: bool,
    use_production_models: bool,
    player_model_versions: str | None,
    team_model_version: str | None,
    train_and_register: bool,
    model_version: str | None,
    # A/B testing options
    ab_test: tuple | None,
    experiment_name: str | None,
    traffic_split: float,
    # Standard options
    wildcard_week: int,
    free_hit_week: int,
    triple_captain_week: int,
    bench_boost_week: int,
    n_previous: int,
    no_current_season: bool,
    team_model: str,
    player_model: str,
    season: str,
):
    """Enhanced AIrsenal pipeline with model versioning support"""

    click.echo("=" * 60)
    click.echo("AIrsenal Enhanced Pipeline with Model Versioning")
    click.echo("=" * 60)

    # Set up multiprocessing
    set_multiprocessing_start_method()

    # Determine number of threads
    if num_thread is None:
        num_thread = multiprocessing.cpu_count()

    # Gameweek range
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))

    with session_scope() as session:
        session.expire_on_commit = False

        try:
            # 1. Database initialization
            if clean:
                click.echo("Cleaning and reinitializing database...")
                check_clean_db(clean=clean, dbsession=session)

                # Determine seasons based on current season and n_previous
                if no_current_season:
                    seasons = get_past_seasons(n_previous)
                else:
                    seasons = [season, *get_past_seasons(n_previous)]

                make_init_db(fpl_team_id or fetcher.FPL_TEAM_ID, seasons, session)

            # 2. Update database with latest data
            click.echo("Updating database with latest FPL data...")
            # Set do_attributes to True by default for complete data updates
            do_attributes = True
            update_db(
                season, do_attributes, fpl_team_id or fetcher.FPL_TEAM_ID, session
            )

            # 3. Model training and registration (if requested)
            if train_and_register:
                click.echo("Training and registering new models...")
                _train_and_register_models(session, season, gw_range[0], model_version)

            # 4. Run predictions
            if ab_test:
                # A/B testing mode
                control_version, treatment_version = ab_test
                experiment_tag = _run_ab_test(
                    session,
                    gw_range,
                    season,
                    control_version,
                    treatment_version,
                    experiment_name,
                    traffic_split,
                )
                click.echo(f"A/B test completed with experiment tag: {experiment_tag}")
                tag = experiment_tag  # Use for optimization

            elif use_versioned_models:
                # Versioned models mode
                click.echo("Running predictions with versioned models...")
                tag = _run_versioned_predictions(
                    session,
                    gw_range,
                    season,
                    use_production_models,
                    player_model_versions,
                    team_model_version,
                )

            else:
                # Standard mode
                click.echo("Running standard predictions...")
                tag = make_predictedscore_table(
                    gw_range=gw_range,
                    season=season,
                    num_thread=num_thread,
                    dbsession=session,
                )

            # 5. Show top predicted points
            get_top_predicted_points(
                gameweek=gw_range,
                tag=tag,
                season=season,
                per_position=True,
                n_players=5,
                dbsession=session,
            )

            # 6. Run optimization
            click.echo("Running transfer optimization...")
            # Construct chip_gameweeks dict from individual chip parameters
            chip_gameweeks = {
                "wildcard": wildcard_week,
                "free_hit": free_hit_week,
                "triple_captain": triple_captain_week,
                "bench_boost": bench_boost_week,
            }
            run_optimization(
                gameweeks=gw_range,
                tag=tag,
                season=season,
                fpl_team_id=fpl_team_id,
                chip_gameweeks=chip_gameweeks,
            )

            # 7. Apply transfers if requested
            if apply_transfers:
                click.echo("Applying transfers via FPL API...")
                make_transfers(fpl_team_id=fpl_team_id)
                set_lineup(fpl_team_id=fpl_team_id)
                click.echo("Transfers applied successfully!")

            click.echo("\nPipeline completed successfully!")

        except Exception as e:
            click.echo(f"Pipeline failed with error: {e}", err=True)
            sys.exit(1)


def _train_and_register_models(
    session: Session,
    season: str,
    gameweek: int,
    version: str | None,
):
    """Train and register new models"""
    manager = VersionedPredictionManager(dbsession=session)

    # Train player models for each position
    for position in ["GK", "DEF", "MID", "FWD"]:
        click.echo(f"  Training {position} player model...")
        manager.fit_and_register_player_model(
            position=position,
            season=season,
            gameweek=gameweek,
            version=version,
            notes=f"Auto-trained {position} model from pipeline",
        )

    # Train team model
    click.echo("  Training team model...")
    manager.fit_and_register_team_model(
        season=season,
        gameweek=gameweek,
        version=version,
        notes="Auto-trained team model from pipeline",
    )

    click.echo("Model training and registration completed!")


def _run_versioned_predictions(
    session: Session,
    gw_range: list[int],
    season: str,
    use_production_models: bool,
    player_model_versions: str | None,
    team_model_version: str | None,
) -> str:
    """Run predictions using versioned models"""

    # Parse player model versions if provided
    player_versions_dict = None
    if player_model_versions:
        import json

        try:
            player_versions_dict = json.loads(player_model_versions)
        except json.JSONDecodeError:
            click.echo("Error: Invalid JSON for player model versions", err=True)
            sys.exit(1)

    manager = VersionedPredictionManager(dbsession=session)

    tag = manager.run_versioned_predictions(
        gw_range=gw_range,
        season=season,
        player_model_versions=player_versions_dict,
        team_model_version=team_model_version,
        use_production_models=use_production_models,
        tag_prefix="versioned_pipeline_",
        experiment_name="pipeline_run",
    )

    click.echo(f"Versioned predictions completed with tag: {tag}")
    return tag


def _run_ab_test(
    session: Session,
    gw_range: list[int],
    season: str,
    control_version: str,
    treatment_version: str,
    experiment_name: str | None,
    traffic_split: float,
) -> str:
    """Run A/B test between two model versions"""

    if not experiment_name:
        experiment_name = f"pipeline_test_{control_version}_vs_{treatment_version}"

    # Determine model type from version format
    # This is a simplified assumption - in practice you'd have a more robust way to determine this
    model_name = "player_model_fwd"  # Default assumption

    ab_manager = ABTestManager(dbsession=session)

    # Create experiment config
    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        control_version=control_version,
        treatment_version=treatment_version,
        traffic_split=traffic_split,
        max_duration_days=1,  # Short duration for pipeline runs
    )

    # Create and start experiment
    experiment = ab_manager.create_experiment(config, "Pipeline A/B test")
    ab_manager.start_experiment(experiment.id)

    # Run experiment iteration
    results = ab_manager.run_experiment_iteration(experiment.id, gw_range, season)

    # For simplicity, return the treatment tag as the main tag
    # In practice, you might want to analyze results first
    return results["treatment_tag"]


if __name__ == "__main__":
    main()
