"""
CLI commands for model versioning and management

This script provides command-line interfaces for:
- Registering models
- Listing model versions
- Loading and comparing models
- Managing A/B tests
- Model deployment and production settings
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.framework.ab_testing import ABTestManager, ExperimentConfig
from airsenal.framework.model_versioning import ModelVersionManager
from airsenal.framework.schema import session_scope
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK
from airsenal.framework.versioned_prediction_utils import VersionedPredictionManager


@click.group()
def cli():
    """AIrsenal Model Management CLI"""
    pass


@cli.group()
def model():
    """Model versioning commands"""
    pass


@cli.group()
def experiment():
    """A/B testing experiment commands"""
    pass


# Model management commands
@model.command("list")
@click.option("--model-name", help="Filter by model name")
@click.option("--limit", type=int, default=20, help="Limit number of results")
@click.option("--format", "output_format", default="table", 
              type=click.Choice(["table", "json", "csv"]), help="Output format")
def list_models(model_name: Optional[str], limit: int, output_format: str):
    """List available model versions"""
    
    with session_scope() as session:
        manager = ModelVersionManager(dbsession=session)
        df = manager.list_model_versions(model_name, limit)
        
        if df.empty:
            click.echo("No models found.")
            return
        
        if output_format == "table":
            click.echo(df.to_string(index=False))
        elif output_format == "json":
            click.echo(df.to_json(indent=2))
        elif output_format == "csv":
            click.echo(df.to_csv(index=False))


@model.command("info")
@click.argument("model_name")
@click.option("--version", help="Specific version to show info for")
def model_info(model_name: str, version: Optional[str]):
    """Show detailed information about a model"""
    
    with session_scope() as session:
        manager = ModelVersionManager(dbsession=session)
        
        try:
            # Get model versions
            df = manager.list_model_versions(model_name)
            
            if df.empty:
                click.echo(f"Model '{model_name}' not found.")
                return
            
            if version:
                df = df[df["version"] == version]
                if df.empty:
                    click.echo(f"Version '{version}' not found for model '{model_name}'.")
                    return
            
            click.echo(f"\n=== Model: {model_name} ===")
            click.echo(df.to_string(index=False))
            
            # Show performance history if available
            if version:
                click.echo(f"\n=== Performance History for v{version} ===")
                # TODO: Add performance history query
                click.echo("Performance history not yet implemented.")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@model.command("compare")
@click.argument("model_name")
@click.argument("versions", nargs=-1, required=True)
@click.option("--metrics", default="validation_mae,validation_rmse,validation_accuracy",
              help="Comma-separated list of metrics to compare")
def compare_models(model_name: str, versions: List[str], metrics: str):
    """Compare performance metrics across model versions"""
    
    metrics_list = [m.strip() for m in metrics.split(",")]
    
    with session_scope() as session:
        manager = ModelVersionManager(dbsession=session)
        
        try:
            df = manager.compare_models(model_name, list(versions), metrics_list)
            
            if df.empty:
                click.echo("No comparison data available.")
                return
            
            click.echo(f"\n=== Model Comparison: {model_name} ===")
            click.echo(df.to_string(index=False))
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@model.command("set-production")
@click.argument("model_name")
@click.argument("version")
@click.option("--confirm", is_flag=True, help="Confirm the action")
def set_production(model_name: str, version: str, confirm: bool):
    """Set a model version as the production model"""
    
    if not confirm:
        click.echo("This will change the production model. Use --confirm to proceed.")
        return
    
    with session_scope() as session:
        manager = ModelVersionManager(dbsession=session)
        
        try:
            model_version = manager.set_production_model(model_name, version)
            click.echo(f"Set {model_name} v{version} as production model.")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@model.command("cleanup")
@click.argument("model_name")
@click.option("--keep-latest", default=5, type=int, help="Number of latest versions to keep")
@click.option("--keep-production", is_flag=True, default=True, help="Keep production model")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
def cleanup_models(model_name: str, keep_latest: int, keep_production: bool, dry_run: bool):
    """Clean up old model versions"""
    
    with session_scope() as session:
        manager = ModelVersionManager(dbsession=session)
        
        try:
            if dry_run:
                click.echo(f"DRY RUN: Would clean up old versions of {model_name}")
                click.echo(f"  Keep latest: {keep_latest}")
                click.echo(f"  Keep production: {keep_production}")
                # TODO: Show what would be deleted
                return
            
            deleted_count = manager.cleanup_old_versions(
                model_name, keep_latest, keep_production
            )
            click.echo(f"Cleaned up {deleted_count} old versions of {model_name}.")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@model.command("train-and-register")
@click.argument("model_name")
@click.argument("model_type", type=click.Choice(["player", "team"]))
@click.option("--version", help="Version string (auto-generated if not provided)")
@click.option("--position", help="Position for player models", type=click.Choice(["GK", "DEF", "MID", "FWD"]))
@click.option("--season", default=CURRENT_SEASON, help="Season to train on")
@click.option("--gameweek", default=NEXT_GAMEWEEK, type=int, help="Current gameweek")
@click.option("--notes", help="Notes about this model version")
def train_and_register(
    model_name: str,
    model_type: str,
    version: Optional[str],
    position: Optional[str],
    season: str,
    gameweek: int,
    notes: Optional[str],
):
    """Train a new model and register it"""
    
    if model_type == "player" and not position:
        click.echo("Position is required for player models.")
        sys.exit(1)
    
    with session_scope() as session:
        manager = VersionedPredictionManager(dbsession=session)
        
        try:
            if model_type == "player":
                # Type assertion: position is guaranteed to be str due to validation above
                assert position is not None, "Position should not be None for player models"
                click.echo(f"Training {position} player model...")
                model = manager.fit_and_register_player_model(
                    position=position,
                    season=season,
                    gameweek=gameweek,
                    version=version,
                    notes=notes,
                )
                click.echo(f"Successfully trained and registered {position} player model.")
                
            elif model_type == "team":
                click.echo("Training team model...")
                model = manager.fit_and_register_team_model(
                    season=season,
                    gameweek=gameweek,
                    version=version,
                    notes=notes,
                )
                click.echo("Successfully trained and registered team model.")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


# Experiment management commands
@experiment.command("create")
@click.argument("experiment_name")
@click.argument("model_name")
@click.argument("control_version")
@click.argument("treatment_version")
@click.option("--traffic-split", default=0.5, type=float, help="Traffic fraction for treatment")
@click.option("--min-sample-size", default=100, type=int, help="Minimum sample size")
@click.option("--max-duration-days", default=14, type=int, help="Maximum experiment duration")
@click.option("--description", help="Experiment description")
def create_experiment(
    experiment_name: str,
    model_name: str,
    control_version: str,
    treatment_version: str,
    traffic_split: float,
    min_sample_size: int,
    max_duration_days: int,
    description: Optional[str],
):
    """Create a new A/B testing experiment"""
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        control_version=control_version,
        treatment_version=treatment_version,
        traffic_split=traffic_split,
        min_sample_size=min_sample_size,
        max_duration_days=max_duration_days,
    )
    
    with session_scope() as session:
        manager = ABTestManager(dbsession=session)
        
        try:
            experiment = manager.create_experiment(config, description)
            click.echo(f"Created experiment '{experiment_name}' (ID: {experiment.id})")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@experiment.command("list")
@click.option("--model-name", help="Filter by model name")
@click.option("--status", help="Filter by status")
@click.option("--limit", default=20, type=int, help="Limit number of results")
def list_experiments(model_name: Optional[str], status: Optional[str], limit: int):
    """List A/B testing experiments"""
    
    with session_scope() as session:
        manager = ABTestManager(dbsession=session)
        
        try:
            df = manager.get_experiment_history(model_name, limit)
            
            if status:
                df = df[df["status"] == status]
            
            if df.empty:
                click.echo("No experiments found.")
                return
            
            click.echo(df.to_string(index=False))
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@experiment.command("start")
@click.argument("experiment_id", type=int)
def start_experiment(experiment_id: int):
    """Start a planned experiment"""
    
    with session_scope() as session:
        manager = ABTestManager(dbsession=session)
        
        try:
            experiment = manager.start_experiment(experiment_id)
            click.echo(f"Started experiment '{experiment.experiment_name}' (ID: {experiment_id})")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@experiment.command("stop")
@click.argument("experiment_id", type=int)
@click.option("--reason", default="Manual stop", help="Reason for stopping")
def stop_experiment(experiment_id: int, reason: str):
    """Stop a running experiment"""
    
    with session_scope() as session:
        manager = ABTestManager(dbsession=session)
        
        try:
            experiment = manager.stop_experiment(experiment_id, reason)
            click.echo(f"Stopped experiment '{experiment.experiment_name}': {reason}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@experiment.command("run")
@click.argument("experiment_id", type=int)
@click.option("--weeks-ahead", default=3, type=int, help="Number of weeks to predict")
@click.option("--season", default=CURRENT_SEASON, help="Season to predict for")
def run_experiment(experiment_id: int, weeks_ahead: int, season: str):
    """Run one iteration of an A/B experiment"""
    
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))
    
    with session_scope() as session:
        manager = ABTestManager(dbsession=session)
        
        try:
            results = manager.run_experiment_iteration(experiment_id, gw_range, season)
            click.echo(f"Ran experiment iteration for experiment {experiment_id}")
            click.echo(f"  Control tag: {results['control_tag']}")
            click.echo(f"  Treatment tag: {results['treatment_tag']}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@experiment.command("analyze")
@click.argument("experiment_id", type=int)
@click.argument("control_tags", nargs=-1, required=True)
@click.option("--treatment-tags", help="Comma-separated treatment tags")
@click.option("--output-file", help="Save detailed results to JSON file")
def analyze_experiment(
    experiment_id: int,
    control_tags: List[str],
    treatment_tags: Optional[str],
    output_file: Optional[str],
):
    """Analyze results of an A/B experiment"""
    
    if not treatment_tags:
        click.echo("Treatment tags are required for analysis.")
        sys.exit(1)
    
    treatment_tag_list = [t.strip() for t in treatment_tags.split(",")]
    
    with session_scope() as session:
        manager = ABTestManager(dbsession=session)
        
        try:
            analysis = manager.analyze_experiment_results(
                experiment_id, list(control_tags), treatment_tag_list
            )
            
            click.echo(f"\n=== Experiment Analysis: {analysis['experiment_name']} ===")
            click.echo(f"Control sample size: {analysis['control_sample_size']}")
            click.echo(f"Treatment sample size: {analysis['treatment_sample_size']}")
            click.echo(f"Control mean prediction: {analysis['control_mean_prediction']:.4f}")
            click.echo(f"Treatment mean prediction: {analysis['treatment_mean_prediction']:.4f}")
            click.echo(f"Effect size (Cohen's d): {analysis['cohens_d']:.4f} ({analysis['effect_size_interpretation']})")
            click.echo(f"T-test p-value: {analysis['t_test_pvalue']:.6f}")
            click.echo(f"Statistically significant: {analysis['t_test_significant']}")
            click.echo(f"Recommendation: {analysis['recommendation']}")
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                click.echo(f"\nDetailed results saved to: {output_file}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


# Prediction commands with versioning
@cli.group()
def predict():
    """Prediction commands with model versioning"""
    pass


@predict.command("run")
@click.option("--weeks-ahead", default=3, type=int, help="Number of weeks to predict")
@click.option("--season", default=CURRENT_SEASON, help="Season to predict for")
@click.option("--use-production", is_flag=True, help="Use production models")
@click.option("--player-model-versions", help="JSON string of player model versions")
@click.option("--team-model-version", help="Specific team model version")
@click.option("--tag-prefix", help="Prefix for prediction tag")
@click.option("--experiment-name", help="Experiment name for grouping")
def run_prediction(
    weeks_ahead: int,
    season: str,
    use_production: bool,
    player_model_versions: Optional[str],
    team_model_version: Optional[str],
    tag_prefix: Optional[str],
    experiment_name: Optional[str],
):
    """Run predictions using versioned models"""
    
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))
    
    # Parse player model versions if provided
    player_versions_dict = None
    if player_model_versions:
        try:
            player_versions_dict = json.loads(player_model_versions)
        except json.JSONDecodeError:
            click.echo("Invalid JSON for player model versions.")
            sys.exit(1)
    
    with session_scope() as session:
        manager = VersionedPredictionManager(dbsession=session)
        
        try:
            tag = manager.run_versioned_predictions(
                gw_range=gw_range,
                season=season,
                player_model_versions=player_versions_dict,
                team_model_version=team_model_version,
                use_production_models=use_production,
                tag_prefix=tag_prefix,
                experiment_name=experiment_name,
            )
            
            click.echo(f"Predictions completed with tag: {tag}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@predict.command("compare-models")
@click.argument("model_name")
@click.argument("versions", nargs=-1, required=True)
@click.option("--weeks-ahead", default=3, type=int, help="Number of weeks to predict")
@click.option("--season", default=CURRENT_SEASON, help="Season to predict for")
def compare_model_predictions(model_name: str, versions: List[str], weeks_ahead: int, season: str):
    """Compare predictions from different model versions"""
    
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))
    
    with session_scope() as session:
        manager = VersionedPredictionManager(dbsession=session)
        
        try:
            results = manager.run_model_comparison(
                model_name, list(versions), gw_range, season
            )
            
            click.echo(f"Model comparison completed for {model_name}")
            click.echo(f"Versions compared: {', '.join(versions)}")
            click.echo(f"Prediction tags:")
            
            for version, tag in results["tags"].items():
                click.echo(f"  {version}: {tag}")
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    cli()