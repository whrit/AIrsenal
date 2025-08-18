"""
A/B Testing Framework for Model Comparison

This module provides A/B testing capabilities for comparing different model versions
in production-like environments with statistical significance testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from airsenal.framework.model_versioning import ModelVersionManager
from airsenal.framework.schema import (
    ModelExperiment,
    ModelRegistry,
    ModelVersion,
    PlayerPrediction,
    session,
)
from airsenal.framework.utils import CURRENT_SEASON
from airsenal.framework.versioned_prediction_utils import VersionedPredictionManager

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration"""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiment"""

    experiment_name: str
    model_name: str
    control_version: str
    treatment_version: str
    traffic_split: float = 0.5  # Fraction of traffic for treatment
    min_sample_size: int = 100
    max_duration_days: int = 14
    significance_level: float = 0.05
    minimum_effect_size: float = 0.02  # Minimum meaningful difference
    early_stopping: bool = True
    success_metrics: list[str] | None = None

    def __post_init__(self):
        if self.success_metrics is None:
            self.success_metrics = ["mae", "prediction_correlation"]


class ABTestManager:
    """Manager for A/B testing experiments"""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession
        self.model_manager = ModelVersionManager(dbsession=dbsession)
        self.prediction_manager = VersionedPredictionManager(dbsession=dbsession)

    def create_experiment(
        self,
        config: ExperimentConfig,
        description: str | None = None,
        created_by: str = "airsenal",
    ) -> ModelExperiment:
        """Create a new A/B testing experiment"""

        # Validate that model versions exist
        try:
            self.model_manager.load_model_version(
                config.model_name, config.control_version
            )
            self.model_manager.load_model_version(
                config.model_name, config.treatment_version
            )
        except Exception as e:
            msg = f"Cannot load model versions: {e}"
            raise ValueError(msg)

        # Get model version IDs
        registry = (
            self.dbsession.query(ModelRegistry)
            .filter_by(model_name=config.model_name, is_active=True)
            .first()
        )

        if not registry:
            msg = f"Model '{config.model_name}' not found in registry"
            raise ValueError(msg)

        control_version_record = (
            self.dbsession.query(ModelVersion)
            .filter_by(registry_id=registry.id, version=config.control_version)
            .first()
        )

        treatment_version_record = (
            self.dbsession.query(ModelVersion)
            .filter_by(registry_id=registry.id, version=config.treatment_version)
            .first()
        )

        if not control_version_record or not treatment_version_record:
            msg = "Model version records not found"
            raise ValueError(msg)

        # Create experiment record
        experiment = ModelExperiment(
            experiment_name=config.experiment_name,
            description=description
            or f"A/B test: {config.control_version} vs {config.treatment_version}",
            model_version_id=treatment_version_record.id,
            control_version_id=control_version_record.id,
            traffic_split=config.traffic_split,
            start_date=datetime.now().isoformat(),
            status=ExperimentStatus.PLANNED.value,
            created_by=created_by,
            notes=f"Config: {config}",
        )

        self.dbsession.add(experiment)
        self.dbsession.commit()

        logger.info(
            f"Created experiment '{config.experiment_name}' (ID: {experiment.id})"
        )
        return experiment

    def start_experiment(self, experiment_id: int) -> ModelExperiment:
        """Start a planned experiment"""

        experiment = (
            self.dbsession.query(ModelExperiment).filter_by(id=experiment_id).first()
        )

        if not experiment:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        if experiment.status != ExperimentStatus.PLANNED.value:
            msg = f"Cannot start experiment in status '{experiment.status}'"
            raise ValueError(msg)

        experiment.status = ExperimentStatus.RUNNING.value
        experiment.start_date = datetime.now().isoformat()

        self.dbsession.commit()

        logger.info(
            f"Started experiment '{experiment.experiment_name}' (ID: {experiment.id})"
        )
        return experiment

    def stop_experiment(
        self,
        experiment_id: int,
        reason: str = "Manual stop",
    ) -> ModelExperiment:
        """Stop a running experiment"""

        experiment = (
            self.dbsession.query(ModelExperiment).filter_by(id=experiment_id).first()
        )

        if not experiment:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        if experiment.status != ExperimentStatus.RUNNING.value:
            msg = f"Cannot stop experiment in status '{experiment.status}'"
            raise ValueError(msg)

        experiment.status = ExperimentStatus.STOPPED.value
        experiment.end_date = datetime.now().isoformat()
        experiment.notes = f"{experiment.notes}\nStopped: {reason}"

        self.dbsession.commit()

        logger.info(f"Stopped experiment '{experiment.experiment_name}': {reason}")
        return experiment

    def run_experiment_iteration(
        self,
        experiment_id: int,
        gw_range: list[int],
        season: str = CURRENT_SEASON,
        sample_players: list[int] | None = None,
    ) -> dict[str, str]:
        """
        Run one iteration of an A/B experiment

        Args:
            experiment_id: ID of the experiment to run
            gw_range: Gameweeks to predict
            season: Season to predict for
            sample_players: Optional list of player IDs to restrict predictions to

        Returns:
            Dictionary with prediction tags for control and treatment groups
        """

        experiment = (
            self.dbsession.query(ModelExperiment).filter_by(id=experiment_id).first()
        )

        if not experiment:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        if experiment.status != ExperimentStatus.RUNNING.value:
            msg = (
                f"Experiment must be running to iterate (current: {experiment.status})"
            )
            raise ValueError(msg)

        # Get model versions
        control_version = (
            self.dbsession.query(ModelVersion)
            .filter_by(id=experiment.control_version_id)
            .first()
        )
        treatment_version = (
            self.dbsession.query(ModelVersion)
            .filter_by(id=experiment.model_version_id)
            .first()
        )

        if not control_version or not treatment_version:
            msg = "Model versions not found"
            raise ValueError(msg)

        # Determine model type and prepare configurations
        model_name = control_version.model_registry.model_name

        if "player_model" in model_name:
            position = model_name.split("_")[-1]
            control_config = {position: control_version.version}
            treatment_config = {position: treatment_version.version}
            control_tag = self.prediction_manager.run_versioned_predictions(
                gw_range=gw_range,
                season=season,
                player_model_versions=control_config,
                tag_prefix=f"ab_control_{experiment.experiment_name}_",
                experiment_name=f"{experiment.experiment_name}_control",
            )
            treatment_tag = self.prediction_manager.run_versioned_predictions(
                gw_range=gw_range,
                season=season,
                player_model_versions=treatment_config,
                tag_prefix=f"ab_treatment_{experiment.experiment_name}_",
                experiment_name=f"{experiment.experiment_name}_treatment",
            )
        elif "team_model" in model_name:
            control_tag = self.prediction_manager.run_versioned_predictions(
                gw_range=gw_range,
                season=season,
                team_model_version=control_version.version,
                tag_prefix=f"ab_control_{experiment.experiment_name}_",
                experiment_name=f"{experiment.experiment_name}_control",
            )
            treatment_tag = self.prediction_manager.run_versioned_predictions(
                gw_range=gw_range,
                season=season,
                team_model_version=treatment_version.version,
                tag_prefix=f"ab_treatment_{experiment.experiment_name}_",
                experiment_name=f"{experiment.experiment_name}_treatment",
            )
        else:
            msg = f"Unsupported model type: {model_name}"
            raise ValueError(msg)

        logger.info(f"Ran experiment iteration for '{experiment.experiment_name}'")
        logger.info(f"  Control tag: {control_tag}")
        logger.info(f"  Treatment tag: {treatment_tag}")

        return {
            "control_tag": control_tag,
            "treatment_tag": treatment_tag,
            "experiment_id": str(experiment_id),
        }

    def analyze_experiment_results(
        self,
        experiment_id: int,
        control_tags: list[str],
        treatment_tags: list[str],
        actual_results: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Analyze results of an A/B experiment

        Args:
            experiment_id: ID of the experiment
            control_tags: List of prediction tags for control group
            treatment_tags: List of prediction tags for treatment group
            actual_results: DataFrame with actual player performance (if available)

        Returns:
            Dictionary with analysis results
        """

        experiment = (
            self.dbsession.query(ModelExperiment).filter_by(id=experiment_id).first()
        )

        if not experiment:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        # Get predictions for both groups
        control_predictions = self._get_predictions_by_tags(control_tags)
        treatment_predictions = self._get_predictions_by_tags(treatment_tags)

        if control_predictions.empty or treatment_predictions.empty:
            msg = "No predictions found for experiment groups"
            raise ValueError(msg)

        # Basic comparison
        analysis = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.experiment_name,
            "control_sample_size": len(control_predictions),
            "treatment_sample_size": len(treatment_predictions),
            "control_mean_prediction": control_predictions["predicted_points"].mean(),
            "treatment_mean_prediction": treatment_predictions[
                "predicted_points"
            ].mean(),
            "control_std_prediction": control_predictions["predicted_points"].std(),
            "treatment_std_prediction": treatment_predictions["predicted_points"].std(),
        }

        # Statistical tests
        control_values = control_predictions["predicted_points"].values
        treatment_values = treatment_predictions["predicted_points"].values

        # Two-sample t-test
        t_stat, t_pvalue = stats.ttest_ind(treatment_values, control_values)
        analysis["t_test_statistic"] = t_stat
        analysis["t_test_pvalue"] = t_pvalue
        analysis["t_test_significant"] = t_pvalue < 0.05

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(control_values) - 1) * np.var(control_values, ddof=1)
                + (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)
            )
            / (len(control_values) + len(treatment_values) - 2)
        )
        cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
        analysis["cohens_d"] = cohens_d
        analysis["effect_size_interpretation"] = self._interpret_effect_size(cohens_d)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(
            treatment_values, control_values, alternative="two-sided"
        )
        analysis["mannwhitney_u_statistic"] = u_stat
        analysis["mannwhitney_u_pvalue"] = u_pvalue
        analysis["mannwhitney_u_significant"] = u_pvalue < 0.05

        # If actual results are provided, calculate accuracy metrics
        if actual_results is not None:
            control_accuracy = self._calculate_prediction_accuracy(
                control_predictions, actual_results
            )
            treatment_accuracy = self._calculate_prediction_accuracy(
                treatment_predictions, actual_results
            )

            analysis["control_accuracy"] = control_accuracy
            analysis["treatment_accuracy"] = treatment_accuracy
            analysis["accuracy_improvement"] = treatment_accuracy.get(
                "mae", 0
            ) - control_accuracy.get("mae", 0)

        # Confidence intervals
        control_ci = stats.t.interval(
            0.95,
            len(control_values) - 1,
            loc=np.mean(control_values),
            scale=stats.sem(control_values),
        )
        treatment_ci = stats.t.interval(
            0.95,
            len(treatment_values) - 1,
            loc=np.mean(treatment_values),
            scale=stats.sem(treatment_values),
        )

        analysis["control_95_ci"] = control_ci
        analysis["treatment_95_ci"] = treatment_ci

        # Recommendation
        analysis["recommendation"] = self._generate_recommendation(analysis)

        logger.info(f"Analyzed experiment '{experiment.experiment_name}':")
        logger.info(f"  Effect size (Cohen's d): {cohens_d:.4f}")
        logger.info(f"  T-test p-value: {t_pvalue:.4f}")
        logger.info(f"  Recommendation: {analysis['recommendation']}")

        return analysis

    def check_early_stopping(
        self,
        experiment_id: int,
        analysis_results: dict[str, Any],
        config: ExperimentConfig,
    ) -> tuple[bool, str]:
        """
        Check if experiment should be stopped early

        Args:
            experiment_id: ID of the experiment
            analysis_results: Results from analyze_experiment_results
            config: Experiment configuration

        Returns:
            Tuple of (should_stop, reason)
        """

        if not config.early_stopping:
            return False, "Early stopping disabled"

        # Check minimum sample size
        min_sample = min(
            analysis_results["control_sample_size"],
            analysis_results["treatment_sample_size"],
        )

        if min_sample < config.min_sample_size:
            return (
                False,
                f"Sample size too small: {min_sample} < {config.min_sample_size}",
            )

        # Check for significant result with sufficient effect size
        is_significant = analysis_results.get("t_test_significant", False)
        effect_size = abs(analysis_results.get("cohens_d", 0))

        if is_significant and effect_size >= config.minimum_effect_size:
            return True, f"Significant result with effect size {effect_size:.4f}"

        # Check maximum duration
        experiment = (
            self.dbsession.query(ModelExperiment).filter_by(id=experiment_id).first()
        )
        if experiment and experiment.start_date:
            start_date = datetime.fromisoformat(experiment.start_date)
            days_running = (datetime.now() - start_date).days

            if days_running >= config.max_duration_days:
                return True, f"Maximum duration reached: {days_running} days"

        return False, "Continue experiment"

    def complete_experiment(
        self,
        experiment_id: int,
        winner_version_id: int | None = None,
        confidence_level: float | None = None,
        notes: str | None = None,
    ) -> ModelExperiment:
        """Complete an experiment and record the winner"""

        experiment = (
            self.dbsession.query(ModelExperiment).filter_by(id=experiment_id).first()
        )

        if not experiment:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        experiment.status = ExperimentStatus.COMPLETED.value
        experiment.end_date = datetime.now().isoformat()
        experiment.winner_version_id = winner_version_id
        experiment.confidence_level = confidence_level

        if notes:
            experiment.notes = f"{experiment.notes}\nCompletion notes: {notes}"

        self.dbsession.commit()

        logger.info(f"Completed experiment '{experiment.experiment_name}'")
        if winner_version_id:
            logger.info(f"  Winner: Model version {winner_version_id}")

        return experiment

    def get_experiment_history(
        self,
        model_name: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get history of experiments"""

        query = self.dbsession.query(
            ModelExperiment.id,
            ModelExperiment.experiment_name,
            ModelExperiment.status,
            ModelExperiment.start_date,
            ModelExperiment.end_date,
            ModelExperiment.confidence_level,
            ModelExperiment.created_by,
        )

        if model_name:
            # Join with ModelVersion to filter by model name
            query = (
                query.join(
                    ModelVersion,
                    ModelExperiment.model_version_id == ModelVersion.id,
                )
                .join(
                    ModelRegistry,
                    ModelVersion.registry_id == ModelRegistry.id,
                )
                .filter(ModelRegistry.model_name == model_name)
            )

        if limit:
            query = query.limit(limit)

        query = query.order_by(ModelExperiment.start_date.desc())

        return pd.read_sql(query.statement, self.dbsession.bind)

    def _get_predictions_by_tags(self, tags: list[str]) -> pd.DataFrame:
        """Get predictions DataFrame for given tags"""

        if not tags:
            return pd.DataFrame()

        query = self.dbsession.query(PlayerPrediction).filter(
            PlayerPrediction.tag.in_(tags)
        )

        return pd.read_sql(query.statement, self.dbsession.bind)

    def _calculate_prediction_accuracy(
        self,
        predictions: pd.DataFrame,
        actual_results: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate prediction accuracy metrics"""

        # This is a simplified implementation
        # In practice, you'd need to match predictions with actual results
        # by player_id and gameweek

        accuracy_metrics = {}

        if (
            "predicted_points" in predictions.columns
            and "actual_points" in actual_results.columns
        ):
            # Merge predictions with actual results
            merged = predictions.merge(
                actual_results,
                on=["player_id", "fixture_id"],
                how="inner",
                suffixes=("_pred", "_actual"),
            )

            if not merged.empty:
                pred_values = merged["predicted_points"].values
                actual_values = merged["actual_points"].values

                # Mean Absolute Error
                mae = np.mean(np.abs(pred_values - actual_values))
                accuracy_metrics["mae"] = mae

                # Root Mean Square Error
                rmse = np.sqrt(np.mean((pred_values - actual_values) ** 2))
                accuracy_metrics["rmse"] = rmse

                # Correlation
                correlation = np.corrcoef(pred_values, actual_values)[0, 1]
                accuracy_metrics["correlation"] = correlation

        return accuracy_metrics

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""

        abs_d = abs(cohens_d)

        if abs_d < 0.2:
            return "negligible"
        if abs_d < 0.5:
            return "small"
        if abs_d < 0.8:
            return "medium"
        return "large"

    def _generate_recommendation(self, analysis: dict[str, Any]) -> str:
        """Generate recommendation based on analysis results"""

        is_significant = analysis.get("t_test_significant", False)
        effect_size = abs(analysis.get("cohens_d", 0))

        treatment_better = analysis.get("treatment_mean_prediction", 0) > analysis.get(
            "control_mean_prediction", 0
        )

        if is_significant and effect_size >= 0.2:
            if treatment_better:
                return (
                    "Deploy treatment version - statistically significant improvement"
                )
            return "Keep control version - treatment performs significantly worse"
        if is_significant and effect_size < 0.2:
            return "Inconclusive - significant but small effect size"
        return "No significant difference - continue with control version"


# Convenience functions
def create_ab_test(
    experiment_name: str,
    model_name: str,
    control_version: str,
    treatment_version: str,
    **kwargs,
) -> ModelExperiment:
    """Convenience function to create an A/B test"""

    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        control_version=control_version,
        treatment_version=treatment_version,
        **kwargs,
    )

    manager = ABTestManager()
    return manager.create_experiment(config)


def run_ab_test_iteration(
    experiment_id: int,
    gw_range: list[int],
    season: str = CURRENT_SEASON,
) -> dict[str, str]:
    """Convenience function to run an A/B test iteration"""

    manager = ABTestManager()
    return manager.run_experiment_iteration(experiment_id, gw_range, season)


def analyze_ab_test(
    experiment_id: int,
    control_tags: list[str],
    treatment_tags: list[str],
    actual_results: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Convenience function to analyze A/B test results"""

    manager = ABTestManager()
    return manager.analyze_experiment_results(
        experiment_id, control_tags, treatment_tags, actual_results
    )
