"""
AIrsenal Feature Validation and Testing Framework

A comprehensive validation system for all engineered features from Sprint 01, providing:
- Statistical validation using Great Expectations
- Data quality testing and outlier detection
- Feature importance analysis and correlation matrix
- Performance benchmarking for all components
- Data drift detection between seasons
- Comprehensive reporting and dashboard functionality

This framework validates the following Sprint 01 features:
- Form metrics (TASK-101): Rolling averages and momentum
- Weighted performance (TASK-102): Position-specific weighted scoring
- Trend detection (TASK-103): Statistical trend analysis
- xG/xA data (TASK-104): Expected goals and assists integration
- Fixture difficulty (TASK-107): Opponent strength ratings
- Home/away adjustments (TASK-108): Venue-specific adjustments
- Team strength (TASK-109): Bayesian strength modeling
- Penalty takers (TASK-110): Set piece specialist detection
- Rotation risk (TASK-111): Player rotation prediction
- Set piece specialists (TASK-112): Comprehensive set piece analysis

Performance Benchmarks:
- Form calculation: <50ms per player
- Weighted metrics: <50ms per player
- xG integration: <500ms API response
- All batch operations: <2s for 650 players

Author: AIrsenal Team
Version: 1.0.0
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

# Statistical and data quality testing
try:
    import scipy.stats as stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not available. Some statistical tests will be disabled.", stacklevel=2
    )

try:
    from great_expectations.dataset import PandasDataset

    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    warnings.warn(
        "Great Expectations not available. Validation rules will use simplified implementations.",
        stacklevel=2,
    )

# Plotting for dashboards
try:
    import importlib.util

    PLOTTING_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "matplotlib/seaborn not available. Dashboard features will be disabled.",
        stacklevel=2,
    )

from airsenal.framework.form_calculator import FormCalculator
from airsenal.framework.schema import (
    PlayerAttributes,
    PlayerScore,
    session,
)
from airsenal.framework.trend_detection import TrendDetector
from airsenal.framework.utils import CURRENT_SEASON
from airsenal.framework.weighted_performance import WeightedPerformanceCalculator

logger = logging.getLogger(__name__)


class FeatureValidationError(Exception):
    """Raised when feature validation encounters an error."""


class InsufficientDataError(Exception):
    """Raised when insufficient data is available for validation."""


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    feature_name: str
    validation_type: str
    passed: bool
    value: float | None
    expected_min: float | None
    expected_max: float | None
    message: str
    timestamp: datetime
    severity: str = "info"  # "critical", "high", "medium", "low", "info"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_name": self.feature_name,
            "validation_type": self.validation_type,
            "passed": self.passed,
            "value": self.value,
            "expected_min": self.expected_min,
            "expected_max": self.expected_max,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
        }


@dataclass
class FeatureExpectations:
    """Expected ranges and properties for features."""

    name: str
    data_type: str
    min_value: float | None = None
    max_value: float | None = None
    mean_range: tuple[float, float] | None = None
    std_range: tuple[float, float] | None = None
    null_percentage_max: float = 0.1
    outlier_percentage_max: float = 0.05
    correlation_with: list[str] | None = None
    expected_distribution: str = "normal"  # "normal", "uniform", "beta", "exponential"

    def __post_init__(self):
        if self.correlation_with is None:
            self.correlation_with = []


class FeatureValidator:
    """
    Main feature validation class using Great Expectations and custom validation rules.

    Provides comprehensive validation of all Sprint 01 features including:
    - Data type validation
    - Range and distribution checks
    - Null value detection
    - Outlier identification
    - Correlation analysis
    - Performance validation
    """

    def __init__(
        self,
        dbsession: Session = session,
        enable_ge: bool = True,
        validation_threshold: float = 0.95,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize the FeatureValidator.

        Args:
            dbsession: SQLAlchemy session for database operations
            enable_ge: Whether to use Great Expectations (if available)
            validation_threshold: Percentage of validations that must pass
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.dbsession = dbsession
        self.enable_ge = enable_ge and GE_AVAILABLE
        self.validation_threshold = validation_threshold
        self.outlier_std_threshold = outlier_std_threshold

        # Initialize supporting components
        self.form_calculator = FormCalculator(dbsession)
        self.weighted_calculator = WeightedPerformanceCalculator(dbsession=dbsession)
        self.trend_detector = TrendDetector(dbsession)

        # Feature expectations (defined based on Sprint 01 analysis)
        self._feature_expectations = self._define_feature_expectations()

        # Validation results storage
        self.validation_results: list[ValidationResult] = []

        if self.enable_ge:
            logger.info("Great Expectations validation enabled")
        else:
            logger.warning(
                "Great Expectations not available, using simplified validation"
            )

    def _define_feature_expectations(self) -> dict[str, FeatureExpectations]:
        """Define expected ranges and properties for all Sprint 01 features."""
        return {
            # Form metrics (TASK-101)
            "form_3_games": FeatureExpectations(
                name="form_3_games",
                data_type="float",
                min_value=0.0,
                max_value=25.0,
                mean_range=(2.0, 8.0),
                std_range=(1.0, 5.0),
                null_percentage_max=0.15,
                expected_distribution="normal",
                correlation_with=["form_5_games", "form_10_games"],
            ),
            "form_5_games": FeatureExpectations(
                name="form_5_games",
                data_type="float",
                min_value=0.0,
                max_value=25.0,
                mean_range=(2.0, 8.0),
                std_range=(1.0, 4.5),
                null_percentage_max=0.15,
                expected_distribution="normal",
                correlation_with=["form_3_games", "form_10_games"],
            ),
            "form_10_games": FeatureExpectations(
                name="form_10_games",
                data_type="float",
                min_value=0.0,
                max_value=25.0,
                mean_range=(2.0, 8.0),
                std_range=(0.8, 4.0),
                null_percentage_max=0.20,
                expected_distribution="normal",
                correlation_with=["form_3_games", "form_5_games"],
            ),
            "momentum": FeatureExpectations(
                name="momentum",
                data_type="float",
                min_value=-1.0,
                max_value=1.0,
                mean_range=(-0.1, 0.1),
                std_range=(0.2, 0.6),
                null_percentage_max=0.20,
                expected_distribution="normal",
            ),
            # xG/xA metrics (TASK-104)
            "xg_per_90": FeatureExpectations(
                name="xg_per_90",
                data_type="float",
                min_value=0.0,
                max_value=1.5,
                mean_range=(0.1, 0.6),
                std_range=(0.1, 0.4),
                null_percentage_max=0.10,
                expected_distribution="exponential",
                correlation_with=["xa_per_90", "xgi_per_90"],
            ),
            "xa_per_90": FeatureExpectations(
                name="xa_per_90",
                data_type="float",
                min_value=0.0,
                max_value=1.0,
                mean_range=(0.05, 0.4),
                std_range=(0.05, 0.3),
                null_percentage_max=0.10,
                expected_distribution="exponential",
                correlation_with=["xg_per_90", "xgi_per_90"],
            ),
            "xgi_per_90": FeatureExpectations(
                name="xgi_per_90",
                data_type="float",
                min_value=0.0,
                max_value=2.0,
                mean_range=(0.15, 0.8),
                std_range=(0.1, 0.5),
                null_percentage_max=0.10,
                expected_distribution="exponential",
                correlation_with=["xg_per_90", "xa_per_90"],
            ),
            # Fixture difficulty (TASK-107)
            "next_3_fixture_difficulty": FeatureExpectations(
                name="next_3_fixture_difficulty",
                data_type="float",
                min_value=1.0,
                max_value=5.0,
                mean_range=(2.5, 3.5),
                std_range=(0.5, 1.2),
                null_percentage_max=0.05,
                expected_distribution="normal",
                correlation_with=["next_5_fixture_difficulty"],
            ),
            "next_5_fixture_difficulty": FeatureExpectations(
                name="next_5_fixture_difficulty",
                data_type="float",
                min_value=1.0,
                max_value=5.0,
                mean_range=(2.5, 3.5),
                std_range=(0.4, 1.0),
                null_percentage_max=0.05,
                expected_distribution="normal",
                correlation_with=["next_3_fixture_difficulty"],
            ),
            # Player roles (TASK-110)
            "is_penalty_taker": FeatureExpectations(
                name="is_penalty_taker",
                data_type="boolean",
                min_value=0.0,
                max_value=1.0,
                mean_range=(0.02, 0.08),  # ~3-5% of players are penalty takers
                null_percentage_max=0.0,
                expected_distribution="beta",
            ),
            "is_free_kick_taker": FeatureExpectations(
                name="is_free_kick_taker",
                data_type="boolean",
                min_value=0.0,
                max_value=1.0,
                mean_range=(0.05, 0.15),  # ~8-12% of players take free kicks
                null_percentage_max=0.0,
                expected_distribution="beta",
            ),
            "is_corner_taker": FeatureExpectations(
                name="is_corner_taker",
                data_type="boolean",
                min_value=0.0,
                max_value=1.0,
                mean_range=(0.08, 0.20),  # ~12-18% of players take corners
                null_percentage_max=0.0,
                expected_distribution="beta",
            ),
            "role_confidence": FeatureExpectations(
                name="role_confidence",
                data_type="float",
                min_value=0.0,
                max_value=1.0,
                mean_range=(0.3, 0.8),
                std_range=(0.2, 0.4),
                null_percentage_max=0.30,
                expected_distribution="beta",
            ),
            # Advanced performance statistics
            "shots_per_90": FeatureExpectations(
                name="shots_per_90",
                data_type="float",
                min_value=0.0,
                max_value=8.0,
                mean_range=(0.5, 3.0),
                std_range=(0.5, 2.0),
                null_percentage_max=0.20,
                expected_distribution="exponential",
            ),
            "key_passes_per_90": FeatureExpectations(
                name="key_passes_per_90",
                data_type="float",
                min_value=0.0,
                max_value=8.0,
                mean_range=(0.3, 2.5),
                std_range=(0.3, 1.5),
                null_percentage_max=0.20,
                expected_distribution="exponential",
            ),
        }

    def validate_feature(
        self,
        feature_name: str,
        data: pd.Series | np.ndarray | list,
        season: str = CURRENT_SEASON,
    ) -> list[ValidationResult]:
        """
        Validate a single feature against its expectations.

        Args:
            feature_name: Name of the feature to validate
            data: Feature data to validate
            season: Season context for validation

        Returns:
            List of ValidationResult objects
        """
        results = []

        if feature_name not in self._feature_expectations:
            logger.warning("No expectations defined for feature: %s", feature_name)
            return results

        expectations = self._feature_expectations[feature_name]

        # Convert to pandas Series for consistent handling
        if isinstance(data, list | np.ndarray):
            data = pd.Series(data)

        try:
            # Basic data validation
            results.extend(
                self._validate_basic_properties(feature_name, data, expectations)
            )

            # Statistical validation
            results.extend(
                self._validate_statistical_properties(feature_name, data, expectations)
            )

            # Distribution validation
            results.extend(
                self._validate_distribution(feature_name, data, expectations)
            )

            # Outlier detection
            results.extend(self._validate_outliers(feature_name, data, expectations))

            # Great Expectations validation (if available)
            if self.enable_ge:
                results.extend(
                    self._validate_with_great_expectations(
                        feature_name, data, expectations
                    )
                )

        except Exception as e:
            logger.error("Failed to validate feature %s: %s", feature_name, e)
            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="validation_error",
                    passed=False,
                    value=None,
                    expected_min=None,
                    expected_max=None,
                    message=f"Validation failed with error: {e}",
                    timestamp=datetime.now(),
                    severity="critical",
                )
            )

        return results

    def _validate_basic_properties(
        self, feature_name: str, data: pd.Series, expectations: FeatureExpectations
    ) -> list[ValidationResult]:
        """Validate basic properties like null percentage and data type."""
        results = []

        # Null percentage validation
        null_percentage = data.isnull().sum() / len(data)
        results.append(
            ValidationResult(
                feature_name=feature_name,
                validation_type="null_percentage",
                passed=null_percentage <= expectations.null_percentage_max,
                value=null_percentage,
                expected_min=0.0,
                expected_max=expectations.null_percentage_max,
                message=f"Null percentage: {null_percentage:.2%} (max allowed: {expectations.null_percentage_max:.2%})",
                timestamp=datetime.now(),
                severity="high"
                if null_percentage > expectations.null_percentage_max
                else "info",
            )
        )

        # Data type validation (for non-null values)
        valid_data = data.dropna()
        if len(valid_data) > 0:
            if expectations.data_type == "float":
                try:
                    pd.to_numeric(valid_data, errors="raise")
                    type_check_passed = True
                    type_message = "All non-null values are numeric"
                except (ValueError, TypeError):
                    type_check_passed = False
                    type_message = "Contains non-numeric values"
            elif expectations.data_type == "boolean":
                type_check_passed = (
                    valid_data.dtype == bool
                    or valid_data.isin([0, 1, True, False]).all()
                )
                type_message = (
                    "All values are boolean or 0/1"
                    if type_check_passed
                    else "Contains non-boolean values"
                )
            else:
                type_check_passed = True
                type_message = (
                    f"Data type validation for {expectations.data_type} not implemented"
                )

            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="data_type",
                    passed=type_check_passed,
                    value=None,
                    expected_min=None,
                    expected_max=None,
                    message=type_message,
                    timestamp=datetime.now(),
                    severity="critical" if not type_check_passed else "info",
                )
            )

        return results

    def _validate_statistical_properties(
        self, feature_name: str, data: pd.Series, expectations: FeatureExpectations
    ) -> list[ValidationResult]:
        """Validate statistical properties like mean and standard deviation."""
        results = []
        valid_data = data.dropna()

        if len(valid_data) == 0:
            return results

        # Range validation
        if expectations.min_value is not None:
            min_violation = (valid_data < expectations.min_value).sum()
            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="min_value",
                    passed=min_violation == 0,
                    value=valid_data.min(),
                    expected_min=expectations.min_value,
                    expected_max=None,
                    message=f"Minimum value: {valid_data.min():.4f} (expected >= {expectations.min_value}), {min_violation} violations",
                    timestamp=datetime.now(),
                    severity="high" if min_violation > 0 else "info",
                )
            )

        if expectations.max_value is not None:
            max_violation = (valid_data > expectations.max_value).sum()
            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="max_value",
                    passed=max_violation == 0,
                    value=valid_data.max(),
                    expected_min=None,
                    expected_max=expectations.max_value,
                    message=f"Maximum value: {valid_data.max():.4f} (expected <= {expectations.max_value}), {max_violation} violations",
                    timestamp=datetime.now(),
                    severity="high" if max_violation > 0 else "info",
                )
            )

        # Mean validation
        if expectations.mean_range is not None:
            mean_val = valid_data.mean()
            mean_min, mean_max = expectations.mean_range
            mean_in_range = mean_min <= mean_val <= mean_max
            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="mean_range",
                    passed=mean_in_range,
                    value=mean_val,
                    expected_min=mean_min,
                    expected_max=mean_max,
                    message=f"Mean: {mean_val:.4f} (expected: [{mean_min:.4f}, {mean_max:.4f}])",
                    timestamp=datetime.now(),
                    severity="medium" if not mean_in_range else "info",
                )
            )

        # Standard deviation validation
        if expectations.std_range is not None:
            std_val = valid_data.std()
            std_min, std_max = expectations.std_range
            std_in_range = std_min <= std_val <= std_max
            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="std_range",
                    passed=std_in_range,
                    value=std_val,
                    expected_min=std_min,
                    expected_max=std_max,
                    message=f"Standard deviation: {std_val:.4f} (expected: [{std_min:.4f}, {std_max:.4f}])",
                    timestamp=datetime.now(),
                    severity="medium" if not std_in_range else "info",
                )
            )

        return results

    def _validate_distribution(
        self, feature_name: str, data: pd.Series, expectations: FeatureExpectations
    ) -> list[ValidationResult]:
        """Validate the distribution of the feature using statistical tests."""
        results = []
        valid_data = data.dropna()

        if len(valid_data) < 30 or not SCIPY_AVAILABLE:
            return results

        try:
            if expectations.expected_distribution == "normal":
                # Shapiro-Wilk test for normality (for smaller samples)
                if len(valid_data) <= 5000:
                    stat, p_value = stats.shapiro(valid_data)
                    test_name = "Shapiro-Wilk"
                else:
                    # Anderson-Darling test for larger samples
                    stat, critical_values, significance_level = stats.anderson(
                        valid_data, dist="norm"
                    )
                    p_value = (
                        0.05 if stat > critical_values[2] else 0.15
                    )  # Rough approximation
                    test_name = "Anderson-Darling"

                normality_passed = p_value > 0.05
                results.append(
                    ValidationResult(
                        feature_name=feature_name,
                        validation_type="normality_test",
                        passed=normality_passed,
                        value=p_value,
                        expected_min=0.05,
                        expected_max=1.0,
                        message=f"{test_name} normality test: p={p_value:.4f} (expected normal distribution)",
                        timestamp=datetime.now(),
                        severity="low" if not normality_passed else "info",
                    )
                )

            elif expectations.expected_distribution == "uniform":
                # Kolmogorov-Smirnov test against uniform distribution
                stat, p_value = stats.kstest(valid_data, "uniform")
                uniform_passed = p_value > 0.05
                results.append(
                    ValidationResult(
                        feature_name=feature_name,
                        validation_type="uniformity_test",
                        passed=uniform_passed,
                        value=p_value,
                        expected_min=0.05,
                        expected_max=1.0,
                        message=f"KS uniformity test: p={p_value:.4f} (expected uniform distribution)",
                        timestamp=datetime.now(),
                        severity="low" if not uniform_passed else "info",
                    )
                )

        except Exception as e:
            logger.warning("Distribution test failed for %s: %s", feature_name, e)

        return results

    def _validate_outliers(
        self, feature_name: str, data: pd.Series, expectations: FeatureExpectations
    ) -> list[ValidationResult]:
        """Detect and validate outlier percentage."""
        results = []
        valid_data = data.dropna()

        if len(valid_data) < 3:
            return results

        # Statistical outlier detection (z-score method)
        z_scores = np.abs(stats.zscore(valid_data))
        outliers_count = (z_scores > self.outlier_std_threshold).sum()
        outlier_percentage = outliers_count / len(valid_data)

        outlier_check_passed = outlier_percentage <= expectations.outlier_percentage_max
        results.append(
            ValidationResult(
                feature_name=feature_name,
                validation_type="outlier_percentage",
                passed=outlier_check_passed,
                value=outlier_percentage,
                expected_min=0.0,
                expected_max=expectations.outlier_percentage_max,
                message=f"Outlier percentage: {outlier_percentage:.2%} ({outliers_count} outliers, max allowed: {expectations.outlier_percentage_max:.2%})",
                timestamp=datetime.now(),
                severity="medium" if not outlier_check_passed else "info",
            )
        )

        # IQR-based outlier detection
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        iqr_outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
        iqr_outlier_percentage = iqr_outliers / len(valid_data)

        iqr_check_passed = iqr_outlier_percentage <= expectations.outlier_percentage_max
        results.append(
            ValidationResult(
                feature_name=feature_name,
                validation_type="iqr_outlier_percentage",
                passed=iqr_check_passed,
                value=iqr_outlier_percentage,
                expected_min=0.0,
                expected_max=expectations.outlier_percentage_max,
                message=f"IQR outlier percentage: {iqr_outlier_percentage:.2%} ({iqr_outliers} outliers)",
                timestamp=datetime.now(),
                severity="medium" if not iqr_check_passed else "info",
            )
        )

        return results

    def _validate_with_great_expectations(
        self, feature_name: str, data: pd.Series, expectations: FeatureExpectations
    ) -> list[ValidationResult]:
        """Validate using Great Expectations framework."""
        if not GE_AVAILABLE:
            return []

        results = []

        try:
            # Create a temporary DataFrame for Great Expectations
            df = pd.DataFrame({feature_name: data})
            ge_df = PandasDataset(df)

            # Apply expectations based on feature configuration
            if expectations.min_value is not None:
                result = ge_df.expect_column_values_to_be_between(
                    feature_name,
                    min_value=expectations.min_value,
                    max_value=expectations.max_value,
                    mostly=0.95,
                )

                results.append(
                    ValidationResult(
                        feature_name=feature_name,
                        validation_type="ge_value_range",
                        passed=result.success,
                        value=result.result.get("unexpected_percent", 0.0),
                        expected_min=0.0,
                        expected_max=5.0,
                        message=f"GE value range check: {result.result.get('unexpected_count', 0)} unexpected values",
                        timestamp=datetime.now(),
                        severity="medium" if not result.success else "info",
                    )
                )

            # Null value expectation
            result = ge_df.expect_column_values_to_not_be_null(
                feature_name, mostly=1.0 - expectations.null_percentage_max
            )

            results.append(
                ValidationResult(
                    feature_name=feature_name,
                    validation_type="ge_not_null",
                    passed=result.success,
                    value=result.result.get("unexpected_percent", 0.0),
                    expected_min=0.0,
                    expected_max=expectations.null_percentage_max * 100,
                    message=f"GE null check: {result.result.get('unexpected_percent', 0.0):.1f}% null values",
                    timestamp=datetime.now(),
                    severity="high" if not result.success else "info",
                )
            )

        except Exception as e:
            logger.warning(
                "Great Expectations validation failed for %s: %s", feature_name, e
            )

        return results

    def validate_all_features(
        self,
        season: str = CURRENT_SEASON,
        gameweek: int | None = None,
        sample_size: int | None = None,
    ) -> dict[str, list[ValidationResult]]:
        """
        Validate all Sprint 01 features comprehensively.

        Args:
            season: Season to validate
            gameweek: Specific gameweek to validate (None for all)
            sample_size: Number of players to sample (None for all)

        Returns:
            Dictionary mapping feature names to validation results
        """
        logger.info("Starting comprehensive feature validation for season %s", season)

        all_results = {}

        try:
            # Get player data for validation
            query = self.dbsession.query(PlayerAttributes).filter(
                PlayerAttributes.season == season
            )

            if gameweek is not None:
                query = query.filter(PlayerAttributes.gameweek == gameweek)

            if sample_size is not None:
                query = query.limit(sample_size)

            player_data = pd.read_sql(query.statement, self.dbsession.bind)

            if len(player_data) == 0:
                msg = f"No player data found for season {season}"
                raise InsufficientDataError(msg)

            logger.info("Validating %d player records", len(player_data))

            # Validate each feature
            for feature_name in self._feature_expectations:
                if feature_name in player_data.columns:
                    logger.debug("Validating feature: %s", feature_name)
                    feature_data = player_data[feature_name]
                    results = self.validate_feature(feature_name, feature_data, season)
                    all_results[feature_name] = results

                    # Store results for later analysis
                    self.validation_results.extend(results)
                else:
                    logger.warning("Feature %s not found in player data", feature_name)
                    all_results[feature_name] = [
                        ValidationResult(
                            feature_name=feature_name,
                            validation_type="missing_feature",
                            passed=False,
                            value=None,
                            expected_min=None,
                            expected_max=None,
                            message=f"Feature {feature_name} not found in database",
                            timestamp=datetime.now(),
                            severity="critical",
                        )
                    ]

            logger.info(
                "Feature validation completed for %d features", len(all_results)
            )

        except Exception as e:
            logger.error("Feature validation failed: %s", e)
            msg = f"Validation failed: {e}"
            raise FeatureValidationError(msg)

        return all_results

    def get_validation_summary(self) -> dict[str, Any]:
        """Get a summary of all validation results."""
        if not self.validation_results:
            return {"status": "no_validations_performed"}

        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r.passed)

        # Group by severity
        severity_counts = {}
        for result in self.validation_results:
            if not result.passed:
                severity_counts[result.severity] = (
                    severity_counts.get(result.severity, 0) + 1
                )

        # Group by feature
        feature_summary = {}
        for result in self.validation_results:
            if result.feature_name not in feature_summary:
                feature_summary[result.feature_name] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                }

            feature_summary[result.feature_name]["total"] += 1
            if result.passed:
                feature_summary[result.feature_name]["passed"] += 1
            else:
                feature_summary[result.feature_name]["failed"] += 1

        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": total_validations - passed_validations,
            "pass_rate": passed_validations / total_validations,
            "meets_threshold": (passed_validations / total_validations)
            >= self.validation_threshold,
            "severity_counts": severity_counts,
            "feature_summary": feature_summary,
            "validation_timestamp": datetime.now().isoformat(),
        }

    def export_validation_report(self, output_path: str, format: str = "json") -> str:
        """
        Export comprehensive validation report.

        Args:
            output_path: Path to save the report
            format: Output format ("json", "html", "csv")

        Returns:
            Path to the exported report
        """
        summary = self.get_validation_summary()

        if format == "json":
            report_data = {
                "summary": summary,
                "detailed_results": [
                    result.to_dict() for result in self.validation_results
                ],
                "feature_expectations": {
                    name: {
                        "data_type": exp.data_type,
                        "min_value": exp.min_value,
                        "max_value": exp.max_value,
                        "mean_range": exp.mean_range,
                        "std_range": exp.std_range,
                        "null_percentage_max": exp.null_percentage_max,
                        "expected_distribution": exp.expected_distribution,
                    }
                    for name, exp in self._feature_expectations.items()
                },
            }

            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        elif format == "csv":
            results_df = pd.DataFrame(
                [result.to_dict() for result in self.validation_results]
            )
            results_df.to_csv(output_path, index=False)

        elif format == "html":
            self._generate_html_report(output_path, summary)

        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

        logger.info("Validation report exported to %s", output_path)
        return output_path

    def _generate_html_report(self, output_path: str, summary: dict[str, Any]) -> None:
        """Generate HTML validation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIrsenal Feature Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .feature {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .passed {{ border-color: #28a745; }}
                .failed {{ border-color: #dc3545; }}
                .critical {{ background-color: #f8d7da; }}
                .high {{ background-color: #fff3cd; }}
                .medium {{ background-color: #cff4fc; }}
                .low {{ background-color: #f8f9fa; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>AIrsenal Feature Validation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Validations:</strong> {summary["total_validations"]}</p>
                <p><strong>Pass Rate:</strong> {summary["pass_rate"]:.1%}</p>
                <p><strong>Meets Threshold:</strong> {"✅ Yes" if summary["meets_threshold"] else "❌ No"}</p>
                <p><strong>Generated:</strong> {summary["validation_timestamp"]}</p>
            </div>

            <h2>Feature Summary</h2>
            <table>
                <tr>
                    <th>Feature Name</th>
                    <th>Total Tests</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Pass Rate</th>
                </tr>
        """

        for feature_name, stats in summary["feature_summary"].items():
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            html_content += f"""
                <tr>
                    <td>{feature_name}</td>
                    <td>{stats["total"]}</td>
                    <td>{stats["passed"]}</td>
                    <td>{stats["failed"]}</td>
                    <td>{pass_rate:.1%}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>Detailed Results</h2>
        """

        for result in self.validation_results:
            status_class = "passed" if result.passed else "failed"
            severity_class = result.severity if not result.passed else ""

            html_content += f"""
            <div class="feature {status_class} {severity_class}">
                <h3>{result.feature_name} - {result.validation_type}</h3>
                <p><strong>Status:</strong> {"✅ Passed" if result.passed else "❌ Failed"}</p>
                <p><strong>Message:</strong> {result.message}</p>
                <p><strong>Severity:</strong> {result.severity}</p>
                <p><strong>Timestamp:</strong> {result.timestamp}</p>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)


class StatisticalTestSuite:
    """
    Comprehensive statistical testing suite for data quality validation.

    Provides advanced statistical tests for:
    - Distribution conformity testing
    - Stationarity analysis
    - Seasonality detection
    - Anomaly detection
    - Data quality scoring
    """

    def __init__(self, dbsession: Session = session, confidence_level: float = 0.95):
        """
        Initialize the StatisticalTestSuite.

        Args:
            dbsession: SQLAlchemy session for database operations
            confidence_level: Confidence level for statistical tests
        """
        self.dbsession = dbsession
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level

        # Test results storage
        self.test_results: list[dict[str, Any]] = []

    def run_comprehensive_tests(
        self, feature_name: str, data: pd.Series, season: str = CURRENT_SEASON
    ) -> dict[str, Any]:
        """
        Run comprehensive statistical tests on a feature.

        Args:
            feature_name: Name of the feature being tested
            data: Feature data to test
            season: Season context for testing

        Returns:
            Dictionary containing all test results
        """
        results = {
            "feature_name": feature_name,
            "season": season,
            "sample_size": len(data),
            "null_count": data.isnull().sum(),
            "tested_at": datetime.now().isoformat(),
            "tests": {},
        }

        valid_data = data.dropna()
        if len(valid_data) < 10:
            results["tests"]["insufficient_data"] = {
                "error": "Insufficient data for statistical testing"
            }
            return results

        # Descriptive statistics
        results["descriptive_stats"] = self._calculate_descriptive_stats(valid_data)

        # Distribution tests
        results["tests"]["distribution"] = self._test_distribution(valid_data)

        # Stationarity tests (if time series data available)
        results["tests"]["stationarity"] = self._test_stationarity(feature_name, season)

        # Outlier tests
        results["tests"]["outliers"] = self._test_outliers(valid_data)

        # Quality scoring
        results["quality_score"] = self._calculate_quality_score(results)

        self.test_results.append(results)
        return results

    def _calculate_descriptive_stats(self, data: pd.Series) -> dict[str, float]:
        """Calculate comprehensive descriptive statistics."""
        try:
            return {
                "count": float(len(data)),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "variance": float(data.var()),
                "min": float(data.min()),
                "max": float(data.max()),
                "range": float(data.max() - data.min()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
                "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "cv": float(data.std() / data.mean())
                if data.mean() != 0
                else float("inf"),
            }

        except Exception as e:
            logger.warning("Failed to calculate descriptive statistics: %s", e)
            return {}

    def _test_distribution(self, data: pd.Series) -> dict[str, Any]:
        """Test distribution properties."""
        tests = {}

        if not SCIPY_AVAILABLE:
            tests["error"] = "scipy not available for distribution testing"
            return tests

        try:
            # Normality tests
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                tests["shapiro_wilk"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > self.alpha,
                    "test": "Shapiro-Wilk normality test",
                }

            # Kolmogorov-Smirnov test against normal distribution
            standardized = (data - data.mean()) / data.std()
            ks_stat, ks_p = stats.kstest(standardized, "norm")
            tests["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "is_normal": ks_p > self.alpha,
                "test": "KS test against normal distribution",
            }

            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(data, dist="norm")
            tests["anderson_darling"] = {
                "statistic": float(ad_stat),
                "critical_values": ad_critical.tolist(),
                "significance_levels": ad_significance.tolist(),
                "is_normal": ad_stat < ad_critical[2],  # 5% significance level
                "test": "Anderson-Darling normality test",
            }

            # D'Agostino's normality test
            dag_stat, dag_p = stats.normaltest(data)
            tests["dagostino"] = {
                "statistic": float(dag_stat),
                "p_value": float(dag_p),
                "is_normal": dag_p > self.alpha,
                "test": "D'Agostino normality test",
            }

            # Test for uniform distribution
            uniform_min, uniform_max = data.min(), data.max()
            uniform_data = (data - uniform_min) / (uniform_max - uniform_min)
            uniform_ks_stat, uniform_ks_p = stats.kstest(uniform_data, "uniform")
            tests["uniform_test"] = {
                "statistic": float(uniform_ks_stat),
                "p_value": float(uniform_ks_p),
                "is_uniform": uniform_ks_p > self.alpha,
                "test": "KS test against uniform distribution",
            }

        except Exception as e:
            tests["error"] = f"Distribution testing failed: {e}"
            logger.warning("Distribution testing failed: %s", e)

        return tests

    def _test_stationarity(self, feature_name: str, season: str) -> dict[str, Any]:
        """Test stationarity of time series data."""
        tests = {}

        try:
            # Get time series data for the feature
            query = (
                self.dbsession.query(
                    PlayerAttributes.gameweek, getattr(PlayerAttributes, feature_name)
                )
                .filter(
                    PlayerAttributes.season == season,
                    getattr(PlayerAttributes, feature_name).isnot(None),
                )
                .order_by(PlayerAttributes.gameweek)
            )

            time_series_data = pd.read_sql(query.statement, self.dbsession.bind)

            if len(time_series_data) < 20:
                tests["insufficient_data"] = (
                    "Not enough time series data for stationarity testing"
                )
                return tests

            # Group by gameweek and calculate mean
            ts_mean = time_series_data.groupby("gameweek")[feature_name].mean()

            if len(ts_mean) < 10:
                tests["insufficient_data"] = (
                    "Not enough gameweeks for stationarity testing"
                )
                return tests

            if SCIPY_AVAILABLE:
                try:
                    # Simple stationarity test using rolling statistics
                    window = min(5, len(ts_mean) // 3)
                    rolling_mean = ts_mean.rolling(window=window).mean()
                    rolling_std = ts_mean.rolling(window=window).std()

                    # Test if rolling mean and std are relatively constant
                    mean_variation = (
                        rolling_mean.std() / ts_mean.mean()
                        if ts_mean.mean() != 0
                        else float("inf")
                    )
                    std_variation = (
                        rolling_std.std() / rolling_std.mean()
                        if rolling_std.mean() != 0
                        else float("inf")
                    )

                    tests["rolling_statistics"] = {
                        "mean_variation_coefficient": float(mean_variation),
                        "std_variation_coefficient": float(std_variation),
                        "is_stationary": mean_variation < 0.1 and std_variation < 0.2,
                        "test": "Rolling statistics stationarity check",
                    }

                except Exception as e:
                    tests["error"] = f"Stationarity testing failed: {e}"

        except Exception as e:
            tests["error"] = f"Time series data retrieval failed: {e}"
            logger.warning("Stationarity testing failed for %s: %s", feature_name, e)

        return tests

    def _test_outliers(self, data: pd.Series) -> dict[str, Any]:
        """Comprehensive outlier detection tests."""
        tests = {}

        try:
            # Z-score based outliers
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > 3).sum()
            tests["z_score"] = {
                "outlier_count": int(z_outliers),
                "outlier_percentage": float(z_outliers / len(data)),
                "max_z_score": float(z_scores.max()),
                "test": "Z-score outlier detection (threshold=3)",
            }

            # IQR based outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            tests["iqr"] = {
                "outlier_count": int(iqr_outliers),
                "outlier_percentage": float(iqr_outliers / len(data)),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "test": "IQR outlier detection (1.5*IQR)",
            }

            # Modified Z-score (more robust)
            median = data.median()
            mad = (data - median).abs().median()
            modified_z_scores = (
                0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
            )
            modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()

            tests["modified_z_score"] = {
                "outlier_count": int(modified_z_outliers),
                "outlier_percentage": float(modified_z_outliers / len(data)),
                "max_modified_z": float(np.abs(modified_z_scores).max()),
                "test": "Modified Z-score outlier detection (threshold=3.5)",
            }

            if SCIPY_AVAILABLE:
                # Isolation Forest would go here if sklearn was available
                pass

        except Exception as e:
            tests["error"] = f"Outlier testing failed: {e}"
            logger.warning("Outlier testing failed: %s", e)

        return tests

    def _calculate_quality_score(self, test_results: dict[str, Any]) -> float:
        """Calculate an overall data quality score based on test results."""
        try:
            score = 1.0

            # Penalize for high null percentage
            null_percentage = test_results["null_count"] / test_results["sample_size"]
            score *= max(0.0, 1.0 - null_percentage * 2)  # Heavy penalty for nulls

            # Penalize for high outlier percentage
            outlier_tests = test_results["tests"].get("outliers", {})
            if "iqr" in outlier_tests:
                outlier_percentage = outlier_tests["iqr"]["outlier_percentage"]
                score *= max(0.0, 1.0 - outlier_percentage * 10)  # Penalty for outliers

            # Bonus for good statistical properties
            distribution_tests = test_results["tests"].get("distribution", {})
            if distribution_tests:
                # Count passing normality tests
                normality_tests = [
                    "shapiro_wilk",
                    "kolmogorov_smirnov",
                    "anderson_darling",
                    "dagostino",
                ]
                passing_tests = sum(
                    1
                    for test in normality_tests
                    if test in distribution_tests
                    and distribution_tests[test].get("is_normal", False)
                )
                score *= 1.0 + passing_tests * 0.05  # Small bonus for good distribution

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning("Quality score calculation failed: %s", e)
            return 0.5  # Default neutral score


class CorrelationAnalyzer:
    """
    Analyze correlations and relationships between features.

    Provides comprehensive correlation analysis including:
    - Pearson and Spearman correlations
    - Feature importance analysis
    - Multicollinearity detection
    - Network analysis of feature relationships
    """

    def __init__(self, dbsession: Session = session):
        """
        Initialize the CorrelationAnalyzer.

        Args:
            dbsession: SQLAlchemy session for database operations
        """
        self.dbsession = dbsession
        self.correlation_results: dict[str, Any] = {}

    def analyze_feature_correlations(
        self, features: list[str], season: str = CURRENT_SEASON, method: str = "both"
    ) -> dict[str, Any]:
        """
        Analyze correlations between multiple features.

        Args:
            features: List of feature names to analyze
            season: Season to analyze
            method: Correlation method ("pearson", "spearman", "both")

        Returns:
            Dictionary containing correlation analysis results
        """
        logger.info("Analyzing correlations for %d features", len(features))

        try:
            # Get feature data
            feature_data = self._get_feature_data(features, season)

            if feature_data.empty:
                msg = "No feature data available for correlation analysis"
                raise InsufficientDataError(msg)

            results = {
                "features": features,
                "season": season,
                "sample_size": len(feature_data),
                "analyzed_at": datetime.now().isoformat(),
                "correlations": {},
            }

            # Calculate correlations
            if method in ["pearson", "both"]:
                pearson_corr = feature_data.corr(method="pearson")
                results["correlations"]["pearson"] = {
                    "matrix": pearson_corr.to_dict(),
                    "significant_pairs": self._find_significant_correlations(
                        pearson_corr, feature_data, "pearson"
                    ),
                }

            if method in ["spearman", "both"]:
                spearman_corr = feature_data.corr(method="spearman")
                results["correlations"]["spearman"] = {
                    "matrix": spearman_corr.to_dict(),
                    "significant_pairs": self._find_significant_correlations(
                        spearman_corr, feature_data, "spearman"
                    ),
                }

            # Multicollinearity analysis
            results["multicollinearity"] = self._analyze_multicollinearity(feature_data)

            # Feature clustering
            results["feature_clusters"] = self._cluster_features(feature_data)

            self.correlation_results[season] = results
            return results

        except InsufficientDataError:
            raise  # Re-raise InsufficientDataError without wrapping
        except Exception as e:
            logger.error("Correlation analysis failed: %s", e)
            msg = f"Correlation analysis failed: {e}"
            raise FeatureValidationError(msg)

    def _get_feature_data(self, features: list[str], season: str) -> pd.DataFrame:
        """Get feature data for correlation analysis."""
        try:
            # Build query to get all requested features
            columns = [PlayerAttributes.player_id, PlayerAttributes.gameweek]
            for feature in features:
                if hasattr(PlayerAttributes, feature):
                    columns.append(getattr(PlayerAttributes, feature))
                else:
                    logger.warning("Feature %s not found in PlayerAttributes", feature)

            query = self.dbsession.query(*columns).filter(
                PlayerAttributes.season == season
            )

            # Convert to DataFrame
            feature_data = pd.read_sql(query.statement, self.dbsession.bind)

            # Drop player_id and gameweek for correlation analysis
            correlation_columns = [
                col
                for col in feature_data.columns
                if col not in ["player_id", "gameweek"] and col in features
            ]

            return feature_data[correlation_columns].dropna()

        except Exception as e:
            logger.error("Failed to get feature data: %s", e)
            return pd.DataFrame()

    def _find_significant_correlations(
        self, corr_matrix: pd.DataFrame, data: pd.DataFrame, method: str
    ) -> list[dict[str, Any]]:
        """Find statistically significant correlations."""
        significant_pairs = []

        if not SCIPY_AVAILABLE:
            return significant_pairs

        n = len(data)
        features = corr_matrix.columns.tolist()

        for i, feature1 in enumerate(features):
            for _j, feature2 in enumerate(features[i + 1 :], i + 1):
                corr_value = corr_matrix.loc[feature1, feature2]

                # Calculate p-value for correlation
                if method == "pearson":
                    # t-test for Pearson correlation
                    t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    # Approximate p-value for Spearman (would need scipy.stats.spearmanr for exact)
                    t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                if abs(corr_value) > 0.1 and p_value < 0.05:  # Significant correlation
                    significant_pairs.append(
                        {
                            "feature1": feature1,
                            "feature2": feature2,
                            "correlation": float(corr_value),
                            "p_value": float(p_value),
                            "strength": self._categorize_correlation_strength(
                                abs(corr_value)
                            ),
                            "direction": "positive" if corr_value > 0 else "negative",
                        }
                    )

        # Sort by absolute correlation value
        significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return significant_pairs

    def _categorize_correlation_strength(self, abs_corr: float) -> str:
        """Categorize correlation strength."""
        if abs_corr >= 0.8:
            return "very_strong"
        if abs_corr >= 0.6:
            return "strong"
        if abs_corr >= 0.4:
            return "moderate"
        if abs_corr >= 0.2:
            return "weak"
        return "very_weak"

    def _analyze_multicollinearity(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze multicollinearity using VIF and condition number."""
        try:
            # Condition number of correlation matrix
            corr_matrix = data.corr()
            eigenvalues = np.linalg.eigvals(corr_matrix)
            condition_number = np.sqrt(eigenvalues.max() / eigenvalues.min())

            # Find highly correlated pairs (potential multicollinearity)
            high_corr_pairs = []
            features = corr_matrix.columns.tolist()

            for i, feature1 in enumerate(features):
                for _j, feature2 in enumerate(features[i + 1 :], i + 1):
                    corr_value = abs(corr_matrix.loc[feature1, feature2])
                    if corr_value > 0.8:  # High correlation threshold
                        high_corr_pairs.append(
                            {
                                "feature1": feature1,
                                "feature2": feature2,
                                "correlation": float(corr_value),
                            }
                        )

            return {
                "condition_number": float(condition_number),
                "multicollinearity_risk": "high"
                if condition_number > 30
                else "medium"
                if condition_number > 15
                else "low",
                "highly_correlated_pairs": high_corr_pairs,
                "eigenvalues": eigenvalues.tolist(),
            }

        except Exception as e:
            logger.warning("Multicollinearity analysis failed: %s", e)
            return {"error": str(e)}

    def _cluster_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Cluster features based on correlation patterns."""
        try:
            if not SCIPY_AVAILABLE:
                return {"error": "scipy not available for clustering"}

            # Use correlation distance for clustering
            corr_matrix = data.corr()
            distance_matrix = 1 - abs(corr_matrix)

            # Convert to condensed distance matrix
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            condensed_distances = squareform(distance_matrix, checks=False)

            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method="ward")

            # Get clusters with different thresholds
            clusters_2 = fcluster(linkage_matrix, 2, criterion="maxclust")
            clusters_3 = fcluster(linkage_matrix, 3, criterion="maxclust")
            clusters_4 = fcluster(linkage_matrix, 4, criterion="maxclust")

            features = corr_matrix.columns.tolist()

            return {
                "2_clusters": {
                    f"cluster_{i}": [
                        features[j] for j, c in enumerate(clusters_2) if c == i
                    ]
                    for i in range(1, 3)
                },
                "3_clusters": {
                    f"cluster_{i}": [
                        features[j] for j, c in enumerate(clusters_3) if c == i
                    ]
                    for i in range(1, 4)
                },
                "4_clusters": {
                    f"cluster_{i}": [
                        features[j] for j, c in enumerate(clusters_4) if c == i
                    ]
                    for i in range(1, 5)
                },
                "linkage_matrix": linkage_matrix.tolist(),
            }

        except Exception as e:
            logger.warning("Feature clustering failed: %s", e)
            return {"error": str(e)}


class PerformanceBenchmarker:
    """
    Benchmark performance of all feature calculations to ensure Sprint 01 targets are met.

    Performance Targets:
    - Form calculation: <50ms per player
    - Weighted metrics: <50ms per player
    - xG integration: <500ms API response
    - All batch operations: <2s for 650 players
    """

    def __init__(self, dbsession: Session = session):
        """
        Initialize the PerformanceBenchmarker.

        Args:
            dbsession: SQLAlchemy session for database operations
        """
        self.dbsession = dbsession
        self.benchmark_results: list[dict[str, Any]] = []

        # Performance targets (in milliseconds)
        self.targets = {
            "form_calculation_per_player": 50,
            "weighted_performance_per_player": 50,
            "xg_integration_api": 500,
            "batch_650_players": 2000,
            "trend_detection_per_player": 100,
            "fixture_difficulty_calculation": 200,
            "rotation_risk_per_player": 75,
        }

    def run_comprehensive_benchmarks(
        self, season: str = CURRENT_SEASON, sample_sizes: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Run comprehensive performance benchmarks for all Sprint 01 features.

        Args:
            season: Season to benchmark
            sample_sizes: Different sample sizes to test

        Returns:
            Dictionary containing all benchmark results
        """
        if sample_sizes is None:
            sample_sizes = [10, 50, 100, 650]
        logger.info("Starting comprehensive performance benchmarks")

        results = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "season": season,
            "sample_sizes": sample_sizes,
            "targets": self.targets,
            "benchmarks": {},
        }

        # Get sample player IDs for testing
        sample_players = self._get_sample_players(season, max(sample_sizes))

        for sample_size in sample_sizes:
            logger.info("Benchmarking with sample size: %d", sample_size)

            test_players = sample_players[:sample_size]
            size_results = {}

            # Form calculation benchmarks
            size_results["form_calculation"] = self._benchmark_form_calculation(
                test_players, season
            )

            # Weighted performance benchmarks
            size_results["weighted_performance"] = self._benchmark_weighted_performance(
                test_players, season
            )

            # Trend detection benchmarks
            size_results["trend_detection"] = self._benchmark_trend_detection(
                test_players, season
            )

            # Batch operation benchmarks
            size_results["batch_operations"] = self._benchmark_batch_operations(
                test_players, season
            )

            # Feature calculation benchmarks
            size_results["feature_calculations"] = self._benchmark_feature_calculations(
                test_players, season
            )

            results["benchmarks"][f"sample_{sample_size}"] = size_results

        # Calculate overall performance summary
        results["summary"] = self._calculate_performance_summary(results)

        self.benchmark_results.append(results)
        return results

    def _get_sample_players(self, season: str, max_size: int) -> list[int]:
        """Get a sample of player IDs for benchmarking."""
        try:
            query = (
                self.dbsession.query(PlayerAttributes.player_id)
                .filter(PlayerAttributes.season == season)
                .distinct()
                .limit(max_size)
            )

            return [row.player_id for row in query.all()]

        except Exception as e:
            logger.error("Failed to get sample players: %s", e)
            return []

    def _benchmark_form_calculation(
        self, player_ids: list[int], season: str
    ) -> dict[str, Any]:
        """Benchmark form calculation performance."""
        try:
            form_calculator = FormCalculator(self.dbsession)

            # Single player benchmark
            if player_ids:
                start_time = time.time()
                form_calculator.calculate_rolling_form(player_ids[0], 3, season)
                single_time = (time.time() - start_time) * 1000  # Convert to ms
            else:
                single_time = 0

            # Batch benchmark
            start_time = time.time()
            form_calculator.batch_calculate_form(player_ids, season)
            batch_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "single_player_ms": single_time,
                "batch_total_ms": batch_time,
                "batch_per_player_ms": batch_time / len(player_ids)
                if player_ids
                else 0,
                "meets_target": single_time
                < self.targets["form_calculation_per_player"],
                "target_ms": self.targets["form_calculation_per_player"],
            }

        except Exception as e:
            logger.error("Form calculation benchmark failed: %s", e)
            return {"error": str(e)}

    def _benchmark_weighted_performance(
        self, player_ids: list[int], season: str
    ) -> dict[str, Any]:
        """Benchmark weighted performance calculation."""
        try:
            weighted_calculator = WeightedPerformanceCalculator(
                dbsession=self.dbsession
            )

            if not player_ids:
                return {"error": "No player IDs provided"}

            # Get sample player score data for testing
            player_score = (
                self.dbsession.query(PlayerScore)
                .filter(PlayerScore.player_id == player_ids[0])
                .first()
            )

            if not player_score:
                return {"error": "No player score data found"}

            # Get player position
            player_attr = (
                self.dbsession.query(PlayerAttributes)
                .filter(
                    PlayerAttributes.player_id == player_ids[0],
                    PlayerAttributes.season == season,
                )
                .first()
            )

            if not player_attr:
                return {"error": "No player attributes found"}

            position = player_attr.position

            # Single calculation benchmark
            start_time = time.time()
            weighted_calculator.calculate_weighted_score(player_score, position)
            single_time = (time.time() - start_time) * 1000  # Convert to ms

            # Batch calculation benchmark
            scores_and_positions = [(player_score, position)] * len(player_ids)
            start_time = time.time()
            weighted_calculator.calculate_batch_scores(scores_and_positions)
            batch_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "single_calculation_ms": single_time,
                "batch_total_ms": batch_time,
                "batch_per_player_ms": batch_time / len(player_ids),
                "meets_target": single_time
                < self.targets["weighted_performance_per_player"],
                "target_ms": self.targets["weighted_performance_per_player"],
            }

        except Exception as e:
            logger.error("Weighted performance benchmark failed: %s", e)
            return {"error": str(e)}

    def _benchmark_trend_detection(
        self, player_ids: list[int], season: str
    ) -> dict[str, Any]:
        """Benchmark trend detection performance."""
        try:
            trend_detector = TrendDetector(self.dbsession)

            if not player_ids:
                return {"error": "No player IDs provided"}

            # Single player benchmark
            start_time = time.time()
            trend_detector.detect_trends(player_ids[0], lookback_days=30, season=season)
            single_time = (time.time() - start_time) * 1000  # Convert to ms

            # Batch benchmark (limited sample to avoid timeout)
            test_sample = player_ids[: min(10, len(player_ids))]
            start_time = time.time()
            trend_detector.batch_detect_trends(
                test_sample, lookback_days=30, season=season
            )
            batch_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "single_player_ms": single_time,
                "batch_total_ms": batch_time,
                "batch_per_player_ms": batch_time / len(test_sample),
                "meets_target": single_time
                < self.targets.get("trend_detection_per_player", 100),
                "target_ms": self.targets.get("trend_detection_per_player", 100),
            }

        except Exception as e:
            logger.error("Trend detection benchmark failed: %s", e)
            return {"error": str(e)}

    def _benchmark_batch_operations(
        self, player_ids: list[int], season: str
    ) -> dict[str, Any]:
        """Benchmark batch operations performance."""
        try:
            form_calculator = FormCalculator(self.dbsession)

            # Benchmark the target 650 players if we have that many
            test_players = player_ids[:650] if len(player_ids) >= 650 else player_ids

            start_time = time.time()
            form_calculator.batch_calculate_form(test_players, season)
            batch_time = (time.time() - start_time) * 1000  # Convert to ms

            # Scale to 650 players if testing with fewer
            if len(test_players) < 650:
                scaled_time = batch_time * (650 / len(test_players))
            else:
                scaled_time = batch_time

            return {
                "actual_players": len(test_players),
                "actual_time_ms": batch_time,
                "scaled_650_players_ms": scaled_time,
                "meets_target": scaled_time < self.targets["batch_650_players"],
                "target_ms": self.targets["batch_650_players"],
            }

        except Exception as e:
            logger.error("Batch operations benchmark failed: %s", e)
            return {"error": str(e)}

    def _benchmark_feature_calculations(
        self, player_ids: list[int], season: str
    ) -> dict[str, Any]:
        """Benchmark individual feature calculations."""
        results = {}

        if not player_ids:
            return {"error": "No player IDs provided"}

        # Test database query performance
        start_time = time.time()
        query = self.dbsession.query(PlayerAttributes).filter(
            PlayerAttributes.player_id.in_(player_ids),
            PlayerAttributes.season == season,
        )
        data = query.all()
        query_time = (time.time() - start_time) * 1000

        results["database_query"] = {
            "time_ms": query_time,
            "records_retrieved": len(data),
            "per_record_ms": query_time / len(data) if data else 0,
        }

        # Test feature extraction
        if data:
            start_time = time.time()
            feature_data = [
                {
                    "form_3_games": attr.form_3_games,
                    "form_5_games": attr.form_5_games,
                    "xg_per_90": attr.xg_per_90,
                    "is_penalty_taker": attr.is_penalty_taker,
                }
                for attr in data
            ]
            extraction_time = (time.time() - start_time) * 1000

            results["feature_extraction"] = {
                "time_ms": extraction_time,
                "features_extracted": len(feature_data) * 4,  # 4 features per record
                "per_feature_ms": extraction_time / (len(feature_data) * 4),
            }

        return results

    def _calculate_performance_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Calculate overall performance summary."""
        summary = {
            "overall_meets_targets": True,
            "failed_targets": [],
            "performance_score": 1.0,
            "fastest_sample_size": None,
            "slowest_sample_size": None,
        }

        # Analyze each sample size
        sample_performances = {}

        for sample_key, sample_results in results["benchmarks"].items():
            sample_size = int(sample_key.split("_")[1])

            # Check form calculation target
            form_results = sample_results.get("form_calculation", {})
            if not form_results.get("meets_target", True):
                summary["overall_meets_targets"] = False
                summary["failed_targets"].append(
                    f"Form calculation ({sample_size} players)"
                )

            # Check weighted performance target
            weighted_results = sample_results.get("weighted_performance", {})
            if not weighted_results.get("meets_target", True):
                summary["overall_meets_targets"] = False
                summary["failed_targets"].append(
                    f"Weighted performance ({sample_size} players)"
                )

            # Check batch operations target (for 650 players)
            if sample_size >= 650:
                batch_results = sample_results.get("batch_operations", {})
                if not batch_results.get("meets_target", True):
                    summary["overall_meets_targets"] = False
                    summary["failed_targets"].append("Batch operations (650 players)")

            # Track overall performance for this sample size
            avg_performance = 0
            performance_count = 0

            for _test_type, test_results in sample_results.items():
                if (
                    isinstance(test_results, dict)
                    and "single_player_ms" in test_results
                ):
                    avg_performance += test_results["single_player_ms"]
                    performance_count += 1

            if performance_count > 0:
                sample_performances[sample_size] = avg_performance / performance_count

        # Find fastest and slowest
        if sample_performances:
            fastest_size = min(
                sample_performances.keys(), key=lambda k: sample_performances[k]
            )
            slowest_size = max(
                sample_performances.keys(), key=lambda k: sample_performances[k]
            )

            summary["fastest_sample_size"] = {
                "size": fastest_size,
                "avg_time_ms": sample_performances[fastest_size],
            }
            summary["slowest_sample_size"] = {
                "size": slowest_size,
                "avg_time_ms": sample_performances[slowest_size],
            }

        # Calculate performance score
        if summary["overall_meets_targets"]:
            summary["performance_score"] = 1.0
        else:
            # Deduct points for each failed target
            deduction = len(summary["failed_targets"]) * 0.2
            summary["performance_score"] = max(0.0, 1.0 - deduction)

        return summary


class DataDriftDetector:
    """
    Detect data drift between seasons to identify when features may need recalibration.

    Provides comprehensive drift detection including:
    - Distribution drift using statistical tests
    - Population stability index (PSI)
    - Feature importance drift
    - Concept drift detection
    """

    def __init__(self, dbsession: Session = session):
        """
        Initialize the DataDriftDetector.

        Args:
            dbsession: SQLAlchemy session for database operations
        """
        self.dbsession = dbsession
        self.drift_results: dict[str, Any] = {}

    def detect_seasonal_drift(
        self,
        features: list[str],
        reference_season: str,
        target_season: str,
        drift_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        Detect drift between two seasons for specified features.

        Args:
            features: List of feature names to analyze
            reference_season: Reference season (baseline)
            target_season: Target season to compare against reference
            drift_threshold: Threshold for drift detection (0.1 = 10% change)

        Returns:
            Dictionary containing drift analysis results
        """
        logger.info("Detecting drift from %s to %s", reference_season, target_season)

        results = {
            "reference_season": reference_season,
            "target_season": target_season,
            "features": features,
            "drift_threshold": drift_threshold,
            "analyzed_at": datetime.now().isoformat(),
            "feature_drifts": {},
            "overall_drift_score": 0.0,
            "significant_drifts": [],
        }

        try:
            # Get data for both seasons
            reference_data = self._get_seasonal_data(features, reference_season)
            target_data = self._get_seasonal_data(features, target_season)

            if reference_data.empty or target_data.empty:
                msg = "Insufficient data for drift detection"
                raise InsufficientDataError(msg)

            total_drift_score = 0.0

            # Analyze drift for each feature
            for feature in features:
                if feature in reference_data.columns and feature in target_data.columns:
                    feature_drift = self._analyze_feature_drift(
                        feature,
                        reference_data[feature],
                        target_data[feature],
                        drift_threshold,
                    )

                    results["feature_drifts"][feature] = feature_drift
                    total_drift_score += feature_drift.get("drift_score", 0.0)

                    # Track significant drifts
                    if feature_drift.get("has_significant_drift", False):
                        results["significant_drifts"].append(
                            {
                                "feature": feature,
                                "drift_score": feature_drift["drift_score"],
                                "drift_type": feature_drift["primary_drift_type"],
                            }
                        )
                else:
                    logger.warning(
                        "Feature %s not found in one or both seasons", feature
                    )

            # Calculate overall drift score
            results["overall_drift_score"] = (
                total_drift_score / len(features) if features else 0.0
            )

            # Determine overall drift status
            results["overall_drift_status"] = self._categorize_drift_severity(
                results["overall_drift_score"]
            )

            self.drift_results[f"{reference_season}_to_{target_season}"] = results

        except Exception as e:
            logger.error("Drift detection failed: %s", e)
            results["error"] = str(e)

        return results

    def _get_seasonal_data(self, features: list[str], season: str) -> pd.DataFrame:
        """Get feature data for a specific season."""
        try:
            # Build query to get all requested features
            columns = [PlayerAttributes.player_id]
            for feature in features:
                if hasattr(PlayerAttributes, feature):
                    columns.append(getattr(PlayerAttributes, feature))

            query = self.dbsession.query(*columns).filter(
                PlayerAttributes.season == season
            )

            # Convert to DataFrame
            data = pd.read_sql(query.statement, self.dbsession.bind)

            # Drop player_id column and keep only feature columns
            feature_columns = [
                col for col in data.columns if col != "player_id" and col in features
            ]
            return data[feature_columns].dropna()

        except Exception as e:
            logger.error("Failed to get seasonal data for %s: %s", season, e)
            return pd.DataFrame()

    def _analyze_feature_drift(
        self,
        feature_name: str,
        reference_data: pd.Series,
        target_data: pd.Series,
        drift_threshold: float,
    ) -> dict[str, Any]:
        """Analyze drift for a single feature."""
        drift_analysis = {
            "feature_name": feature_name,
            "reference_samples": len(reference_data),
            "target_samples": len(target_data),
            "drift_tests": {},
            "drift_score": 0.0,
            "has_significant_drift": False,
            "primary_drift_type": "none",
        }

        # Basic statistical comparison
        drift_analysis["statistics"] = {
            "reference_mean": float(reference_data.mean()),
            "target_mean": float(target_data.mean()),
            "reference_std": float(reference_data.std()),
            "target_std": float(target_data.std()),
            "mean_change_percent": float(
                (target_data.mean() - reference_data.mean())
                / reference_data.mean()
                * 100
            )
            if reference_data.mean() != 0
            else 0.0,
            "std_change_percent": float(
                (target_data.std() - reference_data.std()) / reference_data.std() * 100
            )
            if reference_data.std() != 0
            else 0.0,
        }

        # Distribution drift tests
        drift_analysis["drift_tests"]["distribution"] = self._test_distribution_drift(
            reference_data, target_data
        )

        # Population Stability Index (PSI)
        drift_analysis["drift_tests"]["psi"] = self._calculate_psi(
            reference_data, target_data
        )

        # Kolmogorov-Smirnov test
        if SCIPY_AVAILABLE:
            drift_analysis["drift_tests"]["ks_test"] = self._perform_ks_test(
                reference_data, target_data
            )

        # Calculate overall drift score
        drift_score = self._calculate_feature_drift_score(drift_analysis)
        drift_analysis["drift_score"] = drift_score

        # Determine if drift is significant
        drift_analysis["has_significant_drift"] = drift_score > drift_threshold

        # Identify primary drift type
        drift_analysis["primary_drift_type"] = self._identify_primary_drift_type(
            drift_analysis
        )

        return drift_analysis

    def _test_distribution_drift(
        self, reference_data: pd.Series, target_data: pd.Series
    ) -> dict[str, Any]:
        """Test for distribution drift using various methods."""
        tests = {}

        try:
            # Compare quantiles
            reference_quantiles = reference_data.quantile([0.25, 0.5, 0.75]).values
            target_quantiles = target_data.quantile([0.25, 0.5, 0.75]).values

            quantile_changes = (
                np.abs(target_quantiles - reference_quantiles) / reference_quantiles
            )

            tests["quantile_drift"] = {
                "q25_change_percent": float(quantile_changes[0] * 100),
                "median_change_percent": float(quantile_changes[1] * 100),
                "q75_change_percent": float(quantile_changes[2] * 100),
                "max_quantile_change": float(quantile_changes.max()),
            }

            # Compare histograms
            # Create bins based on reference data
            bins = np.histogram_bin_edges(reference_data, bins=10)
            ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
            target_hist, _ = np.histogram(target_data, bins=bins, density=True)

            # Calculate histogram difference
            hist_diff = np.sum(np.abs(target_hist - ref_hist))

            tests["histogram_drift"] = {
                "histogram_difference": float(hist_diff),
                "bins_used": len(bins) - 1,
            }

        except Exception as e:
            tests["error"] = f"Distribution drift test failed: {e}"

        return tests

    def _calculate_psi(
        self, reference_data: pd.Series, target_data: pd.Series
    ) -> dict[str, Any]:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data deciles
            bins = reference_data.quantile(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ).values
            bins = np.concatenate([[-np.inf], bins, [np.inf]])

            # Calculate frequencies for each bin
            ref_freq = np.histogram(reference_data, bins=bins)[0]
            target_freq = np.histogram(target_data, bins=bins)[0]

            # Convert to proportions
            ref_prop = ref_freq / len(reference_data)
            target_prop = target_freq / len(target_data)

            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            target_prop = np.where(target_prop == 0, 0.0001, target_prop)

            # Calculate PSI
            psi = np.sum((target_prop - ref_prop) * np.log(target_prop / ref_prop))

            # Interpret PSI
            if psi < 0.1:
                interpretation = "no_drift"
            elif psi < 0.2:
                interpretation = "moderate_drift"
            else:
                interpretation = "significant_drift"

            return {
                "psi_value": float(psi),
                "interpretation": interpretation,
                "bins_used": len(bins) - 1,
            }

        except Exception as e:
            return {"error": f"PSI calculation failed: {e}"}

    def _perform_ks_test(
        self, reference_data: pd.Series, target_data: pd.Series
    ) -> dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for distribution similarity."""
        try:
            ks_stat, p_value = stats.ks_2samp(reference_data, target_data)

            return {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "distributions_similar": p_value > 0.05,
                "test": "Two-sample Kolmogorov-Smirnov test",
            }

        except Exception as e:
            return {"error": f"KS test failed: {e}"}

    def _calculate_feature_drift_score(self, drift_analysis: dict[str, Any]) -> float:
        """Calculate overall drift score for a feature."""
        try:
            score = 0.0

            # Weight different types of drift

            # Statistical drift (25% weight)
            stats = drift_analysis.get("statistics", {})
            mean_change = abs(stats.get("mean_change_percent", 0.0)) / 100
            std_change = abs(stats.get("std_change_percent", 0.0)) / 100

            score += 0.25 * min(1.0, (mean_change + std_change) / 2)

            # PSI drift (35% weight)
            psi_info = drift_analysis.get("drift_tests", {}).get("psi", {})
            psi_value = psi_info.get("psi_value", 0.0)
            score += 0.35 * min(1.0, psi_value / 0.2)  # Normalize by PSI threshold

            # Distribution drift (25% weight)
            dist_tests = drift_analysis.get("drift_tests", {}).get("distribution", {})
            quantile_drift = dist_tests.get("quantile_drift", {})
            max_quantile_change = quantile_drift.get("max_quantile_change", 0.0)
            score += 0.25 * min(1.0, max_quantile_change)

            # KS test drift (15% weight)
            ks_test = drift_analysis.get("drift_tests", {}).get("ks_test", {})
            if "ks_statistic" in ks_test:
                ks_stat = ks_test["ks_statistic"]
                score += 0.15 * min(1.0, ks_stat * 2)  # Scale KS statistic

            return min(1.0, score)  # Cap at 1.0

        except Exception as e:
            logger.warning("Drift score calculation failed: %s", e)
            return 0.5  # Default moderate score

    def _identify_primary_drift_type(self, drift_analysis: dict[str, Any]) -> str:
        """Identify the primary type of drift occurring."""
        try:
            stats = drift_analysis.get("statistics", {})
            mean_change = abs(stats.get("mean_change_percent", 0.0))
            std_change = abs(stats.get("std_change_percent", 0.0))

            psi_info = drift_analysis.get("drift_tests", {}).get("psi", {})
            psi_interpretation = psi_info.get("interpretation", "no_drift")

            # Determine primary drift type
            if mean_change > 20:
                return "mean_shift"
            if std_change > 30:
                return "variance_change"
            if psi_interpretation in ["moderate_drift", "significant_drift"]:
                return "distribution_change"
            if drift_analysis.get("drift_score", 0.0) > 0.1:
                return "general_drift"
            return "no_drift"

        except Exception as e:
            logger.warning("Drift type identification failed: %s", e)
            return "unknown"

    def _categorize_drift_severity(self, drift_score: float) -> str:
        """Categorize the severity of overall drift."""
        if drift_score < 0.05:
            return "no_drift"
        if drift_score < 0.1:
            return "low_drift"
        if drift_score < 0.2:
            return "moderate_drift"
        if drift_score < 0.4:
            return "high_drift"
        return "severe_drift"

    def generate_drift_report(self, output_path: str, format: str = "json") -> str:
        """
        Generate comprehensive drift detection report.

        Args:
            output_path: Path to save the report
            format: Output format ("json", "html")

        Returns:
            Path to the exported report
        """
        if not self.drift_results:
            logger.warning("No drift results available for report generation")
            return ""

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.drift_results, f, indent=2, default=str)

        elif format == "html":
            self._generate_drift_html_report(output_path)

        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

        logger.info("Drift detection report exported to %s", output_path)
        return output_path

    def _generate_drift_html_report(self, output_path: str) -> None:
        """Generate HTML drift detection report."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIrsenal Data Drift Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .drift-analysis { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .no-drift { border-left: 4px solid #28a745; }
                .low-drift { border-left: 4px solid #ffc107; }
                .moderate-drift { border-left: 4px solid #fd7e14; }
                .high-drift { border-left: 4px solid #dc3545; }
                .severe-drift { border-left: 4px solid #6f42c1; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>AIrsenal Data Drift Detection Report</h1>
        """

        for _comparison_key, drift_data in self.drift_results.items():
            reference_season = drift_data.get("reference_season", "Unknown")
            target_season = drift_data.get("target_season", "Unknown")
            overall_status = drift_data.get("overall_drift_status", "unknown")
            overall_score = drift_data.get("overall_drift_score", 0.0)

            html_content += f"""
            <div class="drift-analysis {overall_status}">
                <h2>Drift Analysis: {reference_season} → {target_season}</h2>
                <div class="summary">
                    <p><strong>Overall Drift Status:</strong> {overall_status.replace("_", " ").title()}</p>
                    <p><strong>Overall Drift Score:</strong> {overall_score:.3f}</p>
                    <p><strong>Significant Drifts:</strong> {len(drift_data.get("significant_drifts", []))}</p>
                    <p><strong>Analysis Date:</strong> {drift_data.get("analyzed_at", "Unknown")}</p>
                </div>

                <h3>Feature-Level Drift Analysis</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Drift Score</th>
                        <th>Status</th>
                        <th>Primary Drift Type</th>
                        <th>Mean Change %</th>
                        <th>PSI Value</th>
                    </tr>
            """

            for feature_name, feature_drift in drift_data.get(
                "feature_drifts", {}
            ).items():
                drift_score = feature_drift.get("drift_score", 0.0)
                has_drift = feature_drift.get("has_significant_drift", False)
                drift_type = feature_drift.get("primary_drift_type", "unknown")

                stats = feature_drift.get("statistics", {})
                mean_change = stats.get("mean_change_percent", 0.0)

                psi_info = feature_drift.get("drift_tests", {}).get("psi", {})
                psi_value = psi_info.get("psi_value", 0.0)

                status = "Significant" if has_drift else "Normal"

                html_content += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td>{drift_score:.3f}</td>
                        <td>{status}</td>
                        <td>{drift_type.replace("_", " ").title()}</td>
                        <td>{mean_change:.1f}%</td>
                        <td>{psi_value:.3f}</td>
                    </tr>
                """

            html_content += """
                </table>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)
