"""
Comprehensive test suite for the AIrsenal Feature Validation and Testing Framework.

Tests all components of TASK-113 including:
- FeatureValidator class and validation rules
- StatisticalTestSuite for data quality
- CorrelationAnalyzer for feature relationships
- PerformanceBenchmarker for speed validation
- DataDriftDetector for seasonal changes
- Integration testing with real database schemas

Author: AIrsenal Team
Version: 1.0.0
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from airsenal.framework.feature_validation import (
    CorrelationAnalyzer,
    DataDriftDetector,
    FeatureExpectations,
    FeatureValidator,
    InsufficientDataError,
    PerformanceBenchmarker,
    StatisticalTestSuite,
    ValidationResult,
)
from airsenal.framework.schema import (
    PlayerAttributes,
)


class TestFeatureValidator:
    """Test suite for the FeatureValidator class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def feature_validator(self, mock_session):
        """Create a FeatureValidator instance for testing."""
        return FeatureValidator(dbsession=mock_session, enable_ge=False)

    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing."""
        np.random.seed(42)  # For reproducible tests
        return pd.Series(
            [
                4.2,
                5.1,
                3.8,
                6.0,
                4.5,
                5.5,
                4.8,
                3.9,
                5.2,
                4.1,
                4.7,
                5.3,
                4.0,
                5.8,
                4.3,
                5.0,
                4.6,
                5.7,
                4.4,
                5.4,
                np.nan,
                4.9,
                5.1,
                4.2,
                5.6,  # Include some null values
            ],
            name="form_5_games",
        )

    def test_initialization(self, feature_validator):
        """Test FeatureValidator initialization."""
        assert feature_validator.validation_threshold == 0.95
        assert feature_validator.outlier_std_threshold == 3.0
        assert len(feature_validator._feature_expectations) > 0
        assert feature_validator.validation_results == []

    def test_feature_expectations_defined(self, feature_validator):
        """Test that all expected features have expectations defined."""
        expected_features = [
            "form_3_games",
            "form_5_games",
            "form_10_games",
            "momentum",
            "xg_per_90",
            "xa_per_90",
            "xgi_per_90",
            "next_3_fixture_difficulty",
            "next_5_fixture_difficulty",
            "is_penalty_taker",
            "is_free_kick_taker",
            "is_corner_taker",
            "role_confidence",
            "shots_per_90",
            "key_passes_per_90",
        ]

        for feature in expected_features:
            assert feature in feature_validator._feature_expectations
            expectations = feature_validator._feature_expectations[feature]
            assert isinstance(expectations, FeatureExpectations)
            assert expectations.name == feature

    def test_validate_basic_properties(self, feature_validator, sample_feature_data):
        """Test basic property validation."""
        expectations = FeatureExpectations(
            name="form_5_games",
            data_type="float",
            min_value=0.0,
            max_value=25.0,
            null_percentage_max=0.15,
        )

        results = feature_validator._validate_basic_properties(
            "form_5_games", sample_feature_data, expectations
        )

        assert len(results) == 2  # null_percentage and data_type

        # Check null percentage validation
        null_result = next(r for r in results if r.validation_type == "null_percentage")
        assert null_result.feature_name == "form_5_games"
        assert null_result.passed  # Should pass with 4% null (1/25)

        # Check data type validation
        type_result = next(r for r in results if r.validation_type == "data_type")
        assert type_result.passed

    def test_validate_statistical_properties(
        self, feature_validator, sample_feature_data
    ):
        """Test statistical property validation."""
        expectations = FeatureExpectations(
            name="form_5_games",
            data_type="float",
            min_value=0.0,
            max_value=25.0,
            mean_range=(3.0, 7.0),
            std_range=(0.5, 2.0),
        )

        results = feature_validator._validate_statistical_properties(
            "form_5_games", sample_feature_data, expectations
        )

        # Should have min, max, mean, and std validations
        assert len(results) >= 4

        # Check range validations
        min_result = next(r for r in results if r.validation_type == "min_value")
        assert min_result.passed

        max_result = next(r for r in results if r.validation_type == "max_value")
        assert max_result.passed

    def test_validate_outliers(self, feature_validator):
        """Test outlier detection validation."""
        # Create data with known outliers
        data_with_outliers = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 100])  # 100 is clear outlier

        expectations = FeatureExpectations(
            name="test_feature", data_type="float", outlier_percentage_max=0.2
        )

        results = feature_validator._validate_outliers(
            "test_feature", data_with_outliers, expectations
        )

        assert len(results) == 2  # z_score and iqr outlier detection

        # Both methods should detect the outlier
        z_result = next(r for r in results if r.validation_type == "outlier_percentage")
        iqr_result = next(
            r for r in results if r.validation_type == "iqr_outlier_percentage"
        )

        assert z_result.value > 0  # Should detect outliers
        assert iqr_result.value > 0  # Should detect outliers

    def test_validate_feature_complete(self, feature_validator, sample_feature_data):
        """Test complete feature validation."""
        results = feature_validator.validate_feature(
            "form_5_games", sample_feature_data
        )

        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.feature_name == "form_5_games" for r in results)

    def test_get_validation_summary(self, feature_validator):
        """Test validation summary generation."""
        # Add some mock results
        feature_validator.validation_results = [
            ValidationResult(
                feature_name="test_feature",
                validation_type="test",
                passed=True,
                value=1.0,
                expected_min=0.0,
                expected_max=2.0,
                message="Test passed",
                timestamp=datetime.now(),
            ),
            ValidationResult(
                feature_name="test_feature",
                validation_type="test2",
                passed=False,
                value=3.0,
                expected_min=0.0,
                expected_max=2.0,
                message="Test failed",
                timestamp=datetime.now(),
                severity="high",
            ),
        ]

        summary = feature_validator.get_validation_summary()

        assert summary["total_validations"] == 2
        assert summary["passed_validations"] == 1
        assert summary["failed_validations"] == 1
        assert summary["pass_rate"] == 0.5
        assert not summary["meets_threshold"]  # Should not meet 95% threshold
        assert "high" in summary["severity_counts"]

    def test_export_validation_report_json(self, feature_validator):
        """Test JSON report export."""
        # Add some mock results
        feature_validator.validation_results = [
            ValidationResult(
                feature_name="test_feature",
                validation_type="test",
                passed=True,
                value=1.0,
                expected_min=0.0,
                expected_max=2.0,
                message="Test passed",
                timestamp=datetime.now(),
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        result_path = feature_validator.export_validation_report(output_path, "json")

        assert result_path == output_path
        assert Path(output_path).exists()

        # Check JSON content
        with open(output_path) as f:
            report_data = json.load(f)

        assert "summary" in report_data
        assert "detailed_results" in report_data
        assert "feature_expectations" in report_data

        # Cleanup
        Path(output_path).unlink()


class TestStatisticalTestSuite:
    """Test suite for the StatisticalTestSuite class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def test_suite(self, mock_session):
        """Create a StatisticalTestSuite instance for testing."""
        return StatisticalTestSuite(dbsession=mock_session)

    @pytest.fixture
    def normal_data(self):
        """Create normally distributed test data."""
        np.random.seed(42)
        return pd.Series(np.random.normal(5.0, 1.5, 100), name="normal_feature")

    @pytest.fixture
    def uniform_data(self):
        """Create uniformly distributed test data."""
        np.random.seed(42)
        return pd.Series(np.random.uniform(0, 10, 100), name="uniform_feature")

    def test_initialization(self, test_suite):
        """Test StatisticalTestSuite initialization."""
        assert test_suite.confidence_level == 0.95
        assert abs(test_suite.alpha - 0.05) < 1e-10
        assert test_suite.test_results == []

    def test_calculate_descriptive_stats(self, test_suite, normal_data):
        """Test descriptive statistics calculation."""
        stats = test_suite._calculate_descriptive_stats(normal_data)

        required_stats = [
            "count",
            "mean",
            "median",
            "std",
            "variance",
            "min",
            "max",
            "range",
            "q25",
            "q75",
            "iqr",
            "skewness",
            "kurtosis",
            "cv",
        ]

        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], float)

        assert stats["count"] == 100
        assert 4.0 < stats["mean"] < 6.0  # Should be around 5.0
        assert stats["std"] > 0

    def test_test_distribution_normal(self, test_suite, normal_data):
        """Test distribution testing with normal data."""
        tests = test_suite._test_distribution(normal_data)

        # Should have multiple normality tests
        expected_tests = [
            "shapiro_wilk",
            "kolmogorov_smirnov",
            "anderson_darling",
            "dagostino",
        ]

        for test in expected_tests:
            if (
                test in tests
            ):  # Some tests might not run depending on scipy availability
                assert "statistic" in tests[test]
                assert "is_normal" in tests[test]
                
                # Anderson-Darling test doesn't have p_value, but critical_values
                if test == "anderson_darling":
                    assert "critical_values" in tests[test]
                    assert "significance_levels" in tests[test]
                else:
                    assert "p_value" in tests[test]

    def test_test_outliers(self, test_suite):
        """Test outlier detection."""
        # Create data with known outliers
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100])

        tests = test_suite._test_outliers(data_with_outliers)

        assert "z_score" in tests
        assert "iqr" in tests
        assert "modified_z_score" in tests

        # Should detect the outlier (100)
        assert tests["z_score"]["outlier_count"] >= 1
        assert tests["iqr"]["outlier_count"] >= 1

    def test_calculate_quality_score(self, test_suite, normal_data):
        """Test quality score calculation."""
        # Create mock test results
        test_results = {
            "sample_size": 100,
            "null_count": 5,  # 5% nulls
            "tests": {
                "outliers": {
                    "iqr": {"outlier_percentage": 0.02}  # 2% outliers
                },
                "distribution": {"shapiro_wilk": {"is_normal": True}},
            },
        }

        score = test_suite._calculate_quality_score(test_results)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent quality with low nulls/outliers

    def test_run_comprehensive_tests(self, test_suite, normal_data):
        """Test complete statistical test suite."""
        results = test_suite.run_comprehensive_tests("normal_feature", normal_data)

        assert results["feature_name"] == "normal_feature"
        assert results["sample_size"] == len(normal_data)
        assert "descriptive_stats" in results
        assert "tests" in results
        assert "quality_score" in results

        assert isinstance(results["quality_score"], float)
        assert 0.0 <= results["quality_score"] <= 1.0

    def test_insufficient_data_handling(self, test_suite):
        """Test handling of insufficient data."""
        small_data = pd.Series([1, 2, 3])  # Too small for meaningful tests

        results = test_suite.run_comprehensive_tests("small_feature", small_data)

        assert "insufficient_data" in results["tests"]


class TestCorrelationAnalyzer:
    """Test suite for the CorrelationAnalyzer class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        mock_session = Mock()

        # Mock the SQL query result
        mock_data = pd.DataFrame(
            {
                "player_id": range(1, 101),
                "gameweek": [1] * 100,
                "form_5_games": np.random.normal(5.0, 1.5, 100),
                "xg_per_90": np.random.normal(0.3, 0.2, 100),
                "shots_per_90": np.random.normal(2.0, 1.0, 100),
            }
        )

        # Make form and shots correlated for testing
        mock_data["shots_per_90"] = mock_data["form_5_games"] * 0.4 + np.random.normal(
            0, 0.3, 100
        )

        with patch("pandas.read_sql", return_value=mock_data):
            yield mock_session

    @pytest.fixture
    def correlation_analyzer(self, mock_session):
        """Create a CorrelationAnalyzer instance for testing."""
        return CorrelationAnalyzer(dbsession=mock_session)

    @pytest.fixture
    def test_features(self):
        """List of features for correlation testing."""
        return ["form_5_games", "xg_per_90", "shots_per_90"]

    def test_initialization(self, correlation_analyzer):
        """Test CorrelationAnalyzer initialization."""
        assert correlation_analyzer.correlation_results == {}

    def test_categorize_correlation_strength(self, correlation_analyzer):
        """Test correlation strength categorization."""
        assert (
            correlation_analyzer._categorize_correlation_strength(0.9) == "very_strong"
        )
        assert correlation_analyzer._categorize_correlation_strength(0.7) == "strong"
        assert correlation_analyzer._categorize_correlation_strength(0.5) == "moderate"
        assert correlation_analyzer._categorize_correlation_strength(0.3) == "weak"
        assert correlation_analyzer._categorize_correlation_strength(0.1) == "very_weak"

    @patch("pandas.read_sql")
    def test_get_feature_data(self, mock_read_sql, correlation_analyzer, test_features):
        """Test feature data retrieval."""
        # Mock return data
        mock_data = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "gameweek": [1, 1, 1],
                "form_5_games": [5.0, 6.0, 4.5],
                "xg_per_90": [0.3, 0.4, 0.2],
                "shots_per_90": [2.0, 2.5, 1.8],
            }
        )
        mock_read_sql.return_value = mock_data

        result = correlation_analyzer._get_feature_data(test_features, "2023-24")

        assert not result.empty
        assert len(result.columns) == len(test_features)
        assert all(feature in result.columns for feature in test_features)

    def test_analyze_multicollinearity(self, correlation_analyzer):
        """Test multicollinearity analysis."""
        # Create data with known correlation structure
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],  # Perfectly correlated with feature1
                "feature3": [5, 4, 3, 2, 1],  # Anti-correlated
            }
        )

        result = correlation_analyzer._analyze_multicollinearity(data)

        assert "condition_number" in result
        assert "multicollinearity_risk" in result
        assert "highly_correlated_pairs" in result

        # Should detect high correlation between feature1 and feature2
        high_corr_pairs = result["highly_correlated_pairs"]
        assert len(high_corr_pairs) > 0

    @patch("pandas.read_sql")
    def test_analyze_feature_correlations(
        self, mock_read_sql, correlation_analyzer, test_features
    ):
        """Test complete correlation analysis."""
        # Mock return data with correlations
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "player_id": range(1, 51),
                "gameweek": [1] * 50,
                "form_5_games": np.random.normal(5.0, 1.5, 50),
                "xg_per_90": np.random.normal(0.3, 0.2, 50),
                "shots_per_90": np.random.normal(2.0, 1.0, 50),
            }
        )
        mock_read_sql.return_value = data

        results = correlation_analyzer.analyze_feature_correlations(
            test_features, "2023-24", "both"
        )

        assert results["features"] == test_features
        assert results["season"] == "2023-24"
        assert "correlations" in results
        assert "pearson" in results["correlations"]
        assert "spearman" in results["correlations"]
        assert "multicollinearity" in results
        assert "feature_clusters" in results

    def test_insufficient_data_error(self, correlation_analyzer):
        """Test handling of insufficient data."""
        with (
            patch.object(
                correlation_analyzer, "_get_feature_data", return_value=pd.DataFrame()
            ),
            pytest.raises(InsufficientDataError),
        ):
            correlation_analyzer.analyze_feature_correlations(["feature1"], "2023-24")


class TestPerformanceBenchmarker:
    """Test suite for the PerformanceBenchmarker class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        mock_session = Mock()

        # Mock player query results
        mock_players = [Mock(player_id=i) for i in range(1, 11)]
        mock_session.query.return_value.filter.return_value.distinct.return_value.limit.return_value.all.return_value = mock_players

        return mock_session

    @pytest.fixture
    def benchmarker(self, mock_session):
        """Create a PerformanceBenchmarker instance for testing."""
        return PerformanceBenchmarker(dbsession=mock_session)

    def test_initialization(self, benchmarker):
        """Test PerformanceBenchmarker initialization."""
        assert benchmarker.targets["form_calculation_per_player"] == 50
        assert benchmarker.targets["weighted_performance_per_player"] == 50
        assert benchmarker.targets["batch_650_players"] == 2000
        assert benchmarker.benchmark_results == []

    def test_get_sample_players(self, benchmarker):
        """Test sample player retrieval."""
        players = benchmarker._get_sample_players("2023-24", 10)

        assert len(players) == 10
        assert all(isinstance(p, int) for p in players)

    @patch("airsenal.framework.feature_validation.FormCalculator")
    def test_benchmark_form_calculation(self, mock_form_calc, benchmarker):
        """Test form calculation benchmarking."""
        # Mock the form calculator
        mock_calculator = Mock()
        mock_form_calc.return_value = mock_calculator

        player_ids = [1, 2, 3]
        result = benchmarker._benchmark_form_calculation(player_ids, "2023-24")

        assert "single_player_ms" in result
        assert "batch_total_ms" in result
        assert "batch_per_player_ms" in result
        assert "meets_target" in result
        assert "target_ms" in result

        # Verify methods were called
        mock_calculator.calculate_rolling_form.assert_called()
        mock_calculator.batch_calculate_form.assert_called()

    def test_calculate_performance_summary(self, benchmarker):
        """Test performance summary calculation."""
        # Mock benchmark results
        mock_results = {
            "benchmarks": {
                "sample_10": {
                    "form_calculation": {"single_player_ms": 30, "meets_target": True},
                    "weighted_performance": {
                        "single_calculation_ms": 40,
                        "meets_target": True,
                    },
                },
                "sample_650": {"batch_operations": {"meets_target": False}},
            }
        }

        summary = benchmarker._calculate_performance_summary(mock_results)

        assert "overall_meets_targets" in summary
        assert "failed_targets" in summary
        assert "performance_score" in summary
        assert isinstance(summary["performance_score"], float)
        assert 0.0 <= summary["performance_score"] <= 1.0


class TestDataDriftDetector:
    """Test suite for the DataDriftDetector class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def drift_detector(self, mock_session):
        """Create a DataDriftDetector instance for testing."""
        return DataDriftDetector(dbsession=mock_session)

    @pytest.fixture
    def reference_data(self):
        """Create reference season data."""
        np.random.seed(42)
        return pd.Series(np.random.normal(5.0, 1.0, 100), name="feature")

    @pytest.fixture
    def drifted_data(self):
        """Create drifted target season data."""
        np.random.seed(123)
        return pd.Series(
            np.random.normal(6.0, 1.5, 100), name="feature"
        )  # Mean shift + variance change

    def test_initialization(self, drift_detector):
        """Test DataDriftDetector initialization."""
        assert drift_detector.drift_results == {}

    def test_calculate_psi(self, drift_detector, reference_data, drifted_data):
        """Test Population Stability Index calculation."""
        result = drift_detector._calculate_psi(reference_data, drifted_data)

        assert "psi_value" in result
        assert "interpretation" in result
        assert "bins_used" in result

        assert isinstance(result["psi_value"], float)
        assert result["interpretation"] in [
            "no_drift",
            "moderate_drift",
            "significant_drift",
        ]
        assert result["bins_used"] > 0

    def test_test_distribution_drift(
        self, drift_detector, reference_data, drifted_data
    ):
        """Test distribution drift testing."""
        result = drift_detector._test_distribution_drift(reference_data, drifted_data)

        assert "quantile_drift" in result
        assert "histogram_drift" in result

        quantile_drift = result["quantile_drift"]
        assert "q25_change_percent" in quantile_drift
        assert "median_change_percent" in quantile_drift
        assert "q75_change_percent" in quantile_drift
        assert "max_quantile_change" in quantile_drift

    def test_analyze_feature_drift(self, drift_detector, reference_data, drifted_data):
        """Test individual feature drift analysis."""
        result = drift_detector._analyze_feature_drift(
            "test_feature", reference_data, drifted_data, 0.1
        )

        assert result["feature_name"] == "test_feature"
        assert "statistics" in result
        assert "drift_tests" in result
        assert "drift_score" in result
        assert "has_significant_drift" in result
        assert "primary_drift_type" in result

        # Should detect drift due to mean shift
        assert result["drift_score"] > 0
        stats = result["statistics"]
        assert abs(stats["mean_change_percent"]) > 10  # Should detect ~20% mean change

    def test_categorize_drift_severity(self, drift_detector):
        """Test drift severity categorization."""
        assert drift_detector._categorize_drift_severity(0.02) == "no_drift"
        assert drift_detector._categorize_drift_severity(0.08) == "low_drift"
        assert drift_detector._categorize_drift_severity(0.15) == "moderate_drift"
        assert drift_detector._categorize_drift_severity(0.3) == "high_drift"
        assert drift_detector._categorize_drift_severity(0.5) == "severe_drift"

    def test_identify_primary_drift_type(self, drift_detector):
        """Test drift type identification."""
        # Test mean shift
        mean_shift_analysis = {
            "statistics": {"mean_change_percent": 25, "std_change_percent": 5},
            "drift_tests": {"psi": {"interpretation": "no_drift"}},
            "drift_score": 0.2,
        }
        assert (
            drift_detector._identify_primary_drift_type(mean_shift_analysis)
            == "mean_shift"
        )

        # Test variance change
        variance_change_analysis = {
            "statistics": {"mean_change_percent": 5, "std_change_percent": 35},
            "drift_tests": {"psi": {"interpretation": "no_drift"}},
            "drift_score": 0.2,
        }
        assert (
            drift_detector._identify_primary_drift_type(variance_change_analysis)
            == "variance_change"
        )

    @patch.object(DataDriftDetector, "_get_seasonal_data")
    def test_detect_seasonal_drift(
        self, mock_get_data, drift_detector, reference_data, drifted_data
    ):
        """Test complete seasonal drift detection."""
        # Mock data retrieval
        mock_get_data.side_effect = [
            pd.DataFrame({"feature1": reference_data}),
            pd.DataFrame({"feature1": drifted_data}),
        ]

        results = drift_detector.detect_seasonal_drift(
            ["feature1"], "2022-23", "2023-24", 0.1
        )

        assert results["reference_season"] == "2022-23"
        assert results["target_season"] == "2023-24"
        assert results["features"] == ["feature1"]
        assert "feature_drifts" in results
        assert "overall_drift_score" in results
        assert "significant_drifts" in results
        assert "overall_drift_status" in results

    def test_generate_drift_report_json(self, drift_detector):
        """Test JSON drift report generation."""
        # Add mock drift results
        drift_detector.drift_results = {
            "2022-23_to_2023-24": {
                "reference_season": "2022-23",
                "target_season": "2023-24",
                "overall_drift_score": 0.15,
                "overall_drift_status": "moderate_drift",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        result_path = drift_detector.generate_drift_report(output_path, "json")

        assert result_path == output_path
        assert Path(output_path).exists()

        # Check JSON content
        with open(output_path) as f:
            report_data = json.load(f)

        assert "2022-23_to_2023-24" in report_data

        # Cleanup
        Path(output_path).unlink()


class TestIntegration:
    """Integration tests for the complete validation framework."""

    @pytest.fixture
    def mock_session(self):
        """Create a comprehensive mock database session."""
        mock_session = Mock()

        # Mock PlayerAttributes data
        mock_data = []
        for i in range(1, 101):
            attr = Mock()
            attr.player_id = i
            attr.season = "2023-24"
            attr.gameweek = 1
            attr.form_5_games = 5.0 + np.random.normal(0, 1.0)
            attr.xg_per_90 = 0.3 + np.random.normal(0, 0.1)
            attr.is_penalty_taker = np.random.choice([True, False], p=[0.05, 0.95])
            attr.position = np.random.choice(["GK", "DEF", "MID", "FWD"])
            mock_data.append(attr)

        # Mock DataFrame for pandas.read_sql
        df_data = pd.DataFrame(
            {
                "player_id": [attr.player_id for attr in mock_data],
                "form_5_games": [attr.form_5_games for attr in mock_data],
                "xg_per_90": [attr.xg_per_90 for attr in mock_data],
                "is_penalty_taker": [attr.is_penalty_taker for attr in mock_data],
            }
        )

        with patch("pandas.read_sql", return_value=df_data):
            mock_session.query.return_value.filter.return_value.all.return_value = (
                mock_data
            )
            yield mock_session

    def test_end_to_end_validation_workflow(self, mock_session):
        """Test complete end-to-end validation workflow."""
        # Initialize all components
        validator = FeatureValidator(dbsession=mock_session, enable_ge=False)
        test_suite = StatisticalTestSuite(dbsession=mock_session)
        correlation_analyzer = CorrelationAnalyzer(dbsession=mock_session)
        benchmarker = PerformanceBenchmarker(dbsession=mock_session)
        drift_detector = DataDriftDetector(dbsession=mock_session)

        # Test feature validation
        features_to_validate = ["form_5_games", "xg_per_90", "is_penalty_taker"]

        with patch.object(validator, "validate_all_features") as mock_validate:
            mock_validate.return_value = {
                feature: [
                    ValidationResult(
                        feature_name=feature,
                        validation_type="test",
                        passed=True,
                        value=1.0,
                        expected_min=0.0,
                        expected_max=2.0,
                        message="Test passed",
                        timestamp=datetime.now(),
                    )
                ]
                for feature in features_to_validate
            }

            validation_results = validator.validate_all_features("2023-24")
            assert len(validation_results) == len(features_to_validate)

        # Test statistical analysis
        test_data = pd.Series(np.random.normal(5.0, 1.0, 100))
        stats_results = test_suite.run_comprehensive_tests("form_5_games", test_data)
        assert "quality_score" in stats_results

        # Test correlation analysis
        with patch.object(correlation_analyzer, "_get_feature_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame(
                {
                    "form_5_games": np.random.normal(5.0, 1.0, 50),
                    "xg_per_90": np.random.normal(0.3, 0.1, 50),
                }
            )

            corr_results = correlation_analyzer.analyze_feature_correlations(
                ["form_5_games", "xg_per_90"], "2023-24"
            )
            assert "correlations" in corr_results

        # Test performance benchmarking
        with (
            patch.object(
                benchmarker, "_get_sample_players", return_value=list(range(1, 11))
            ),
            patch("airsenal.framework.feature_validation.FormCalculator"),
            patch(
                "airsenal.framework.feature_validation.WeightedPerformanceCalculator"
            ),
            patch("airsenal.framework.feature_validation.TrendDetector"),
        ):
            perf_results = benchmarker.run_comprehensive_benchmarks("2023-24", [10])
            assert "summary" in perf_results

        # Test drift detection
        with patch.object(drift_detector, "_get_seasonal_data") as mock_seasonal:
            mock_seasonal.side_effect = [
                pd.DataFrame({"form_5_games": np.random.normal(5.0, 1.0, 50)}),
                pd.DataFrame({"form_5_games": np.random.normal(5.2, 1.1, 50)}),
            ]

            drift_results = drift_detector.detect_seasonal_drift(
                ["form_5_games"], "2022-23", "2023-24"
            )
            assert "overall_drift_score" in drift_results

    def test_validation_with_real_schema_expectations(self):
        """Test that validator expectations align with actual database schema."""
        validator = FeatureValidator()

        # Test that expected features exist in PlayerAttributes
        schema_attributes = [
            attr
            for attr in dir(PlayerAttributes)
            if not attr.startswith("_")
            and not callable(getattr(PlayerAttributes, attr))
        ]

        for feature_name in validator._feature_expectations:
            # Skip derived features that might not be direct columns
            if feature_name in [
                "next_3_fixture_difficulty",
                "next_5_fixture_difficulty",
            ]:
                continue

            # Check if feature exists in schema or is a computed feature
            assert (
                feature_name in schema_attributes
                or feature_name.startswith("is_")  # Role-based features
                or feature_name.endswith("_per_90")  # Rate-based features
                or feature_name
                in ["role_confidence", "momentum"]  # Special computed features
            ), f"Feature {feature_name} not found in schema"

    def test_performance_target_validation(self):
        """Test that performance targets are reasonable."""
        benchmarker = PerformanceBenchmarker()

        # Verify all required targets are defined
        required_targets = [
            "form_calculation_per_player",
            "weighted_performance_per_player",
            "batch_650_players",
        ]

        for target in required_targets:
            assert target in benchmarker.targets
            assert benchmarker.targets[target] > 0

        # Verify targets are reasonable (not too strict or too loose)
        assert (
            benchmarker.targets["form_calculation_per_player"] <= 100
        )  # Max 100ms per player
        assert (
            benchmarker.targets["weighted_performance_per_player"] <= 100
        )  # Max 100ms per player
        assert (
            benchmarker.targets["batch_650_players"] <= 5000
        )  # Max 5s for 650 players


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "validator":
            pytest.main(["-v", "TestFeatureValidator"])
        elif test_category == "stats":
            pytest.main(["-v", "TestStatisticalTestSuite"])
        elif test_category == "correlation":
            pytest.main(["-v", "TestCorrelationAnalyzer"])
        elif test_category == "performance":
            pytest.main(["-v", "TestPerformanceBenchmarker"])
        elif test_category == "drift":
            pytest.main(["-v", "TestDataDriftDetector"])
        elif test_category == "integration":
            pytest.main(["-v", "TestIntegration"])
        else:
            print(
                "Available test categories: validator, stats, correlation, performance, drift, integration"
            )
    else:
        # Run all tests
        pytest.main(["-v", __file__])
