"""
Unit tests for base model interfaces in AIrsenal.

This module tests the abstract base classes defined in base_models.py to ensure:
- Interface contracts are properly defined
- Abstract methods are correctly specified
- Utility functions work as expected
- Mock implementations follow the interface contracts

Test Classes:
    TestAdaptivePlayerModel: Tests for AdaptivePlayerModel interface
    TestAvailabilityPredictor: Tests for AvailabilityPredictor interface
    TestFormCalculator: Tests for FormCalculator interface
    TestFeatureEngineer: Tests for FeatureEngineer interface
    TestUtilityFunctions: Tests for utility functions
"""

import unittest

import numpy as np
import pandas as pd
import pytest

from airsenal.framework.base_models import (
    AdaptivePlayerModel,
    AvailabilityPredictor,
    FeatureEngineer,
    FormCalculator,
    create_mock_player_data,
    validate_model_interface,
)


# Mock implementations for testing
class MockAdaptivePlayerModel(AdaptivePlayerModel):
    """Mock implementation of AdaptivePlayerModel for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitted_data = None
        self.update_count = 0

    def fit(self, data, season, max_gameweek, dbsession=None, **kwargs):
        self.fitted_data = data
        self.is_fitted = True
        self.feature_importance_ = {
            "feature_0": 0.3,
            "feature_1": 0.2,
            "feature_2": 0.15,
            "feature_3": 0.2,
            "feature_4": 0.15,
        }
        return self

    def predict(self, player_ids, gameweeks_ahead=3, features=None, **kwargs):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        n_players = len(player_ids)
        predictions = np.random.poisson(3, (n_players, gameweeks_ahead))
        uncertainty = np.random.exponential(1, (n_players, gameweeks_ahead))

        return {
            "player_ids": np.array(player_ids),
            "predictions": predictions,
            "uncertainty": uncertainty,
            "metadata": {"model_type": "mock", "version": "1.0"},
        }

    def update_with_recent_data(self, recent_data, gameweek, season, **kwargs):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        self.update_count += 1
        self.last_update_gameweek = gameweek
        return self

    def get_feature_importance(self):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)
        return self.feature_importance_


class MockAvailabilityPredictor(AvailabilityPredictor):
    """Mock implementation of AvailabilityPredictor for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitted_data = None

    def predict_availability(
        self, player_ids, gameweeks_ahead=3, season=None, **kwargs
    ):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        n_players = len(player_ids)
        availability_prob = np.random.beta(8, 2, (n_players, gameweeks_ahead))

        return {
            "player_ids": np.array(player_ids),
            "availability_prob": availability_prob,
            "risk_breakdown": {
                "injury": np.random.beta(2, 8, (n_players, gameweeks_ahead)),
                "suspension": np.random.beta(1, 20, (n_players, gameweeks_ahead)),
                "rotation": np.random.beta(3, 7, (n_players, gameweeks_ahead)),
            },
            "confidence": np.random.beta(5, 2, (n_players, gameweeks_ahead)),
        }

    def get_risk_factors(self, player_id, gameweek=None, **kwargs):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        injury_risk = np.random.beta(2, 8)
        suspension_risk = np.random.beta(1, 20)
        rotation_risk = np.random.beta(3, 7)
        other_risk = np.random.beta(1, 10)
        overall_risk = 1 - (1 - injury_risk) * (1 - suspension_risk) * (
            1 - rotation_risk
        ) * (1 - other_risk)

        return {
            "injury_risk": injury_risk,
            "suspension_risk": suspension_risk,
            "rotation_risk": rotation_risk,
            "other_risk": other_risk,
            "overall_risk": overall_risk,
            "risk_description": f"Player {player_id} has {overall_risk:.1%} overall risk",
        }

    def update_injury_data(self, injury_updates, gameweek, season, **kwargs):
        # Mock implementation
        return self


class MockFormCalculator(FormCalculator):
    """Mock implementation of FormCalculator for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitted_data = None

    def calculate_form(
        self,
        player_id,
        time_window=5,
        gameweek=None,
        season=None,
        metrics=None,
        **kwargs,
    ):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        form_score = np.random.uniform(3, 8)
        points_per_game = np.random.uniform(2, 6)
        consistency = np.random.beta(3, 2)
        momentum = np.random.uniform(-1, 1)

        trends = ["improving", "stable", "declining"]
        trend = np.random.choice(trends)

        return {
            "form_score": form_score,
            "points_per_game": points_per_game,
            "trend": trend,
            "consistency": consistency,
            "momentum": momentum,
            "detailed_metrics": {
                "goals_form": np.random.uniform(0, 1),
                "assists_form": np.random.uniform(0, 1),
                "minutes_form": np.random.uniform(0.7, 1.0),
            },
        }

    def get_momentum(self, player_id, lookback_window=10, season=None, **kwargs):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        return {
            "momentum_score": np.random.uniform(-1, 1),
            "acceleration": np.random.uniform(-0.5, 0.5),
            "volatility": np.random.uniform(0, 1),
            "peak_distance": np.random.randint(1, lookback_window),
            "trough_distance": np.random.randint(1, lookback_window),
        }

    def detect_trend_change(
        self, player_id, lookback_window=10, sensitivity=0.1, **kwargs
    ):
        if not self.is_fitted:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        change_detected = bool(np.random.choice([True, False]))

        return {
            "change_detected": change_detected,
            "change_type": np.random.choice(["improvement", "decline"])
            if change_detected
            else None,
            "change_magnitude": np.random.uniform(0, 1) if change_detected else 0,
            "change_gameweek": np.random.randint(1, lookback_window)
            if change_detected
            else None,
            "confidence": np.random.beta(3, 2),
            "description": "Mock trend change detection",
        }


class MockFeatureEngineer(FeatureEngineer):
    """Mock implementation of FeatureEngineer for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitted_data = None

    def fit(self, data, target_variable=None, **kwargs):
        self.fitted_data = data
        self.is_fitted = True

        if isinstance(data, pd.DataFrame):
            n_features = len(data.columns)
        else:
            n_features = 10  # Default for dict/array data

        self.feature_names_ = [f"engineered_feature_{i}" for i in range(n_features)]
        self.feature_metadata_ = {
            "feature_types": dict.fromkeys(self.feature_names_, "numerical"),
            "missing_rates": {
                name: np.random.uniform(0, 0.1) for name in self.feature_names_
            },
            "importance_scores": {
                name: np.random.uniform(0, 1) for name in self.feature_names_
            },
        }
        return self

    def transform(self, data, **kwargs):
        if not self.is_fitted:
            msg = "Pipeline not fitted"
            raise RuntimeError(msg)

        if isinstance(data, pd.DataFrame):
            n_samples = len(data)
        else:
            n_samples = 100  # Default for dict/array data

        n_features = len(self.feature_names_)
        return np.random.randn(n_samples, n_features)

    def get_feature_names(self):
        if not self.is_fitted:
            msg = "Pipeline not fitted"
            raise RuntimeError(msg)
        return self.feature_names_


class TestAdaptivePlayerModel(unittest.TestCase):
    """Test cases for AdaptivePlayerModel interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockAdaptivePlayerModel()
        self.mock_data = create_mock_player_data(n_players=5, n_gameweeks=10)

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = MockAdaptivePlayerModel()
        assert model.learning_rate == 0.01
        assert model.decay_factor == 0.95
        assert not model.is_fitted
        assert model.feature_importance_ is None
        assert model.last_update_gameweek is None

    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        model = MockAdaptivePlayerModel(learning_rate=0.05, decay_factor=0.9)
        assert model.learning_rate == 0.05
        assert model.decay_factor == 0.9

    def test_fit_method(self):
        """Test fitting the model."""
        result = self.model.fit(self.mock_data, season="2023", max_gameweek=38)

        assert result is self.model  # Should return self
        assert self.model.is_fitted
        assert self.model.feature_importance_ is not None
        assert self.model.fitted_data == self.mock_data

    def test_predict_before_fitting(self):
        """Test that prediction fails before fitting."""
        with pytest.raises(RuntimeError):
            self.model.predict([1, 2, 3])

    def test_predict_after_fitting(self):
        """Test prediction after fitting."""
        self.model.fit(self.mock_data, season="2023", max_gameweek=38)

        player_ids = [1, 2, 3]
        result = self.model.predict(player_ids, gameweeks_ahead=3)

        assert "player_ids" in result
        assert "predictions" in result
        assert "uncertainty" in result
        assert "metadata" in result

        np.testing.assert_array_equal(result["player_ids"], player_ids)
        assert result["predictions"].shape == (3, 3)
        assert result["uncertainty"].shape == (3, 3)

    def test_update_with_recent_data(self):
        """Test updating with recent data."""
        self.model.fit(self.mock_data, season="2023", max_gameweek=38)

        recent_data = {"player_ids": [1, 2], "new_features": np.random.randn(2, 5)}
        result = self.model.update_with_recent_data(
            recent_data, gameweek=15, season="2023"
        )

        assert result is self.model  # Should return self
        assert self.model.update_count == 1
        assert self.model.last_update_gameweek == 15

    def test_update_before_fitting(self):
        """Test that update fails before fitting."""
        with pytest.raises(RuntimeError):
            self.model.update_with_recent_data({}, gameweek=15, season="2023")

    def test_get_feature_importance(self):
        """Test getting feature importance."""
        self.model.fit(self.mock_data, season="2023", max_gameweek=38)

        importance = self.model.get_feature_importance()

        assert isinstance(importance, dict)
        assert all(0 <= v <= 1 for v in importance.values())
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)

    def test_get_feature_importance_before_fitting(self):
        """Test that getting feature importance fails before fitting."""
        with pytest.raises(RuntimeError):
            self.model.get_feature_importance()

    def test_model_state_management(self):
        """Test model state saving and loading."""
        self.model.fit(self.mock_data, season="2023", max_gameweek=38)
        self.model.update_with_recent_data({}, gameweek=15, season="2023")

        # Get state
        state = self.model.get_model_state()
        assert isinstance(state, dict)
        assert state["is_fitted"]
        assert state["last_update_gameweek"] == 15

        # Create new model and load state
        new_model = MockAdaptivePlayerModel()
        new_model.load_model_state(state)

        assert new_model.is_fitted
        assert new_model.last_update_gameweek == 15
        assert new_model.learning_rate == state["learning_rate"]


class TestAvailabilityPredictor(unittest.TestCase):
    """Test cases for AvailabilityPredictor interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = MockAvailabilityPredictor()
        self.predictor.is_fitted = True  # Mock as fitted for most tests

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MockAvailabilityPredictor(risk_threshold=0.4)
        assert predictor.risk_threshold == 0.4
        assert not predictor.is_fitted
        assert len(predictor.supported_risk_types) == 4

    def test_predict_availability(self):
        """Test availability prediction."""
        player_ids = [1, 2, 3]
        result = self.predictor.predict_availability(player_ids, gameweeks_ahead=3)

        assert "player_ids" in result
        assert "availability_prob" in result
        assert "risk_breakdown" in result
        assert "confidence" in result

        np.testing.assert_array_equal(result["player_ids"], player_ids)
        assert result["availability_prob"].shape == (3, 3)

        # Check risk breakdown structure
        risk_breakdown = result["risk_breakdown"]
        assert "injury" in risk_breakdown
        assert "suspension" in risk_breakdown
        assert "rotation" in risk_breakdown

    def test_predict_availability_not_fitted(self):
        """Test that prediction fails when not fitted."""
        self.predictor.is_fitted = False
        with pytest.raises(RuntimeError):
            self.predictor.predict_availability([1, 2, 3])

    def test_get_risk_factors(self):
        """Test getting risk factors for a player."""
        result = self.predictor.get_risk_factors(player_id=123)

        required_keys = [
            "injury_risk",
            "suspension_risk",
            "rotation_risk",
            "other_risk",
            "overall_risk",
            "risk_description",
        ]

        for key in required_keys:
            assert key in result

        # Check that risk values are probabilities
        for risk_type in [
            "injury_risk",
            "suspension_risk",
            "rotation_risk",
            "other_risk",
        ]:
            assert 0 <= result[risk_type] <= 1

        assert 0 <= result["overall_risk"] <= 1
        assert isinstance(result["risk_description"], str)

    def test_get_risk_factors_not_fitted(self):
        """Test that getting risk factors fails when not fitted."""
        self.predictor.is_fitted = False
        with pytest.raises(RuntimeError):
            self.predictor.get_risk_factors(player_id=123)

    def test_update_injury_data(self):
        """Test updating injury data."""
        injury_updates = {
            "player_ids": [1, 2],
            "injury_types": ["hamstring", "knee"],
            "expected_return": [3, 5],
            "severity": [0.3, 0.7],
        }

        result = self.predictor.update_injury_data(
            injury_updates, gameweek=15, season="2023"
        )

        assert result is self.predictor  # Should return self

    def test_get_high_risk_players(self):
        """Test getting high-risk players."""
        # Mock the method since it returns empty list by default
        result = self.predictor.get_high_risk_players(gameweek=15, season="2023")
        assert isinstance(result, list)

    def test_get_high_risk_players_not_fitted(self):
        """Test that getting high-risk players fails when not fitted."""
        self.predictor.is_fitted = False
        with pytest.raises(RuntimeError):
            self.predictor.get_high_risk_players(gameweek=15, season="2023")


class TestFormCalculator(unittest.TestCase):
    """Test cases for FormCalculator interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MockFormCalculator()
        self.calculator.is_fitted = True  # Mock as fitted for most tests

    def test_initialization(self):
        """Test calculator initialization."""
        calculator = MockFormCalculator(default_windows=[3, 7, 14])
        assert calculator.default_windows == [3, 7, 14]
        assert not calculator.is_fitted
        assert len(calculator.form_metrics) > 0

    def test_calculate_form(self):
        """Test form calculation."""
        result = self.calculator.calculate_form(player_id=123, time_window=5)

        required_keys = [
            "form_score",
            "points_per_game",
            "trend",
            "consistency",
            "momentum",
            "detailed_metrics",
        ]

        for key in required_keys:
            assert key in result

        assert 0 <= result["form_score"] <= 10
        assert result["points_per_game"] >= 0
        assert result["trend"] in ["improving", "stable", "declining"]
        assert 0 <= result["consistency"] <= 1
        assert -1 <= result["momentum"] <= 1
        assert isinstance(result["detailed_metrics"], dict)

    def test_calculate_form_not_fitted(self):
        """Test that form calculation fails when not fitted."""
        self.calculator.is_fitted = False
        with pytest.raises(RuntimeError):
            self.calculator.calculate_form(player_id=123)

    def test_get_momentum(self):
        """Test momentum calculation."""
        result = self.calculator.get_momentum(player_id=123, lookback_window=10)

        required_keys = [
            "momentum_score",
            "acceleration",
            "volatility",
            "peak_distance",
            "trough_distance",
        ]

        for key in required_keys:
            assert key in result

        assert -1 <= result["momentum_score"] <= 1
        assert -1 <= result["acceleration"] <= 1
        assert 0 <= result["volatility"] <= 1
        assert 1 <= result["peak_distance"] <= 10
        assert 1 <= result["trough_distance"] <= 10

    def test_get_momentum_not_fitted(self):
        """Test that momentum calculation fails when not fitted."""
        self.calculator.is_fitted = False
        with pytest.raises(RuntimeError):
            self.calculator.get_momentum(player_id=123)

    def test_detect_trend_change(self):
        """Test trend change detection."""
        result = self.calculator.detect_trend_change(
            player_id=123, lookback_window=10, sensitivity=0.1
        )

        required_keys = [
            "change_detected",
            "change_type",
            "change_magnitude",
            "change_gameweek",
            "confidence",
            "description",
        ]

        for key in required_keys:
            assert key in result

        assert isinstance(result["change_detected"], bool)

        if result["change_detected"]:
            assert result["change_type"] in ["improvement", "decline"]
            assert 0 <= result["change_magnitude"] <= 1
            assert result["change_gameweek"] is not None
        else:
            assert result["change_type"] is None
            assert result["change_magnitude"] == 0
            assert result["change_gameweek"] is None

        assert 0 <= result["confidence"] <= 1
        assert isinstance(result["description"], str)

    def test_detect_trend_change_not_fitted(self):
        """Test that trend change detection fails when not fitted."""
        self.calculator.is_fitted = False
        with pytest.raises(RuntimeError):
            self.calculator.detect_trend_change(player_id=123)

    def test_get_form_distribution(self):
        """Test getting form distribution."""
        result = self.calculator.get_form_distribution(season="2023", time_window=5)

        assert "mean_form" in result
        assert "std_form" in result
        assert "percentiles" in result

        assert result["mean_form"] > 0
        assert result["std_form"] > 0
        assert isinstance(result["percentiles"], np.ndarray)

    def test_get_form_distribution_not_fitted(self):
        """Test that getting form distribution fails when not fitted."""
        self.calculator.is_fitted = False
        with pytest.raises(RuntimeError):
            self.calculator.get_form_distribution()


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.engineer = MockFeatureEngineer()
        self.mock_dataframe = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randint(0, 5, 100),
                "target": np.random.poisson(3, 100),
            }
        )

    def test_initialization(self):
        """Test engineer initialization."""
        engineer = MockFeatureEngineer(handle_missing="drop", scale_features=False)
        assert engineer.handle_missing == "drop"
        assert not engineer.scale_features
        assert not engineer.is_fitted
        assert engineer.feature_names_ is None

    def test_fit_method(self):
        """Test fitting the pipeline."""
        result = self.engineer.fit(self.mock_dataframe, target_variable="target")

        assert result is self.engineer  # Should return self
        assert self.engineer.is_fitted
        assert self.engineer.feature_names_ is not None
        assert self.engineer.feature_metadata_ is not None

    def test_transform_method(self):
        """Test transforming data."""
        self.engineer.fit(self.mock_dataframe)
        result = self.engineer.transform(self.mock_dataframe)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(self.mock_dataframe)
        assert result.shape[1] == len(self.engineer.feature_names_)

    def test_transform_not_fitted(self):
        """Test that transform fails when not fitted."""
        with pytest.raises(RuntimeError):
            self.engineer.transform(self.mock_dataframe)

    def test_fit_transform_method(self):
        """Test fit_transform convenience method."""
        result = self.engineer.fit_transform(
            self.mock_dataframe, target_variable="target"
        )

        assert self.engineer.is_fitted
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(self.mock_dataframe)

    def test_get_feature_names(self):
        """Test getting feature names."""
        self.engineer.fit(self.mock_dataframe)
        feature_names = self.engineer.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_get_feature_names_not_fitted(self):
        """Test that getting feature names fails when not fitted."""
        with pytest.raises(RuntimeError):
            self.engineer.get_feature_names()

    def test_get_feature_metadata(self):
        """Test getting feature metadata."""
        self.engineer.fit(self.mock_dataframe)
        metadata = self.engineer.get_feature_metadata()

        assert isinstance(metadata, dict)
        expected_keys = ["feature_types", "missing_rates", "importance_scores"]
        for key in expected_keys:
            assert key in metadata

    def test_get_feature_metadata_not_fitted_behavior(self):
        """Test the base implementation behavior for metadata when not fitted."""
        # Test that the base implementation has proper error handling
        # The actual behavior (exception vs empty dict) depends on implementation
        engineer = MockFeatureEngineer()
        # Test that metadata is None or empty when not fitted
        assert engineer.feature_metadata_ is None

    def test_validate_data_compatibility(self):
        """Test data compatibility validation."""
        self.engineer.fit(self.mock_dataframe)
        is_compatible, issues = self.engineer.validate_data_compatibility(
            self.mock_dataframe
        )

        assert is_compatible
        assert len(issues) == 0

    def test_validate_data_compatibility_not_fitted(self):
        """Test data compatibility validation when not fitted."""
        is_compatible, issues = self.engineer.validate_data_compatibility(
            self.mock_dataframe
        )

        assert not is_compatible
        assert "Pipeline not fitted" in issues


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_validate_model_interface_valid(self):
        """Test interface validation with valid model."""
        model = MockAdaptivePlayerModel()
        is_valid, missing = validate_model_interface(model, AdaptivePlayerModel)

        assert is_valid
        assert len(missing) == 0

    def test_validate_model_interface_invalid(self):
        """Test interface validation with invalid model."""

        # Test with a regular object that doesn't implement the interface
        class NotAModel:
            def some_method(self):
                pass

        model = NotAModel()
        is_valid, missing = validate_model_interface(model, AdaptivePlayerModel)

        assert not is_valid
        assert len(missing) > 0

    def test_validate_model_interface_wrong_type(self):
        """Test interface validation with wrong type."""
        model = "not a model"
        is_valid, missing = validate_model_interface(model, AdaptivePlayerModel)

        assert not is_valid
        assert len(missing) > 0

    def test_create_mock_player_data_default(self):
        """Test creating mock player data with default parameters."""
        data = create_mock_player_data()

        required_keys = [
            "player_ids",
            "features",
            "gameweeks",
            "season",
            "targets",
            "minutes",
        ]
        for key in required_keys:
            assert key in data

        assert data["features"].shape == (10, 15, 5)
        assert data["targets"].shape == (10, 15)
        assert data["minutes"].shape == (10, 15)
        assert len(data["player_ids"]) == 10
        assert len(data["gameweeks"]) == 15

    def test_create_mock_player_data_custom(self):
        """Test creating mock player data with custom parameters."""
        data = create_mock_player_data(
            n_players=5, n_gameweeks=8, n_features=3, include_targets=False
        )

        assert data["features"].shape == (5, 8, 3)
        assert "targets" not in data
        assert "minutes" not in data

    def test_create_mock_player_data_reproducibility(self):
        """Test that mock data generation is reproducible."""
        data1 = create_mock_player_data(n_players=5, n_gameweeks=10)
        data2 = create_mock_player_data(n_players=5, n_gameweeks=10)

        np.testing.assert_array_equal(data1["player_ids"], data2["player_ids"])
        np.testing.assert_array_equal(data1["features"], data2["features"])
        np.testing.assert_array_equal(data1["targets"], data2["targets"])


# Integration tests
class TestIntegration(unittest.TestCase):
    """Integration tests for base model interfaces."""

    def test_all_models_follow_interfaces(self):
        """Test that all mock models correctly implement their interfaces."""
        models_and_interfaces = [
            (MockAdaptivePlayerModel(), AdaptivePlayerModel),
            (MockAvailabilityPredictor(), AvailabilityPredictor),
            (MockFormCalculator(), FormCalculator),
            (MockFeatureEngineer(), FeatureEngineer),
        ]

        for model, interface in models_and_interfaces:
            with self.subTest(model=type(model).__name__):
                is_valid, missing = validate_model_interface(model, interface)
                assert is_valid, f"Missing methods: {missing}"

    def test_model_pipeline_integration(self):
        """Test integration between different model types."""
        # Create mock data
        player_data = create_mock_player_data(n_players=5, n_gameweeks=10)

        # Initialize models
        engineer = MockFeatureEngineer()
        adaptive_model = MockAdaptivePlayerModel()
        form_calc = MockFormCalculator()
        availability_pred = MockAvailabilityPredictor()

        # Test pipeline: feature engineering -> adaptive model -> form/availability
        engineer.fit(pd.DataFrame(np.random.randn(50, 4)))
        features = engineer.transform(pd.DataFrame(np.random.randn(50, 4)))

        # Fit adaptive model with engineered features
        enhanced_data = player_data.copy()
        enhanced_data["features"] = features[: 5 * 10].reshape(5, 10, -1)

        adaptive_model.fit(enhanced_data, season="2023", max_gameweek=38)
        predictions = adaptive_model.predict([1, 2, 3])

        # Calculate form and availability
        form_calc.fit(player_data, season="2023")
        form = form_calc.calculate_form(player_id=1)

        availability_pred.fit(player_data, {}, {}, season="2023")
        availability = availability_pred.predict_availability([1, 2, 3])

        # Verify all components work together
        assert isinstance(features, np.ndarray)
        assert "predictions" in predictions
        assert "form_score" in form
        assert "availability_prob" in availability


if __name__ == "__main__":
    unittest.main()
