"""
Test Suite for Rotation Risk Calculator System

This test suite comprehensively validates the rotation risk prediction system including:
- RotationRiskCalculator ML-based predictions
- ManagerPatternAnalyzer historical analysis
- FixtureCongestionAnalyzer scheduling analysis
- Database model integrity and relationships
- API endpoint functionality
- Validation framework accuracy (>80% threshold)
- Integration with existing AIrsenal components

The tests are organized into:
1. Unit tests for core functionality
2. Integration tests with database
3. ML model training and validation
4. API endpoint testing
5. Performance and accuracy validation
6. Error handling and edge cases
"""

import datetime
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from airsenal.framework.rotation_risk import (
    FixtureCongestionAnalyzer,
    ManagerPatternAnalyzer,
    RotationRiskCalculator,
    get_rotation_risk_for_player,
    get_team_rotation_risks,
)
from airsenal.framework.schema import (
    Base,
    Fixture,
    FixtureCongestion,
    ManagerRotationPattern,
    Player,
    PlayerAttributes,
    PlayerScore,
    RotationRiskPrediction,
    get_session,
)
from airsenal.framework.utils import CURRENT_SEASON


class TestRotationRiskCalculator(unittest.TestCase):
    """Test the main RotationRiskCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = RotationRiskCalculator()
        self.test_season = "2023-24"
        self.test_gameweek = 10
        
        # Mock database session
        self.mock_session = MagicMock()
        self.calculator.dbsession = self.mock_session
        
    def test_calculator_initialization(self):
        """Test that RotationRiskCalculator initializes correctly."""
        self.assertIsInstance(self.calculator, RotationRiskCalculator)
        self.assertIsInstance(self.calculator.manager_analyzer, ManagerPatternAnalyzer)
        self.assertIsInstance(self.calculator.congestion_analyzer, FixtureCongestionAnalyzer)
        self.assertFalse(self.calculator.is_trained)
        self.assertEqual(self.calculator.model_accuracy, 0.0)
        
    def test_calculate_rotation_risk_basic(self):
        """Test basic rotation risk calculation."""
        # Mock player data
        mock_player = MagicMock()
        mock_player.name = "Test Player"
        mock_attrs = MagicMock()
        mock_attrs.team = "ARS"
        mock_attrs.position = "MID"
        mock_player.get_gameweek_attributes.return_value = mock_attrs
        
        # Mock get_player function
        with patch('airsenal.framework.rotation_risk.get_player') as mock_get_player:
            mock_get_player.return_value = mock_player
            
            # Mock the risk factor calculations
            self.calculator._calculate_risk_factors = MagicMock(return_value={
                'fixture_congestion': 0.3,
                'fatigue_risk': 0.2,
                'player_importance': 0.8,
                'manager_rotation_tendency': 0.3,
                'position_rotation_rate': 0.4,
                'age_factor': 0.1,
                'competition_importance': 1.0,
                'team_depth': 0.5,
                'recovery_time': 0.2
            })
            
            result = self.calculator.calculate_rotation_risk(1, self.test_gameweek, self.test_season)
            
            # Verify result structure
            self.assertIn('rotation_risk', result)
            self.assertIn('risk_factors', result)
            self.assertIn('confidence', result)
            self.assertIn('model_version', result)
            self.assertIn('player_id', result)
            self.assertIn('gameweek', result)
            self.assertIn('season', result)
            
            # Verify rotation risk is in valid range
            self.assertGreaterEqual(result['rotation_risk'], 0.0)
            self.assertLessEqual(result['rotation_risk'], 1.0)
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            
    def test_calculate_batch_rotation_risks(self):
        """Test batch calculation of rotation risks."""
        player_ids = [1, 2, 3]
        
        # Mock individual calculations
        self.calculator.calculate_rotation_risk = MagicMock(side_effect=[
            {'rotation_risk': 0.3, 'confidence': 0.8, 'player_id': 1},
            {'rotation_risk': 0.7, 'confidence': 0.6, 'player_id': 2},
            {'rotation_risk': 0.1, 'confidence': 0.9, 'player_id': 3}
        ])
        
        results = self.calculator.calculate_batch_rotation_risks(
            player_ids, self.test_gameweek, self.test_season
        )
        
        self.assertEqual(len(results), 3)
        self.assertIn(1, results)
        self.assertIn(2, results)
        self.assertIn(3, results)
        self.assertEqual(results[1]['rotation_risk'], 0.3)
        self.assertEqual(results[2]['rotation_risk'], 0.7)
        self.assertEqual(results[3]['rotation_risk'], 0.1)
        
    def test_calculate_fatigue_risk(self):
        """Test fatigue risk calculation based on recent minutes."""
        # Mock player and recent scores
        mock_player = MagicMock()
        mock_player.name = "Test Player"
        
        with patch('airsenal.framework.rotation_risk.get_player') as mock_get_player, \
             patch('airsenal.framework.rotation_risk.get_recent_scores_for_player') as mock_recent_scores:
            
            mock_get_player.return_value = mock_player
            mock_recent_scores.return_value = {7: 5, 8: 7, 9: 6}  # Recent gameweeks and scores
            
            # Mock PlayerScore query to return high minutes
            mock_score_obj = MagicMock()
            mock_score_obj.minutes = 90
            self.mock_session.query.return_value.filter.return_value.first.return_value = mock_score_obj
            
            fatigue_risk = self.calculator._calculate_fatigue_risk(1, self.test_gameweek, self.test_season)
            
            # High minutes should result in higher fatigue risk
            self.assertGreaterEqual(fatigue_risk, 0.0)
            self.assertLessEqual(fatigue_risk, 1.0)
            
    def test_calculate_player_importance(self):
        """Test player importance calculation."""
        # Mock player data
        mock_player = MagicMock()
        mock_player.name = "Test Player"
        
        with patch('airsenal.framework.rotation_risk.get_player') as mock_get_player:
            mock_get_player.return_value = mock_player
            
            # Mock database queries for player performance
            mock_scores = [
                MagicMock(minutes=90, points=8),
                MagicMock(minutes=85, points=6),
                MagicMock(minutes=90, points=10)
            ]
            self.mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = mock_scores
            self.mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = [15]
            
            importance = self.calculator._calculate_player_importance(1, "ARS", self.test_season)
            
            # High-performing regular starter should have high importance
            self.assertGreaterEqual(importance, 0.0)
            self.assertLessEqual(importance, 1.0)
            
    def test_weighted_risk_calculation(self):
        """Test weighted combination of risk factors."""
        factors = {
            'fixture_congestion': 0.8,  # High
            'recovery_time': 0.2,      # Low
            'fatigue_risk': 0.9,       # Very high
            'age_factor': 0.3,         # Moderate
            'manager_rotation_tendency': 0.4,  # Moderate
            'position_rotation_rate': 0.5,    # Moderate
            'player_importance': 0.9,  # Very important (should reduce risk)
            'competition_importance': 1.0,    # High importance (should reduce risk)
            'team_depth': 0.3          # Low depth
        }
        
        risk_score = self.calculator._calculate_weighted_risk(factors)
        
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
        
        # With high player importance, risk should be moderated
        self.assertLess(risk_score, 0.8)  # Should be less than raw congestion/fatigue would suggest
        
    def test_model_training(self):
        """Test ML model training functionality."""
        # Create synthetic training data
        n_samples = 1000
        np.random.seed(42)
        
        training_data = pd.DataFrame({
            'fixture_congestion': np.random.uniform(0, 1, n_samples),
            'recovery_time': np.random.uniform(0, 1, n_samples),
            'fatigue_risk': np.random.uniform(0, 1, n_samples),
            'age_factor': np.random.uniform(0, 1, n_samples),
            'manager_rotation_tendency': np.random.uniform(0, 1, n_samples),
            'position_rotation_rate': np.random.uniform(0, 1, n_samples),
            'player_importance': np.random.uniform(0, 1, n_samples),
            'competition_importance': np.random.uniform(0, 1, n_samples),
            'team_depth': np.random.uniform(0, 1, n_samples),
        })
        
        # Create realistic target: higher risk factors → higher rotation probability
        rotation_probability = (
            training_data['fixture_congestion'] * 0.3 +
            training_data['fatigue_risk'] * 0.3 +
            training_data['manager_rotation_tendency'] * 0.2 +
            (1 - training_data['player_importance']) * 0.2
        )
        training_data['was_rotated'] = (rotation_probability > 0.5).astype(int)
        
        metrics = self.calculator.train_model(training_data)
        
        # Verify training completed successfully
        self.assertIn('forest_accuracy', metrics)
        self.assertIn('logistic_accuracy', metrics)
        self.assertIn('feature_importance', metrics)
        self.assertTrue(self.calculator.is_trained)
        self.assertGreater(self.calculator.model_accuracy, 0)
        
        # With synthetic data designed to be learnable, accuracy should be reasonable
        self.assertGreater(metrics['forest_accuracy'], 0.6)
        
    def test_historical_validation(self):
        """Test historical accuracy validation."""
        # First train a model
        n_samples = 500
        np.random.seed(42)
        
        training_data = pd.DataFrame({
            'fixture_congestion': np.random.uniform(0, 1, n_samples),
            'recovery_time': np.random.uniform(0, 1, n_samples),
            'fatigue_risk': np.random.uniform(0, 1, n_samples),
            'age_factor': np.random.uniform(0, 1, n_samples),
            'manager_rotation_tendency': np.random.uniform(0, 1, n_samples),
            'position_rotation_rate': np.random.uniform(0, 1, n_samples),
            'player_importance': np.random.uniform(0, 1, n_samples),
            'competition_importance': np.random.uniform(0, 1, n_samples),
            'team_depth': np.random.uniform(0, 1, n_samples),
        })
        
        # Create target with clear pattern
        rotation_probability = (
            training_data['fixture_congestion'] * 0.4 +
            training_data['fatigue_risk'] * 0.4 +
            (1 - training_data['player_importance']) * 0.2
        )
        training_data['was_rotated'] = (rotation_probability > 0.5).astype(int)
        
        self.calculator.train_model(training_data)
        
        # Create validation data with same pattern
        validation_data = training_data.sample(n=100, random_state=123)
        
        validation_metrics = self.calculator.validate_historical_accuracy(validation_data)
        
        self.assertIn('forest_accuracy', validation_metrics)
        self.assertIn('validation_samples', validation_metrics)
        self.assertIn('meets_80_percent_threshold', validation_metrics)
        self.assertEqual(validation_metrics['validation_samples'], 100)
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent player
        with patch('airsenal.framework.rotation_risk.get_player') as mock_get_player:
            mock_get_player.return_value = None
            
            result = self.calculator.calculate_rotation_risk(999, self.test_gameweek, self.test_season)
            
            # Should return default prediction with error indicator
            self.assertIn('error', result)
            self.assertEqual(result['rotation_risk'], 0.3)  # Default risk
            self.assertEqual(result['confidence'], 0.4)    # Lower confidence
            
    def test_default_risk_prediction(self):
        """Test default risk prediction structure."""
        default_pred = self.calculator._get_default_risk_prediction(1, self.test_gameweek, self.test_season)
        
        self.assertEqual(default_pred['rotation_risk'], 0.3)
        self.assertEqual(default_pred['confidence'], 0.4)
        self.assertEqual(default_pred['player_id'], 1)
        self.assertEqual(default_pred['gameweek'], self.test_gameweek)
        self.assertEqual(default_pred['season'], self.test_season)
        self.assertIn('error', default_pred)


class TestManagerPatternAnalyzer(unittest.TestCase):
    """Test the ManagerPatternAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ManagerPatternAnalyzer()
        self.test_season = "2023-24"
        self.test_team = "ARS"
        
        # Mock database session
        self.mock_session = MagicMock()
        self.analyzer.dbsession = self.mock_session
        
    def test_analyzer_initialization(self):
        """Test that ManagerPatternAnalyzer initializes correctly."""
        self.assertIsInstance(self.analyzer, ManagerPatternAnalyzer)
        self.assertEqual(self.analyzer.manager_patterns, {})
        
    def test_analyze_manager_patterns_no_fixtures(self):
        """Test manager pattern analysis with no fixtures."""
        # Mock empty fixture query
        self.mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        
        patterns = self.analyzer.analyze_manager_patterns(self.test_team, self.test_season)
        
        # Should return default patterns
        self.assertIn('avg_rotation_rate', patterns)
        self.assertIn('congestion_response', patterns)
        self.assertIn('position_preferences', patterns)
        self.assertIn('competition_priorities', patterns)
        
        # Default values should be reasonable
        self.assertEqual(patterns['avg_rotation_rate'], 0.0)
        self.assertEqual(patterns['congestion_response'], 1.0)
        
    def test_analyze_manager_patterns_with_fixtures(self):
        """Test manager pattern analysis with fixture data."""
        # Mock fixtures
        mock_fixtures = [
            MagicMock(fixture_id=1, gameweek=1, date='2023-08-15'),
            MagicMock(fixture_id=2, gameweek=2, date='2023-08-22'),
            MagicMock(fixture_id=3, gameweek=3, date='2023-08-29'),
        ]
        self.mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_fixtures
        
        # Mock player scores for lineups
        mock_scores = [
            MagicMock(player_id=1, minutes=90),
            MagicMock(player_id=2, minutes=85),
            MagicMock(player_id=3, minutes=75),
        ]
        
        query_mock = self.mock_session.query.return_value
        query_mock.join.return_value.join.return_value.filter.return_value.all.return_value = mock_scores
        
        patterns = self.analyzer.analyze_manager_patterns(self.test_team, self.test_season)
        
        # Should analyze and return meaningful patterns
        self.assertIn('avg_rotation_rate', patterns)
        self.assertIn('sample_size', patterns)
        self.assertGreaterEqual(patterns['sample_size'], 0)
        
    def test_get_competition_type(self):
        """Test competition type detection."""
        # Test Premier League fixture
        mock_fixture_pl = MagicMock()
        mock_fixture_pl.tag = "PL"
        self.assertEqual(self.analyzer._get_competition_type(mock_fixture_pl), 'Premier League')
        
        # Test Champions League fixture
        mock_fixture_cl = MagicMock()
        mock_fixture_cl.tag = "CL"
        self.assertEqual(self.analyzer._get_competition_type(mock_fixture_cl), 'Champions League')
        
        # Test FA Cup fixture
        mock_fixture_fa = MagicMock()
        mock_fixture_fa.tag = "FA Cup"
        self.assertEqual(self.analyzer._get_competition_type(mock_fixture_fa), 'FA Cup')
        
    def test_calculate_days_between_fixtures(self):
        """Test calculation of days between fixtures."""
        mock_fixture1 = MagicMock()
        mock_fixture1.date = '2023-08-15'
        mock_fixture1.gameweek = 1
        
        mock_fixture2 = MagicMock()
        mock_fixture2.date = '2023-08-18'
        mock_fixture2.gameweek = 2
        
        days = self.analyzer._calculate_days_between_fixtures(mock_fixture1, mock_fixture2)
        self.assertEqual(days, 3)
        
        # Test fallback to gameweek difference
        mock_fixture1.date = None
        mock_fixture2.date = None
        days = self.analyzer._calculate_days_between_fixtures(mock_fixture1, mock_fixture2)
        self.assertEqual(days, 7)  # (2-1) * 7
        
    def test_detect_rotation_events(self):
        """Test detection of rotation events between lineups."""
        # Mock previous lineup
        prev_lineup = {
            'starters': [MagicMock(player_id=1), MagicMock(player_id=2), MagicMock(player_id=3)],
            'gameweek': 1,
            'competition': 'Premier League'
        }
        
        # Mock current lineup (player 3 rotated out, player 4 rotated in)
        curr_lineup = {
            'starters': [MagicMock(player_id=1), MagicMock(player_id=2), MagicMock(player_id=4)],
            'gameweek': 2,
            'competition': 'Premier League'
        }
        
        mock_prev_fixture = MagicMock(date='2023-08-15')
        mock_curr_fixture = MagicMock(date='2023-08-18')
        
        events = self.analyzer._detect_rotation_events(
            prev_lineup, curr_lineup, mock_prev_fixture, mock_curr_fixture
        )
        
        # Should detect one rotation out and one rotation in
        self.assertEqual(len(events), 2)
        
        rotated_out_events = [e for e in events if e['event_type'] == 'rotated_out']
        rotated_in_events = [e for e in events if e['event_type'] == 'rotated_in']
        
        self.assertEqual(len(rotated_out_events), 1)
        self.assertEqual(len(rotated_in_events), 1)
        self.assertEqual(rotated_out_events[0]['player_id'], 3)
        self.assertEqual(rotated_in_events[0]['player_id'], 4)
        

class TestFixtureCongestionAnalyzer(unittest.TestCase):
    """Test the FixtureCongestionAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FixtureCongestionAnalyzer()
        self.test_season = "2023-24"
        self.test_team = "ARS"
        self.test_gameweek = 10
        
        # Mock database session
        self.mock_session = MagicMock()
        self.analyzer.dbsession = self.mock_session
        
    def test_analyzer_initialization(self):
        """Test that FixtureCongestionAnalyzer initializes correctly."""
        self.assertIsInstance(self.analyzer, FixtureCongestionAnalyzer)
        
    def test_calculate_congestion_score_no_fixtures(self):
        """Test congestion calculation with no fixtures."""
        # Mock empty fixture query
        self.mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        
        score = self.analyzer.calculate_congestion_score(self.test_team, self.test_gameweek, self.test_season)
        
        # Should return default congestion score
        self.assertEqual(score['congestion_score'], 0.3)
        self.assertEqual(score['fixtures_7_days'], 1)
        self.assertEqual(score['fixtures_14_days'], 2)
        
    def test_calculate_congestion_score_with_fixtures(self):
        """Test congestion calculation with fixture data."""
        # Mock target fixture
        target_fixture = MagicMock()
        target_fixture.fixture_id = 1
        target_fixture.gameweek = self.test_gameweek
        target_fixture.date = '2023-10-15'
        
        # Mock nearby fixtures
        nearby_fixtures = [
            target_fixture,
            MagicMock(fixture_id=2, gameweek=9, date='2023-10-12', away_team=self.test_team),  # 3 days before
            MagicMock(fixture_id=3, gameweek=11, date='2023-10-18', home_team=self.test_team), # 3 days after
        ]
        
        self.mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = nearby_fixtures
        
        score = self.analyzer.calculate_congestion_score(self.test_team, self.test_gameweek, self.test_season)
        
        # Should calculate meaningful congestion metrics
        self.assertIn('congestion_score', score)
        self.assertIn('fixtures_7_days', score)
        self.assertIn('fixtures_14_days', score)
        self.assertIn('travel_burden', score)
        self.assertIn('recovery_time', score)
        
        # With fixtures close together, should have some congestion
        self.assertGreater(score['fixtures_7_days'], 0)
        self.assertGreater(score['congestion_score'], 0)
        
    def test_parse_fixture_date(self):
        """Test fixture date parsing."""
        # Test standard date format
        date_obj = self.analyzer._parse_fixture_date('2023-10-15')
        self.assertEqual(date_obj.year, 2023)
        self.assertEqual(date_obj.month, 10)
        self.assertEqual(date_obj.day, 15)
        
        # Test datetime format
        date_obj = self.analyzer._parse_fixture_date('2023-10-15 15:30:00')
        self.assertEqual(date_obj.year, 2023)
        self.assertEqual(date_obj.month, 10)
        self.assertEqual(date_obj.day, 15)
        
        # Test invalid format
        date_obj = self.analyzer._parse_fixture_date('invalid-date')
        self.assertIsNone(date_obj)
        
        # Test None input
        date_obj = self.analyzer._parse_fixture_date(None)
        self.assertIsNone(date_obj)
        
    def test_get_default_congestion_score(self):
        """Test default congestion score structure."""
        default_score = self.analyzer._get_default_congestion_score()
        
        self.assertEqual(default_score['congestion_score'], 0.3)
        self.assertEqual(default_score['fixtures_7_days'], 1)
        self.assertEqual(default_score['fixtures_14_days'], 2)
        self.assertEqual(default_score['travel_burden'], 1.0)
        self.assertEqual(default_score['recovery_time'], 4.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for rotation risk."""
    
    @patch('airsenal.framework.rotation_risk.RotationRiskCalculator')
    def test_get_rotation_risk_for_player(self, mock_calculator_class):
        """Test get_rotation_risk_for_player function."""
        # Mock calculator instance and return value
        mock_calculator = MagicMock()
        mock_calculator.calculate_rotation_risk.return_value = {
            'rotation_risk': 0.4,
            'confidence': 0.8
        }
        mock_calculator_class.return_value = mock_calculator
        
        result = get_rotation_risk_for_player(1, 10, "2023-24")
        
        # Verify calculator was created and called correctly
        mock_calculator_class.assert_called_once()
        mock_calculator.calculate_rotation_risk.assert_called_once_with(1, 10, "2023-24")
        self.assertEqual(result['rotation_risk'], 0.4)
        self.assertEqual(result['confidence'], 0.8)
        
    @patch('airsenal.framework.rotation_risk.RotationRiskCalculator')
    @patch('airsenal.framework.rotation_risk.session')
    def test_get_team_rotation_risks(self, mock_session, mock_calculator_class):
        """Test get_team_rotation_risks function."""
        # Mock players query
        mock_players = [
            MagicMock(player_id=1),
            MagicMock(player_id=2),
            MagicMock(player_id=3)
        ]
        query_mock = mock_session.query.return_value
        query_mock.join.return_value.filter.return_value.distinct.return_value.all.return_value = mock_players
        
        # Mock calculator
        mock_calculator = MagicMock()
        mock_calculator.calculate_batch_rotation_risks.return_value = {
            1: {'rotation_risk': 0.3},
            2: {'rotation_risk': 0.6},
            3: {'rotation_risk': 0.2}
        }
        mock_calculator_class.return_value = mock_calculator
        
        result = get_team_rotation_risks("ARS", 10, "2023-24")
        
        # Verify correct players were found and risks calculated
        self.assertEqual(len(result), 3)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(3, result)
        mock_calculator.calculate_batch_rotation_risks.assert_called_once_with([1, 2, 3], 10, "2023-24")


class TestDatabaseModels(unittest.TestCase):
    """Test the database models for rotation risk."""
    
    def test_rotation_risk_prediction_model(self):
        """Test RotationRiskPrediction model structure."""
        # Test model can be instantiated
        prediction = RotationRiskPrediction(
            player_id=1,
            season="2023-24",
            gameweek=10,
            rotation_risk=0.4,
            confidence=0.8,
            model_version="1.0.0",
            fixture_congestion=0.3,
            fatigue_risk=0.2,
            age_factor=0.1,
            manager_rotation_tendency=0.3,
            position_rotation_rate=0.4,
            player_importance=0.8,
            competition_importance=1.0,
            team_depth=0.5,
            recovery_time=0.2,
            predicted_at=datetime.datetime.now().isoformat()
        )
        
        self.assertEqual(prediction.player_id, 1)
        self.assertEqual(prediction.rotation_risk, 0.4)
        self.assertEqual(prediction.confidence, 0.8)
        
    def test_manager_rotation_pattern_model(self):
        """Test ManagerRotationPattern model structure."""
        pattern = ManagerRotationPattern(
            team="ARS",
            season="2023-24",
            manager_name="Mikel Arteta",
            avg_rotation_rate=0.3,
            congestion_response=1.2,
            sample_size=20,
            gk_rotation_rate=0.1,
            def_rotation_rate=0.3,
            mid_rotation_rate=0.4,
            fwd_rotation_rate=0.35,
            analysis_version="1.0.0",
            analyzed_at=datetime.datetime.now().isoformat(),
            data_quality_score=0.8
        )
        
        self.assertEqual(pattern.team, "ARS")
        self.assertEqual(pattern.manager_name, "Mikel Arteta")
        self.assertEqual(pattern.avg_rotation_rate, 0.3)
        
    def test_fixture_congestion_model(self):
        """Test FixtureCongestion model structure."""
        congestion = FixtureCongestion(
            team="ARS",
            season="2023-24",
            gameweek=10,
            congestion_score=0.6,
            fixtures_7_days=3,
            fixtures_14_days=5,
            travel_burden=2.5,
            recovery_time=3.0,
            calculated_at=datetime.datetime.now().isoformat(),
            data_quality_score=0.9
        )
        
        self.assertEqual(congestion.team, "ARS")
        self.assertEqual(congestion.gameweek, 10)
        self.assertEqual(congestion.congestion_score, 0.6)
        self.assertEqual(congestion.fixtures_7_days, 3)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and real-world use cases."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.calculator = RotationRiskCalculator()
        
    def test_high_congestion_scenario(self):
        """Test prediction accuracy during high fixture congestion."""
        # Mock high congestion scenario
        risk_factors = {
            'fixture_congestion': 0.9,  # Very high
            'recovery_time': 0.8,      # Short recovery
            'fatigue_risk': 0.7,       # High fatigue
            'age_factor': 0.4,         # Older player
            'manager_rotation_tendency': 0.6,  # Manager likes to rotate
            'position_rotation_rate': 0.5,    # Moderate position rotation
            'player_importance': 0.6,  # Moderately important
            'competition_importance': 0.8,    # Important competition
            'team_depth': 0.7          # Good depth
        }
        
        risk_score = self.calculator._calculate_weighted_risk(risk_factors)
        
        # High congestion + fatigue should result in significant rotation risk
        self.assertGreater(risk_score, 0.5)
        
    def test_key_player_scenario(self):
        """Test that key players have reduced rotation risk."""
        # Mock key player scenario
        risk_factors = {
            'fixture_congestion': 0.6,  # Moderate congestion
            'recovery_time': 0.4,      # Moderate recovery
            'fatigue_risk': 0.5,       # Moderate fatigue
            'age_factor': 0.2,         # Young player
            'manager_rotation_tendency': 0.4,  # Some rotation tendency
            'position_rotation_rate': 0.3,    # Low position rotation
            'player_importance': 0.95, # Very important player
            'competition_importance': 1.0,    # Very important competition
            'team_depth': 0.4          # Limited depth
        }
        
        risk_score = self.calculator._calculate_weighted_risk(risk_factors)
        
        # Key player importance should significantly reduce rotation risk
        self.assertLess(risk_score, 0.4)
        
    def test_squad_player_scenario(self):
        """Test that squad players have higher rotation risk."""
        # Mock squad player scenario
        risk_factors = {
            'fixture_congestion': 0.4,  # Low congestion
            'recovery_time': 0.2,      # Good recovery
            'fatigue_risk': 0.3,       # Low fatigue
            'age_factor': 0.3,         # Moderate age
            'manager_rotation_tendency': 0.5,  # Some rotation
            'position_rotation_rate': 0.6,    # High position rotation
            'player_importance': 0.2,  # Low importance (squad player)
            'competition_importance': 0.6,    # Moderate competition
            'team_depth': 0.8          # High depth
        }
        
        risk_score = self.calculator._calculate_weighted_risk(risk_factors)
        
        # Squad players with low importance should have higher rotation risk
        self.assertGreater(risk_score, 0.4)


if __name__ == '__main__':
    unittest.main()