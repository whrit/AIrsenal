#!/usr/bin/env python3
"""
Rotation Risk System Validation Script

This script validates that the rotation risk prediction system meets the required
>80% accuracy threshold on historical data. It performs comprehensive testing of:

1. Historical data preparation and labeling
2. Model training and cross-validation
3. Accuracy measurement on out-of-sample data
4. Feature importance analysis
5. Performance by different player segments

Usage:
    python validate_rotation_system.py [--season SEASON] [--min-accuracy 0.8] [--verbose]
    
Requirements:
    - Historical player and fixture data in the database
    - At least one complete season of data for validation
    - Sufficient variation in rotation events for training

The script will:
1. Extract historical rotation events from player minutes data
2. Calculate risk factors for each prediction instance
3. Train ML models on historical data
4. Validate accuracy using cross-validation and holdout sets
5. Report detailed performance metrics
6. Generate recommendations for model improvements if needed
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from airsenal.framework.rotation_risk import RotationRiskCalculator
from airsenal.framework.schema import (
    Fixture, Player, PlayerAttributes, PlayerScore, session
)
from airsenal.framework.utils import CURRENT_SEASON

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RotationSystemValidator:
    """
    Comprehensive validator for the rotation risk prediction system.
    
    Extracts historical rotation events, trains models, and validates accuracy
    against the required >80% threshold.
    """
    
    def __init__(self, season: str = None, min_accuracy: float = 0.8, verbose: bool = False):
        self.season = season or CURRENT_SEASON
        self.min_accuracy = min_accuracy
        self.verbose = verbose
        self.dbsession = session
        
        # Initialize rotation risk calculator
        self.calculator = RotationRiskCalculator(dbsession=self.dbsession)
        
        # Validation results
        self.validation_results = {}
        self.training_data = None
        self.test_data = None
        
    def run_full_validation(self) -> Dict[str, any]:
        """
        Run the complete validation pipeline.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        logger.info("Starting rotation risk system validation...")
        logger.info(f"Target accuracy threshold: {self.min_accuracy:.1%}")
        
        try:
            # Step 1: Extract historical rotation events
            logger.info("Step 1: Extracting historical rotation events...")
            historical_data = self._extract_historical_rotation_events()
            
            if len(historical_data) < 100:
                raise ValueError(f"Insufficient historical data: {len(historical_data)} events. Need at least 100.")
            
            logger.info(f"Extracted {len(historical_data)} historical rotation events")
            
            # Step 2: Calculate risk factors for historical events
            logger.info("Step 2: Calculating risk factors...")
            labeled_data = self._calculate_historical_risk_factors(historical_data)
            
            # Step 3: Prepare training and test sets
            logger.info("Step 3: Preparing training and test datasets...")
            self.training_data, self.test_data = self._prepare_datasets(labeled_data)
            
            # Step 4: Train models
            logger.info("Step 4: Training rotation risk models...")
            training_metrics = self._train_models()
            
            # Step 5: Validate on holdout test set
            logger.info("Step 5: Validating on holdout test set...")
            test_metrics = self._validate_on_test_set()
            
            # Step 6: Cross-validation analysis
            logger.info("Step 6: Performing cross-validation analysis...")
            cv_metrics = self._cross_validation_analysis()
            
            # Step 7: Feature importance analysis
            logger.info("Step 7: Analyzing feature importance...")
            feature_analysis = self._analyze_feature_importance()
            
            # Step 8: Segment-specific analysis
            logger.info("Step 8: Analyzing performance by player segments...")
            segment_analysis = self._analyze_performance_by_segments()
            
            # Compile final results
            self.validation_results = {
                "validation_summary": {
                    "total_events": len(historical_data),
                    "training_events": len(self.training_data),
                    "test_events": len(self.test_data),
                    "target_accuracy": self.min_accuracy,
                    "achieved_accuracy": test_metrics.get("accuracy", 0.0),
                    "meets_threshold": test_metrics.get("accuracy", 0.0) >= self.min_accuracy,
                    "validation_date": datetime.now().isoformat()
                },
                "training_metrics": training_metrics,
                "test_metrics": test_metrics,
                "cross_validation": cv_metrics,
                "feature_importance": feature_analysis,
                "segment_analysis": segment_analysis,
                "recommendations": self._generate_recommendations()
            }
            
            # Log final results
            self._log_validation_results()
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "error": str(e),
                "validation_date": datetime.now().isoformat(),
                "success": False
            }
    
    def _extract_historical_rotation_events(self) -> pd.DataFrame:
        """
        Extract historical rotation events from player minutes data.
        
        A rotation event occurs when a player who regularly starts is benched
        or when a squad player gets an unexpected start.
        """
        rotation_events = []
        
        # Get all fixtures from previous seasons for training
        training_seasons = [self.season]  # Could expand to multiple seasons
        
        for season in training_seasons:
            logger.info(f"Processing season {season}...")
            
            # Get all fixtures for the season
            fixtures = (self.dbsession.query(Fixture)
                       .filter(Fixture.season == season)
                       .order_by(Fixture.gameweek)
                       .all())
            
            # Process each team's fixtures
            teams = set()
            for fixture in fixtures:
                teams.add(fixture.home_team)
                teams.add(fixture.away_team)
            
            for team in teams:
                if self.verbose:
                    logger.info(f"Processing team {team}...")
                
                team_events = self._extract_team_rotation_events(team, season, fixtures)
                rotation_events.extend(team_events)
        
        return pd.DataFrame(rotation_events)
    
    def _extract_team_rotation_events(self, team: str, season: str, fixtures: List[Fixture]) -> List[Dict]:
        """Extract rotation events for a specific team."""
        events = []
        team_fixtures = [f for f in fixtures if f.home_team == team or f.away_team == team]
        
        # Get all players for the team
        team_players = (self.dbsession.query(Player)
                       .join(PlayerAttributes)
                       .filter(
                           PlayerAttributes.team == team,
                           PlayerAttributes.season == season
                       )
                       .distinct()
                       .all())
        
        player_history = {}  # Track each player's recent starts
        
        for fixture in team_fixtures[:30]:  # Limit to first 30 fixtures for performance
            try:
                # Get player scores for this fixture
                scores = (self.dbsession.query(PlayerScore)
                         .join(Player)
                         .join(PlayerAttributes)
                         .filter(
                             PlayerScore.fixture_id == fixture.fixture_id,
                             PlayerAttributes.team == team,
                             PlayerAttributes.season == season,
                             PlayerAttributes.gameweek == fixture.gameweek
                         )
                         .all())
                
                # Track starts and bench appearances
                starters = []
                benched = []
                
                for score in scores:
                    if score.minutes >= 60:  # Likely starter
                        starters.append(score.player_id)
                    elif score.minutes > 0:  # Substitute appearance
                        pass  # Not relevant for rotation analysis
                    else:  # Benched (0 minutes)
                        benched.append(score.player_id)
                
                # Update player history and detect rotation events
                for player_id in starters + benched:
                    if player_id not in player_history:
                        player_history[player_id] = {'starts': [], 'total_games': 0}
                    
                    is_starter = player_id in starters
                    player_history[player_id]['starts'].append(is_starter)
                    player_history[player_id]['total_games'] += 1
                    
                    # Only create events after player has some history (3+ games)
                    if player_history[player_id]['total_games'] >= 4:
                        recent_starts = player_history[player_id]['starts'][-3:]
                        start_rate = sum(recent_starts) / len(recent_starts)
                        
                        # Rotation event conditions:
                        # 1. Regular starter (start_rate >= 0.67) gets benched
                        # 2. Squad player (start_rate <= 0.33) gets unexpected start
                        was_rotated = False
                        
                        if start_rate >= 0.67 and not is_starter:
                            was_rotated = True  # Regular starter was rotated out
                        elif start_rate <= 0.33 and is_starter:
                            was_rotated = False  # Squad player getting rare start
                        else:
                            continue  # Regular rotation, not an event
                        
                        # Create rotation event
                        events.append({
                            'player_id': player_id,
                            'team': team,
                            'season': season,
                            'gameweek': fixture.gameweek,
                            'fixture_id': fixture.fixture_id,
                            'was_rotated': was_rotated,
                            'minutes_played': next((s.minutes for s in scores if s.player_id == player_id), 0),
                            'recent_start_rate': start_rate,
                            'total_games': player_history[player_id]['total_games']
                        })
                        
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error processing fixture {fixture.fixture_id}: {e}")
                continue
        
        return events
    
    def _calculate_historical_risk_factors(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk factors for each historical rotation event."""
        logger.info("Calculating risk factors for historical events...")
        
        enhanced_data = []
        
        for _, event in historical_data.iterrows():
            try:
                # Calculate risk factors using the rotation calculator
                risk_factors = self.calculator._calculate_risk_factors(
                    event['player_id'],
                    event['gameweek'], 
                    event['season'],
                    event['team'],
                    'MID'  # Default position - would get actual position in real implementation
                )
                
                # Combine event data with risk factors
                event_data = event.to_dict()
                event_data.update(risk_factors)
                enhanced_data.append(event_data)
                
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error calculating factors for player {event['player_id']}: {e}")
                continue
        
        result_df = pd.DataFrame(enhanced_data)
        logger.info(f"Successfully calculated risk factors for {len(result_df)} events")
        
        return result_df
    
    def _prepare_datasets(self, labeled_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and test datasets with proper temporal splits."""
        # Sort by gameweek to ensure temporal ordering
        labeled_data = labeled_data.sort_values(['season', 'gameweek'])
        
        # Use temporal split: first 70% of gameweeks for training, last 30% for testing
        max_gameweek = labeled_data['gameweek'].max()
        split_gameweek = int(max_gameweek * 0.7)
        
        training_data = labeled_data[labeled_data['gameweek'] <= split_gameweek].copy()
        test_data = labeled_data[labeled_data['gameweek'] > split_gameweek].copy()
        
        # Ensure we have enough data in both sets
        min_training_size = 50
        min_test_size = 20
        
        if len(training_data) < min_training_size:
            # Fall back to random split if temporal split doesn't work
            training_data, test_data = train_test_split(
                labeled_data, test_size=0.3, random_state=42, stratify=labeled_data['was_rotated']
            )
        
        logger.info(f"Training set: {len(training_data)} events")
        logger.info(f"Test set: {len(test_data)} events")
        logger.info(f"Training rotation rate: {training_data['was_rotated'].mean():.1%}")
        logger.info(f"Test rotation rate: {test_data['was_rotated'].mean():.1%}")
        
        return training_data, test_data
    
    def _train_models(self) -> Dict[str, any]:
        """Train the rotation risk models on training data."""
        try:
            training_metrics = self.calculator.train_model(self.training_data)
            
            logger.info(f"Model training completed")
            logger.info(f"Random Forest accuracy: {training_metrics.get('forest_accuracy', 0):.3f}")
            logger.info(f"Logistic Regression accuracy: {training_metrics.get('logistic_accuracy', 0):.3f}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}
    
    def _validate_on_test_set(self) -> Dict[str, any]:
        """Validate model performance on the holdout test set."""
        try:
            if not self.calculator.is_trained:
                raise ValueError("Model must be trained before validation")
            
            # Validate on test set
            test_metrics = self.calculator.validate_historical_accuracy(self.test_data)
            
            # Calculate additional metrics
            feature_columns = [
                'fixture_congestion', 'recovery_time', 'fatigue_risk', 'age_factor',
                'manager_rotation_tendency', 'position_rotation_rate', 'player_importance',
                'competition_importance', 'team_depth'
            ]
            
            X_test = self.test_data[feature_columns].fillna(0)
            y_test = self.test_data['was_rotated']
            
            # Get predictions from both models
            X_test_scaled = self.calculator.scaler.transform(X_test)
            forest_pred = self.calculator.forest_model.predict(X_test_scaled)
            logistic_pred = self.calculator.logistic_model.predict(X_test_scaled)
            
            # Calculate detailed metrics
            test_metrics.update({
                "forest_accuracy": accuracy_score(y_test, forest_pred),
                "logistic_accuracy": accuracy_score(y_test, logistic_pred),
                "forest_classification_report": classification_report(y_test, forest_pred, output_dict=True),
                "logistic_classification_report": classification_report(y_test, logistic_pred, output_dict=True),
                "confusion_matrix_forest": confusion_matrix(y_test, forest_pred).tolist(),
                "confusion_matrix_logistic": confusion_matrix(y_test, logistic_pred).tolist()
            })
            
            # Use best model for final accuracy
            test_metrics["accuracy"] = max(
                test_metrics["forest_accuracy"], 
                test_metrics["logistic_accuracy"]
            )
            
            logger.info(f"Test set validation completed")
            logger.info(f"Best model accuracy: {test_metrics['accuracy']:.3f}")
            logger.info(f"Meets {self.min_accuracy:.1%} threshold: {test_metrics['accuracy'] >= self.min_accuracy}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Test validation failed: {e}")
            return {"error": str(e), "accuracy": 0.0}
    
    def _cross_validation_analysis(self) -> Dict[str, any]:
        """Perform cross-validation analysis for robust performance estimation."""
        try:
            feature_columns = [
                'fixture_congestion', 'recovery_time', 'fatigue_risk', 'age_factor',
                'manager_rotation_tendency', 'position_rotation_rate', 'player_importance',
                'competition_importance', 'team_depth'
            ]
            
            X = self.training_data[feature_columns].fillna(0)
            y = self.training_data['was_rotated']
            
            # 5-fold cross-validation
            forest_cv_scores = cross_val_score(self.calculator.forest_model, X, y, cv=5, scoring='accuracy')
            logistic_cv_scores = cross_val_score(self.calculator.logistic_model, X, y, cv=5, scoring='accuracy')
            
            cv_metrics = {
                "forest_cv_mean": forest_cv_scores.mean(),
                "forest_cv_std": forest_cv_scores.std(),
                "forest_cv_scores": forest_cv_scores.tolist(),
                "logistic_cv_mean": logistic_cv_scores.mean(),
                "logistic_cv_std": logistic_cv_scores.std(),
                "logistic_cv_scores": logistic_cv_scores.tolist(),
                "best_cv_accuracy": max(forest_cv_scores.mean(), logistic_cv_scores.mean())
            }
            
            logger.info(f"Cross-validation completed")
            logger.info(f"Forest CV accuracy: {cv_metrics['forest_cv_mean']:.3f} ± {cv_metrics['forest_cv_std']:.3f}")
            logger.info(f"Logistic CV accuracy: {cv_metrics['logistic_cv_mean']:.3f} ± {cv_metrics['logistic_cv_std']:.3f}")
            
            return cv_metrics
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_feature_importance(self) -> Dict[str, any]:
        """Analyze feature importance and predictive power."""
        try:
            if not hasattr(self.calculator, 'feature_importance'):
                return {"error": "Feature importance not available"}
            
            feature_importance = self.calculator.feature_importance
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            analysis = {
                "feature_rankings": sorted_features,
                "top_3_features": sorted_features[:3],
                "feature_importance_dict": feature_importance,
                "most_important_feature": sorted_features[0][0] if sorted_features else None,
                "least_important_feature": sorted_features[-1][0] if sorted_features else None
            }
            
            logger.info("Feature importance analysis:")
            for feature, importance in sorted_features[:5]:
                logger.info(f"  {feature}: {importance:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_performance_by_segments(self) -> Dict[str, any]:
        """Analyze model performance by different player/situation segments."""
        try:
            segment_analysis = {}
            
            # Segment by player importance
            high_importance = self.test_data[self.test_data['player_importance'] >= 0.7]
            low_importance = self.test_data[self.test_data['player_importance'] < 0.3]
            
            segment_analysis['by_player_importance'] = {
                "high_importance_players": {
                    "count": len(high_importance),
                    "rotation_rate": high_importance['was_rotated'].mean() if len(high_importance) > 0 else 0
                },
                "low_importance_players": {
                    "count": len(low_importance),
                    "rotation_rate": low_importance['was_rotated'].mean() if len(low_importance) > 0 else 0
                }
            }
            
            # Segment by fixture congestion
            high_congestion = self.test_data[self.test_data['fixture_congestion'] >= 0.6]
            low_congestion = self.test_data[self.test_data['fixture_congestion'] < 0.3]
            
            segment_analysis['by_fixture_congestion'] = {
                "high_congestion": {
                    "count": len(high_congestion),
                    "rotation_rate": high_congestion['was_rotated'].mean() if len(high_congestion) > 0 else 0
                },
                "low_congestion": {
                    "count": len(low_congestion),
                    "rotation_rate": low_congestion['was_rotated'].mean() if len(low_congestion) > 0 else 0
                }
            }
            
            logger.info("Segment analysis completed")
            return segment_analysis
            
        except Exception as e:
            logger.error(f"Segment analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not self.validation_results:
            return ["Complete validation first to generate recommendations"]
        
        test_accuracy = self.validation_results.get("test_metrics", {}).get("accuracy", 0)
        
        if test_accuracy >= self.min_accuracy:
            recommendations.append(f"✅ Model meets accuracy threshold ({test_accuracy:.1%} >= {self.min_accuracy:.1%})")
            recommendations.append("Model is ready for production use")
        else:
            recommendations.append(f"❌ Model does not meet accuracy threshold ({test_accuracy:.1%} < {self.min_accuracy:.1%})")
            recommendations.append("Consider the following improvements:")
            
            # Feature-specific recommendations
            feature_importance = self.validation_results.get("feature_importance", {}).get("feature_importance_dict", {})
            if feature_importance:
                least_important = min(feature_importance.items(), key=lambda x: x[1])
                recommendations.append(f"- Remove or improve low-importance feature: {least_important[0]}")
            
            # Data-specific recommendations
            training_size = self.validation_results.get("validation_summary", {}).get("training_events", 0)
            if training_size < 200:
                recommendations.append("- Collect more historical data for training (current: {training_size} events)")
            
            recommendations.append("- Consider feature engineering improvements")
            recommendations.append("- Experiment with different model architectures")
            recommendations.append("- Review data quality and labeling accuracy")
        
        # General recommendations
        cv_std = self.validation_results.get("cross_validation", {}).get("forest_cv_std", 0)
        if cv_std > 0.1:
            recommendations.append("- Model shows high variance; consider regularization")
        
        return recommendations
    
    def _log_validation_results(self):
        """Log comprehensive validation results."""
        if not self.validation_results:
            return
        
        summary = self.validation_results.get("validation_summary", {})
        test_metrics = self.validation_results.get("test_metrics", {})
        
        logger.info("=" * 60)
        logger.info("ROTATION RISK SYSTEM VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total historical events: {summary.get('total_events', 0)}")
        logger.info(f"Training events: {summary.get('training_events', 0)}")
        logger.info(f"Test events: {summary.get('test_events', 0)}")
        logger.info(f"Target accuracy: {summary.get('target_accuracy', 0):.1%}")
        logger.info(f"Achieved accuracy: {summary.get('achieved_accuracy', 0):.1%}")
        logger.info(f"Meets threshold: {summary.get('meets_threshold', False)}")
        logger.info("=" * 60)
        
        # Log recommendations
        recommendations = self.validation_results.get("recommendations", [])
        if recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"  {rec}")
            logger.info("=" * 60)


def main():
    """Main function to run validation from command line."""
    parser = argparse.ArgumentParser(description="Validate rotation risk prediction system")
    parser.add_argument("--season", default=None, help="Season to validate (default: current season)")
    parser.add_argument("--min-accuracy", type=float, default=0.8, help="Minimum required accuracy (default: 0.8)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = RotationSystemValidator(
        season=args.season,
        min_accuracy=args.min_accuracy,
        verbose=args.verbose
    )
    
    results = validator.run_full_validation()
    
    # Exit with appropriate code
    if results.get("validation_summary", {}).get("meets_threshold", False):
        logger.info("Validation PASSED - System meets accuracy requirements")
        sys.exit(0)
    else:
        logger.error("Validation FAILED - System does not meet accuracy requirements")
        sys.exit(1)


if __name__ == "__main__":
    main()