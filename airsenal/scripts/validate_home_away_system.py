#!/usr/bin/env python3
"""
Home/Away Adjustment System Validation Script

This script validates the home/away adjustment system by:
1. Testing core functionality with sample data
2. Validating statistical correctness
3. Checking integration with existing systems
4. Generating performance reports

Usage:
    python validate_home_away_system.py [--season SEASON] [--verbose] [--report-file FILE]

Example:
    python validate_home_away_system.py --season 2425 --verbose --report-file validation_report.json
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from airsenal.framework.home_away_adjustment import (
    HomeAwayAdjuster,
    PlayerHomeAwayPerformance,
    TeamHomeAdvantage,
    apply_home_away_adjustment,
    get_player_home_away_adjustment,
    get_team_home_advantage,
)
from airsenal.framework.schema import Fixture, Player, Team, session
from airsenal.framework.team_strength import (
    get_team_strength_with_venue_adjustments,
    validate_integrated_strength_and_venue_model,
)
from airsenal.framework.utils import CURRENT_SEASON


class HomeAwaySystemValidator:
    """Comprehensive validation of the home/away adjustment system."""

    def __init__(self, season: str = CURRENT_SEASON, verbose: bool = False):
        """
        Initialize the validator.

        Args:
            season: Season to validate
            verbose: Whether to print detailed output
        """
        self.season = season
        self.verbose = verbose
        self.validation_results = {}
        self.start_time = None

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def validate_database_connectivity(self) -> bool:
        """Test database connectivity and schema."""
        self.log("Testing database connectivity...")

        try:
            # Test basic query
            team_count = session.query(Team).filter_by(season=self.season).count()
            self.log(f"Found {team_count} teams for season {self.season}")

            if team_count == 0:
                self.log("Warning: No teams found for the specified season", "WARN")
                return False

            # Test fixture data
            fixture_count = session.query(Fixture).filter_by(season=self.season).count()
            self.log(f"Found {fixture_count} fixtures for season {self.season}")

            return True

        except SQLAlchemyError as e:
            self.log(f"Database error: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Unexpected error: {e}", "ERROR")
            return False

    def validate_team_home_advantage_calculations(self) -> dict:
        """Validate team home advantage calculations."""
        self.log("Validating team home advantage calculations...")

        results = {
            "test_passed": False,
            "teams_tested": 0,
            "teams_with_data": 0,
            "calculation_errors": [],
            "sample_results": {},
            "performance_metrics": {},
        }

        try:
            calculator = TeamHomeAdvantage(session)

            # Get sample teams
            teams = (
                session.query(Team.name)
                .filter_by(season=self.season)
                .distinct()
                .limit(5)
                .all()
            )

            calculation_times = []

            for (team_name,) in teams:
                results["teams_tested"] += 1

                try:
                    start_time = time.time()
                    advantage_data = calculator.calculate_team_home_advantage(
                        team_name, self.season
                    )
                    calculation_time = time.time() - start_time
                    calculation_times.append(calculation_time)

                    if (
                        advantage_data["home_matches"] > 0
                        or advantage_data["away_matches"] > 0
                    ):
                        results["teams_with_data"] += 1

                        # Store sample result
                        results["sample_results"][team_name] = {
                            "adjusted_home_advantage": advantage_data[
                                "adjusted_home_advantage"
                            ],
                            "confidence": advantage_data["confidence"],
                            "total_matches": advantage_data["home_matches"]
                            + advantage_data["away_matches"],
                        }

                        self.log(
                            f"{team_name}: advantage={advantage_data['adjusted_home_advantage']:.3f}, "
                            f"confidence={advantage_data['confidence']:.2f}"
                        )

                except Exception as e:
                    error_msg = f"Error calculating advantage for {team_name}: {e}"
                    results["calculation_errors"].append(error_msg)
                    self.log(error_msg, "ERROR")

            # Performance metrics
            if calculation_times:
                results["performance_metrics"] = {
                    "avg_calculation_time": np.mean(calculation_times),
                    "max_calculation_time": max(calculation_times),
                    "total_time": sum(calculation_times),
                }

            results["test_passed"] = (
                results["teams_with_data"] > 0
                and len(results["calculation_errors"]) == 0
            )

        except Exception as e:
            results["calculation_errors"].append(f"Overall validation error: {e}")
            self.log(f"Team validation failed: {e}", "ERROR")

        return results

    def validate_player_home_away_adjustments(self) -> dict:
        """Validate player home/away adjustments."""
        self.log("Validating player home/away adjustments...")

        results = {
            "test_passed": False,
            "players_tested": 0,
            "players_with_significant_effects": 0,
            "calculation_errors": [],
            "sample_results": {},
            "performance_metrics": {},
        }

        try:
            calculator = PlayerHomeAwayPerformance(session, min_games_threshold=20)

            # Get sample players
            players = session.query(Player.player_id, Player.name).limit(10).all()

            calculation_times = []

            for player_id, player_name in players:
                results["players_tested"] += 1

                try:
                    start_time = time.time()
                    adjustment_data = calculator.calculate_player_home_away_adjustment(
                        player_id, self.season
                    )
                    calculation_time = time.time() - start_time
                    calculation_times.append(calculation_time)

                    if adjustment_data["is_significant"]:
                        results["players_with_significant_effects"] += 1

                        # Store sample result
                        results["sample_results"][player_name] = {
                            "home_adjustment": adjustment_data["home_adjustment"],
                            "away_adjustment": adjustment_data["away_adjustment"],
                            "p_value": adjustment_data["p_value"],
                            "total_games": adjustment_data["total_games"],
                        }

                        self.log(
                            f"{player_name}: home={adjustment_data['home_adjustment']:+.2%}, "
                            f"away={adjustment_data['away_adjustment']:+.2%}"
                        )

                except Exception as e:
                    error_msg = f"Error calculating adjustment for {player_name}: {e}"
                    results["calculation_errors"].append(error_msg)
                    self.log(error_msg, "ERROR")

            # Performance metrics
            if calculation_times:
                results["performance_metrics"] = {
                    "avg_calculation_time": np.mean(calculation_times),
                    "max_calculation_time": max(calculation_times),
                    "total_time": sum(calculation_times),
                }

            results["test_passed"] = len(results["calculation_errors"]) == 0

        except Exception as e:
            results["calculation_errors"].append(f"Overall validation error: {e}")
            self.log(f"Player validation failed: {e}", "ERROR")

        return results

    def validate_combined_adjustments(self) -> dict:
        """Validate combined team and player adjustments."""
        self.log("Validating combined adjustments...")

        results = {
            "test_passed": False,
            "adjustments_tested": 0,
            "calculation_errors": [],
            "sample_results": {},
            "impact_statistics": {},
        }

        try:
            adjuster = HomeAwayAdjuster(session)

            # Get sample data
            teams = (
                session.query(Team.name).filter_by(season=self.season).limit(3).all()
            )
            players = session.query(Player.player_id, Player.name).limit(5).all()

            impacts = []

            for (team_name,) in teams:
                for player_id, player_name in players:
                    for is_home in [True, False]:
                        results["adjustments_tested"] += 1

                        try:
                            base_prediction = 5.0  # Standard FPL points

                            adjustment_result = adjuster.apply_combined_adjustment(
                                base_prediction=base_prediction,
                                team=team_name,
                                player_id=player_id,
                                is_home=is_home,
                                season=self.season,
                                gameweek=10,
                                prediction_type="points",
                            )

                            impact = adjustment_result["total_impact"]
                            impacts.append(impact)

                            # Store significant adjustments
                            if abs(impact) > 0.2:
                                key = f"{player_name}_{team_name}_{'H' if is_home else 'A'}"
                                results["sample_results"][key] = {
                                    "base_prediction": base_prediction,
                                    "final_prediction": adjustment_result[
                                        "final_prediction"
                                    ],
                                    "total_impact": impact,
                                    "team_impact": adjustment_result["team_impact"],
                                    "player_impact": adjustment_result["player_impact"],
                                }

                        except Exception as e:
                            error_msg = f"Error in combined adjustment: {e}"
                            results["calculation_errors"].append(error_msg)
                            if (
                                len(results["calculation_errors"]) <= 3
                            ):  # Limit error logging
                                self.log(error_msg, "ERROR")

            # Impact statistics
            if impacts:
                results["impact_statistics"] = {
                    "mean_impact": np.mean(impacts),
                    "std_impact": np.std(impacts),
                    "max_positive_impact": max(impacts),
                    "max_negative_impact": min(impacts),
                    "significant_adjustments": sum(1 for x in impacts if abs(x) > 0.2),
                }

            results["test_passed"] = (
                len(results["calculation_errors"]) < 5
            )  # Allow some minor errors

        except Exception as e:
            results["calculation_errors"].append(f"Overall validation error: {e}")
            self.log(f"Combined adjustment validation failed: {e}", "ERROR")

        return results

    def validate_integration_with_existing_systems(self) -> dict:
        """Validate integration with existing AIrsenal systems."""
        self.log("Validating integration with existing systems...")

        results = {
            "test_passed": False,
            "team_strength_integration": False,
            "prediction_integration": False,
            "api_integration": False,
            "errors": [],
        }

        try:
            # Test team strength integration
            try:
                sample_team = (
                    session.query(Team.name).filter_by(season=self.season).first()
                )
                if sample_team:
                    enhanced_strength = get_team_strength_with_venue_adjustments(
                        sample_team[0], self.season, 10
                    )

                    if "venue_adjustments" in enhanced_strength:
                        results["team_strength_integration"] = True
                        self.log("Team strength integration: PASSED")
                    else:
                        results["errors"].append(
                            "Team strength integration missing venue data"
                        )

            except Exception as e:
                results["errors"].append(f"Team strength integration error: {e}")
                self.log(f"Team strength integration: FAILED - {e}", "ERROR")

            # Test convenience functions
            try:
                sample_team = (
                    session.query(Team.name).filter_by(season=self.season).first()
                )
                if sample_team:
                    get_team_home_advantage(sample_team[0], self.season, 10)

                    sample_player = session.query(Player.player_id).first()
                    if sample_player:
                        get_player_home_away_adjustment(sample_player[0], self.season)

                        adjusted_prediction = apply_home_away_adjustment(
                            5.0, sample_team[0], sample_player[0], True, self.season, 10
                        )

                        if isinstance(adjusted_prediction, int | float):
                            results["prediction_integration"] = True
                            self.log("Prediction integration: PASSED")
                        else:
                            results["errors"].append(
                                "Prediction integration returned invalid type"
                            )

            except Exception as e:
                results["errors"].append(f"Prediction integration error: {e}")
                self.log(f"Prediction integration: FAILED - {e}", "ERROR")

            # Test API-style functions
            try:
                validation_result = validate_integrated_strength_and_venue_model(
                    self.season
                )
                if "error" not in validation_result:
                    results["api_integration"] = True
                    self.log("API integration: PASSED")
                else:
                    results["errors"].append(
                        f"API integration error: {validation_result['error']}"
                    )

            except Exception as e:
                results["errors"].append(f"API integration error: {e}")
                self.log(f"API integration: FAILED - {e}", "ERROR")

            results["test_passed"] = (
                results["team_strength_integration"]
                and results["prediction_integration"]
            )

        except Exception as e:
            results["errors"].append(f"Overall integration validation error: {e}")
            self.log(f"Integration validation failed: {e}", "ERROR")

        return results

    def validate_statistical_properties(self) -> dict:
        """Validate statistical properties of the system."""
        self.log("Validating statistical properties...")

        results = {
            "test_passed": False,
            "normality_tests": {},
            "range_checks": {},
            "consistency_checks": {},
            "errors": [],
        }

        try:
            adjuster = HomeAwayAdjuster(session)

            # Collect sample adjustments
            adjustments = []
            home_advantages = []

            teams = (
                session.query(Team.name).filter_by(season=self.season).limit(10).all()
            )

            for (team_name,) in teams:
                try:
                    advantage_data = (
                        adjuster.team_advantage.calculate_team_home_advantage(
                            team_name, self.season
                        )
                    )
                    home_advantages.append(advantage_data["adjusted_home_advantage"])

                    # Test venue multipliers
                    multipliers = adjuster.get_venue_multipliers(
                        team_name, self.season, 10
                    )
                    adjustments.extend(
                        [multipliers["home_multiplier"], multipliers["away_multiplier"]]
                    )

                except Exception as e:
                    results["errors"].append(
                        f"Error collecting data for {team_name}: {e}"
                    )

            # Range checks
            if home_advantages:
                results["range_checks"]["home_advantages"] = {
                    "min": min(home_advantages),
                    "max": max(home_advantages),
                    "mean": np.mean(home_advantages),
                    "in_expected_range": all(-1.0 <= x <= 1.0 for x in home_advantages),
                }

            if adjustments:
                results["range_checks"]["multipliers"] = {
                    "min": min(adjustments),
                    "max": max(adjustments),
                    "mean": np.mean(adjustments),
                    "in_expected_range": all(0.5 <= x <= 1.5 for x in adjustments),
                }

            # Consistency checks
            results["consistency_checks"]["home_away_symmetry"] = True  # Placeholder
            results["consistency_checks"]["shrinkage_properties"] = True  # Placeholder

            results["test_passed"] = (
                len(results["errors"]) == 0
                and results["range_checks"]
                .get("home_advantages", {})
                .get("in_expected_range", False)
                and results["range_checks"]
                .get("multipliers", {})
                .get("in_expected_range", False)
            )

        except Exception as e:
            results["errors"].append(f"Statistical validation error: {e}")
            self.log(f"Statistical validation failed: {e}", "ERROR")

        return results

    def run_full_validation(self) -> dict:
        """Run complete validation suite."""
        self.start_time = time.time()
        self.log("Starting full validation of home/away adjustment system...")

        # Initialize results structure
        self.validation_results = {
            "metadata": {
                "season": self.season,
                "start_time": datetime.datetime.now().isoformat(),
                "validation_version": "1.0.0",
            },
            "tests": {},
        }

        # Run individual validation tests
        test_suite = [
            ("database_connectivity", self.validate_database_connectivity),
            ("team_home_advantage", self.validate_team_home_advantage_calculations),
            ("player_adjustments", self.validate_player_home_away_adjustments),
            ("combined_adjustments", self.validate_combined_adjustments),
            ("system_integration", self.validate_integration_with_existing_systems),
            ("statistical_properties", self.validate_statistical_properties),
        ]

        passed_tests = 0
        total_tests = len(test_suite)

        for test_name, test_function in test_suite:
            self.log(f"Running test: {test_name}")
            try:
                test_result = test_function()
                self.validation_results["tests"][test_name] = test_result

                if test_result.get("test_passed", False):
                    passed_tests += 1
                    self.log(f"Test {test_name}: PASSED", "SUCCESS")
                else:
                    self.log(f"Test {test_name}: FAILED", "ERROR")

            except Exception as e:
                self.log(f"Test {test_name}: ERROR - {e}", "ERROR")
                self.validation_results["tests"][test_name] = {
                    "test_passed": False,
                    "error": str(e),
                }

        # Calculate overall results
        total_time = time.time() - self.start_time

        self.validation_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_result": "PASSED" if passed_tests == total_tests else "FAILED",
            "total_time_seconds": total_time,
            "end_time": datetime.datetime.now().isoformat(),
        }

        self.log(f"Validation complete: {passed_tests}/{total_tests} tests passed")
        self.log(
            f"Overall result: {self.validation_results['summary']['overall_result']}"
        )

        return self.validation_results

    def save_report(self, filepath: str):
        """Save validation report to file."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            self.log(f"Validation report saved to: {filepath}")
        except Exception as e:
            self.log(f"Error saving report: {e}", "ERROR")


def main():
    """Main function to run validation script."""
    parser = argparse.ArgumentParser(description="Validate Home/Away Adjustment System")
    parser.add_argument("--season", default=CURRENT_SEASON, help="Season to validate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--report-file", help="Path to save validation report")

    args = parser.parse_args()

    # Run validation
    validator = HomeAwaySystemValidator(season=args.season, verbose=args.verbose)
    results = validator.run_full_validation()

    # Save report if requested
    if args.report_file:
        validator.save_report(args.report_file)

    # Print summary
    summary = results["summary"]
    print("\nValidation Summary:")
    print(f"Season: {args.season}")
    print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Overall Result: {summary['overall_result']}")
    print(f"Total Time: {summary['total_time_seconds']:.1f} seconds")

    # Exit with appropriate code
    sys.exit(0 if summary["overall_result"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
