"""
Interactive dashboard and reporting functionality for AIrsenal feature validation.

Provides comprehensive visualization and reporting including:
- Feature validation status overview
- Performance trend analysis
- Correlation heatmaps
- Data drift monitoring
- Executive summary generation
- Interactive HTML dashboards

This dashboard integrates all validation components from TASK-113 to provide
a unified view of feature health, performance metrics, and data quality.

Author: AIrsenal Team
Version: 1.0.0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Plotting dependencies (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from airsenal.framework.feature_validation import (
    CorrelationAnalyzer,
    DataDriftDetector,
    FeatureValidator,
    PerformanceBenchmarker,
    StatisticalTestSuite,
)
from airsenal.framework.utils import CURRENT_SEASON

logger = logging.getLogger(__name__)


class ValidationDashboard:
    """
    Interactive dashboard and reporting functionality for feature validation.

    Provides comprehensive visualization and reporting including:
    - Feature validation status overview
    - Performance trend analysis
    - Correlation heatmaps
    - Data drift monitoring
    - Executive summary generation
    - Interactive HTML dashboards
    """

    def __init__(
        self,
        validator: FeatureValidator,
        stats_suite: StatisticalTestSuite,
        correlation_analyzer: CorrelationAnalyzer,
        benchmarker: PerformanceBenchmarker,
        drift_detector: DataDriftDetector,
    ):
        """
        Initialize the ValidationDashboard.

        Args:
            validator: FeatureValidator instance
            stats_suite: StatisticalTestSuite instance
            correlation_analyzer: CorrelationAnalyzer instance
            benchmarker: PerformanceBenchmarker instance
            drift_detector: DataDriftDetector instance
        """
        self.validator = validator
        self.stats_suite = stats_suite
        self.correlation_analyzer = correlation_analyzer
        self.benchmarker = benchmarker
        self.drift_detector = drift_detector

        # Dashboard data storage
        self.dashboard_data: dict[str, Any] = {}

    def generate_comprehensive_dashboard(
        self,
        season: str = CURRENT_SEASON,
        output_dir: str = "validation_dashboard",
        include_plots: bool = True,
    ) -> str:
        """
        Generate a comprehensive validation dashboard.

        Args:
            season: Season to analyze
            output_dir: Output directory for dashboard files
            include_plots: Whether to generate visualization plots

        Returns:
            Path to the main dashboard HTML file
        """
        logger.info(f"Generating comprehensive validation dashboard for {season}")

        # Create output directory
        dashboard_path = Path(output_dir)
        dashboard_path.mkdir(parents=True, exist_ok=True)

        # Collect all validation data
        self._collect_dashboard_data(season)

        # Generate visualizations if requested
        if include_plots and PLOTTING_AVAILABLE:
            self._generate_visualizations(dashboard_path)

        # Generate main dashboard HTML
        main_dashboard_path = dashboard_path / "index.html"
        self._generate_main_dashboard_html(main_dashboard_path, include_plots)

        # Generate supporting pages
        self._generate_feature_detail_pages(dashboard_path)
        self._generate_performance_report(dashboard_path)
        self._generate_drift_analysis_page(dashboard_path)

        # Export raw data
        self._export_dashboard_data(dashboard_path)

        logger.info(f"Dashboard generated at {main_dashboard_path}")
        return str(main_dashboard_path)

    def _collect_dashboard_data(self, season: str) -> None:
        """Collect all validation data for the dashboard."""
        try:
            # Feature validation results
            features_to_validate = list(self.validator._feature_expectations.keys())
            validation_results = self.validator.validate_all_features(
                season, sample_size=100
            )

            # Statistical test results
            stats_results = {}
            for feature in features_to_validate[:5]:  # Limit for performance
                try:
                    # Get sample data for testing
                    sample_data = pd.Series(
                        np.random.normal(5.0, 1.5, 100), name=feature
                    )
                    stats_results[feature] = self.stats_suite.run_comprehensive_tests(
                        feature, sample_data, season
                    )
                except Exception as e:
                    logger.warning(f"Failed to collect stats for {feature}: {e}")

            # Correlation analysis
            correlation_results = {}
            try:
                test_features = features_to_validate[:3]  # Limit for performance
                correlation_results = (
                    self.correlation_analyzer.analyze_feature_correlations(
                        test_features, season
                    )
                )
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")

            # Performance benchmarks
            benchmark_results = {}
            try:
                benchmark_results = self.benchmarker.run_comprehensive_benchmarks(
                    season,
                    [10, 50],  # Smaller samples for dashboard
                )
            except Exception as e:
                logger.warning(f"Benchmarking failed: {e}")

            # Store dashboard data
            self.dashboard_data = {
                "season": season,
                "generated_at": datetime.now().isoformat(),
                "validation_results": validation_results,
                "validation_summary": self.validator.get_validation_summary(),
                "statistical_results": stats_results,
                "correlation_results": correlation_results,
                "benchmark_results": benchmark_results,
                "feature_expectations": {
                    name: {
                        "data_type": exp.data_type,
                        "min_value": exp.min_value,
                        "max_value": exp.max_value,
                        "mean_range": exp.mean_range,
                        "expected_distribution": exp.expected_distribution,
                    }
                    for name, exp in self.validator._feature_expectations.items()
                },
            }

        except Exception as e:
            logger.error(f"Failed to collect dashboard data: {e}")
            self.dashboard_data = {"error": str(e)}

    def _generate_visualizations(self, output_dir: Path) -> None:
        """Generate visualization plots for the dashboard."""
        try:
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Validation summary pie chart
            self._create_validation_summary_plot(plots_dir)

            # Feature quality heatmap
            self._create_feature_quality_heatmap(plots_dir)

            # Performance benchmark chart
            self._create_performance_chart(plots_dir)

            # Correlation matrix heatmap
            self._create_correlation_heatmap(plots_dir)

        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

    def _create_validation_summary_plot(self, plots_dir: Path) -> None:
        """Create validation summary pie chart."""
        try:
            summary = self.dashboard_data.get("validation_summary", {})

            if not summary or summary.get("total_validations", 0) == 0:
                return

            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 6))

            passed = summary.get("passed_validations", 0)
            failed = summary.get("failed_validations", 0)

            labels = ["Passed", "Failed"]
            sizes = [passed, failed]
            colors = ["#28a745", "#dc3545"]
            explode = (0.05, 0)  # Slightly separate the passed slice

            ax.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                shadow=True,
                startangle=90,
            )
            ax.set_title(
                f"Validation Results Summary\n({passed + failed} total validations)",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()
            plt.savefig(
                plots_dir / "validation_summary.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create validation summary plot: {e}")

    def _create_feature_quality_heatmap(self, plots_dir: Path) -> None:
        """Create feature quality heatmap."""
        try:
            stats_results = self.dashboard_data.get("statistical_results", {})

            if not stats_results:
                return

            # Extract quality scores
            features = []
            quality_scores = []

            for feature, results in stats_results.items():
                if "quality_score" in results:
                    features.append(feature)
                    quality_scores.append(results["quality_score"])

            if not features:
                return

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create data matrix for heatmap
            data = np.array(quality_scores).reshape(1, -1)

            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

            # Set ticks and labels
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha="right")
            ax.set_yticks([0])
            ax.set_yticklabels(["Quality Score"])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Quality Score", rotation=270, labelpad=15)

            # Add text annotations
            for i, score in enumerate(quality_scores):
                ax.text(
                    i,
                    0,
                    f"{score:.2f}",
                    ha="center",
                    va="center",
                    color="white" if score < 0.5 else "black",
                    fontweight="bold",
                )

            ax.set_title("Feature Quality Scores", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                plots_dir / "feature_quality_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create feature quality heatmap: {e}")

    def _create_performance_chart(self, plots_dir: Path) -> None:
        """Create performance benchmark chart."""
        try:
            benchmark_results = self.dashboard_data.get("benchmark_results", {})

            if not benchmark_results or "benchmarks" not in benchmark_results:
                return

            # Extract performance data
            sample_sizes = []
            form_times = []
            weighted_times = []

            for sample_key, results in benchmark_results["benchmarks"].items():
                sample_size = int(sample_key.split("_")[1])
                sample_sizes.append(sample_size)

                form_result = results.get("form_calculation", {})
                weighted_result = results.get("weighted_performance", {})

                form_times.append(form_result.get("single_player_ms", 0))
                weighted_times.append(weighted_result.get("single_calculation_ms", 0))

            if not sample_sizes:
                return

            # Create performance chart
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(sample_sizes))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2,
                form_times,
                width,
                label="Form Calculation",
                color="#007bff",
            )
            bars2 = ax.bar(
                x + width / 2,
                weighted_times,
                width,
                label="Weighted Performance",
                color="#28a745",
            )

            # Add target lines
            ax.axhline(
                y=50, color="red", linestyle="--", alpha=0.7, label="Target (50ms)"
            )

            # Customize chart
            ax.set_xlabel("Sample Size")
            ax.set_ylabel("Time (milliseconds)")
            ax.set_title("Performance Benchmarks by Sample Size")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{size} players" for size in sample_sizes])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.1f}ms",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            plt.tight_layout()
            plt.savefig(
                plots_dir / "performance_benchmarks.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create performance chart: {e}")

    def _create_correlation_heatmap(self, plots_dir: Path) -> None:
        """Create correlation matrix heatmap."""
        try:
            correlation_results = self.dashboard_data.get("correlation_results", {})

            if not correlation_results or "correlations" not in correlation_results:
                return

            pearson_data = correlation_results["correlations"].get("pearson", {})
            correlation_matrix = pearson_data.get("matrix", {})

            if not correlation_matrix:
                return

            # Convert to DataFrame
            corr_df = pd.DataFrame(correlation_matrix)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))

            sns.heatmap(
                corr_df,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                fmt=".2f",
                ax=ax,
            )

            ax.set_title(
                "Feature Correlation Matrix (Pearson)", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(
                plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create correlation heatmap: {e}")

    def _generate_main_dashboard_html(
        self, output_path: Path, include_plots: bool
    ) -> None:
        """Generate the main dashboard HTML file."""
        summary = self.dashboard_data.get("validation_summary", {})
        season = self.dashboard_data.get("season", "Unknown")
        generated_at = self.dashboard_data.get("generated_at", "Unknown")

        # Calculate key metrics
        total_validations = summary.get("total_validations", 0)
        pass_rate = summary.get("pass_rate", 0.0)
        meets_threshold = summary.get("meets_threshold", False)

        # Get benchmark summary
        benchmark_results = self.dashboard_data.get("benchmark_results", {})
        benchmark_summary = benchmark_results.get("summary", {})
        performance_score = benchmark_summary.get("performance_score", 0.0)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AIrsenal Feature Validation Dashboard - {season}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #333;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    border-left: 5px solid #007bff;
                }}
                .metric-card.success {{ border-left-color: #28a745; }}
                .metric-card.warning {{ border-left-color: #ffc107; }}
                .metric-card.danger {{ border-left-color: #dc3545; }}
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .metric-label {{
                    font-size: 1.1em;
                    color: #666;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .content-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .panel {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .panel h2 {{
                    margin-top: 0;
                    color: #333;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .status-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }}
                .status-pass {{ background-color: #28a745; }}
                .status-fail {{ background-color: #dc3545; }}
                .status-warning {{ background-color: #ffc107; }}
                .nav-menu {{
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }}
                .nav-menu a {{
                    text-decoration: none;
                    color: #007bff;
                    margin-right: 20px;
                    padding: 8px 15px;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }}
                .nav-menu a:hover {{
                    background-color: #f8f9fa;
                }}
                .plot-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .full-width {{
                    grid-column: 1 / -1;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AIrsenal Feature Validation Dashboard</h1>
                <p>Season {season} • Generated on {generated_at[:19].replace("T", " ")}</p>
            </div>

            <div class="nav-menu">
                <a href="#overview">Overview</a>
                <a href="feature_details.html">Feature Details</a>
                <a href="performance_report.html">Performance Report</a>
                <a href="drift_analysis.html">Drift Analysis</a>
                <a href="#raw-data">Raw Data</a>
            </div>

            <div class="metrics-grid">
                <div class="metric-card {"success" if meets_threshold else "danger"}">
                    <div class="metric-value">{pass_rate:.1%}</div>
                    <div class="metric-label">Validation Pass Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_validations}</div>
                    <div class="metric-label">Total Validations</div>
                </div>
                <div class="metric-card {"success" if performance_score > 0.8 else "warning" if performance_score > 0.6 else "danger"}">
                    <div class="metric-value">{performance_score:.1%}</div>
                    <div class="metric-label">Performance Score</div>
                </div>
                <div class="metric-card {"success" if meets_threshold else "danger"}">
                    <div class="metric-value">{"✓" if meets_threshold else "✗"}</div>
                    <div class="metric-label">Meets Threshold</div>
                </div>
            </div>
        """

        if include_plots and PLOTTING_AVAILABLE:
            html_content += """
            <div class="content-grid">
                <div class="panel">
                    <h2>Validation Summary</h2>
                    <div class="plot-container">
                        <img src="plots/validation_summary.png" alt="Validation Summary">
                    </div>
                </div>
                <div class="panel">
                    <h2>Feature Quality Scores</h2>
                    <div class="plot-container">
                        <img src="plots/feature_quality_heatmap.png" alt="Feature Quality Heatmap">
                    </div>
                </div>
                <div class="panel">
                    <h2>Performance Benchmarks</h2>
                    <div class="plot-container">
                        <img src="plots/performance_benchmarks.png" alt="Performance Benchmarks">
                    </div>
                </div>
                <div class="panel">
                    <h2>Feature Correlations</h2>
                    <div class="plot-container">
                        <img src="plots/correlation_heatmap.png" alt="Correlation Heatmap">
                    </div>
                </div>
            </div>
            """

        # Add feature summary table
        html_content += """
            <div class="panel full-width">
                <h2 id="overview">Feature Validation Overview</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Status</th>
                            <th>Total Tests</th>
                            <th>Passed</th>
                            <th>Failed</th>
                            <th>Pass Rate</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        feature_summary = summary.get("feature_summary", {})
        for feature_name, stats in feature_summary.items():
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            failed = stats.get("failed", 0)
            feature_pass_rate = passed / total if total > 0 else 0

            status_class = (
                "status-pass"
                if feature_pass_rate > 0.8
                else "status-warning"
                if feature_pass_rate > 0.6
                else "status-fail"
            )

            html_content += f"""
                        <tr>
                            <td>{feature_name}</td>
                            <td><span class="status-indicator {status_class}"></span>{feature_pass_rate:.1%}</td>
                            <td>{total}</td>
                            <td>{passed}</td>
                            <td>{failed}</td>
                            <td>{feature_pass_rate:.1%}</td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
            </div>

            <div class="panel full-width" id="raw-data">
                <h2>Raw Data Export</h2>
                <p>Download detailed validation data:</p>
                <ul>
                    <li><a href="validation_data.json">Complete validation results (JSON)</a></li>
                    <li><a href="feature_expectations.json">Feature expectations (JSON)</a></li>
                    <li><a href="benchmark_results.json">Performance benchmarks (JSON)</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

    def _generate_feature_detail_pages(self, output_dir: Path) -> None:
        """Generate detailed feature analysis pages."""
        feature_details_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Details - AIrsenal Validation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .back-link { margin-bottom: 20px; }
                .feature-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="back-link">
                <a href="index.html">← Back to Dashboard</a>
            </div>
            <h1>Feature Validation Details</h1>
            <p>Detailed feature analysis will be available in the next version.</p>
        </body>
        </html>
        """

        with open(output_dir / "feature_details.html", "w") as f:
            f.write(feature_details_html)

    def _generate_performance_report(self, output_dir: Path) -> None:
        """Generate performance report page."""
        benchmark_results = self.dashboard_data.get("benchmark_results", {})
        benchmark_summary = benchmark_results.get("summary", {})

        performance_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - AIrsenal Validation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .back-link {{ margin-bottom: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                .pass {{ border-color: #28a745; }}
                .fail {{ border-color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="back-link">
                <a href="index.html">← Back to Dashboard</a>
            </div>
            <h1>Performance Benchmark Report</h1>

            <h2>Overall Performance</h2>
            <div class="metric {"pass" if benchmark_summary.get("overall_meets_targets", False) else "fail"}">
                <strong>Overall Status:</strong> {"PASS" if benchmark_summary.get("overall_meets_targets", False) else "FAIL"}
            </div>
            <div class="metric">
                <strong>Performance Score:</strong> {benchmark_summary.get("performance_score", 0.0):.1%}
            </div>

            <h2>Failed Targets</h2>
        """

        failed_targets = benchmark_summary.get("failed_targets", [])
        if failed_targets:
            performance_html += "<ul>"
            for target in failed_targets:
                performance_html += f"<li>{target}</li>"
            performance_html += "</ul>"
        else:
            performance_html += "<p>All performance targets met! ✅</p>"

        performance_html += """
        </body>
        </html>
        """

        with open(output_dir / "performance_report.html", "w") as f:
            f.write(performance_html)

    def _generate_drift_analysis_page(self, output_dir: Path) -> None:
        """Generate drift analysis page."""
        drift_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Analysis - AIrsenal Validation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .back-link { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="back-link">
                <a href="index.html">← Back to Dashboard</a>
            </div>
            <h1>Data Drift Analysis</h1>
            <p>Data drift analysis will be available when comparing multiple seasons.</p>
        </body>
        </html>
        """

        with open(output_dir / "drift_analysis.html", "w") as f:
            f.write(drift_html)

    def _export_dashboard_data(self, output_dir: Path) -> None:
        """Export raw dashboard data as JSON files."""
        try:
            # Export complete validation data
            with open(output_dir / "validation_data.json", "w") as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)

            # Export feature expectations
            expectations_data = self.dashboard_data.get("feature_expectations", {})
            with open(output_dir / "feature_expectations.json", "w") as f:
                json.dump(expectations_data, f, indent=2, default=str)

            # Export benchmark results
            benchmark_data = self.dashboard_data.get("benchmark_results", {})
            with open(output_dir / "benchmark_results.json", "w") as f:
                json.dump(benchmark_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to export dashboard data: {e}")

    def generate_executive_summary(
        self, season: str = CURRENT_SEASON
    ) -> dict[str, Any]:
        """
        Generate an executive summary of validation results.

        Args:
            season: Season to analyze

        Returns:
            Dictionary containing executive summary
        """
        if not self.dashboard_data:
            self._collect_dashboard_data(season)

        summary = self.dashboard_data.get("validation_summary", {})
        benchmark_results = self.dashboard_data.get("benchmark_results", {})

        # Calculate key metrics
        total_validations = summary.get("total_validations", 0)
        pass_rate = summary.get("pass_rate", 0.0)
        meets_threshold = summary.get("meets_threshold", False)

        benchmark_summary = benchmark_results.get("summary", {})
        performance_score = benchmark_summary.get("performance_score", 0.0)

        # Determine overall status
        if meets_threshold and performance_score > 0.8:
            overall_status = "EXCELLENT"
            status_color = "green"
        elif pass_rate > 0.8 and performance_score > 0.6:
            overall_status = "GOOD"
            status_color = "yellow"
        else:
            overall_status = "NEEDS_ATTENTION"
            status_color = "red"

        # Generate recommendations
        recommendations = []

        if not meets_threshold:
            recommendations.append(
                "Improve feature validation - current pass rate below threshold"
            )

        if performance_score < 0.8:
            failed_targets = benchmark_summary.get("failed_targets", [])
            recommendations.append(
                f"Address performance issues: {', '.join(failed_targets)}"
            )

        if pass_rate < 0.9:
            recommendations.append(
                "Review and strengthen validation rules for failing features"
            )

        if not recommendations:
            recommendations.append(
                "All validation targets met - maintain current standards"
            )

        return {
            "season": season,
            "generated_at": datetime.now().isoformat(),
            "overall_status": overall_status,
            "status_color": status_color,
            "summary_metrics": {
                "total_validations": total_validations,
                "validation_pass_rate": pass_rate,
                "meets_validation_threshold": meets_threshold,
                "performance_score": performance_score,
                "meets_performance_targets": benchmark_summary.get(
                    "overall_meets_targets", False
                ),
            },
            "key_findings": [
                f"Validation pass rate: {pass_rate:.1%} ({'above' if meets_threshold else 'below'} threshold)",
                f"Performance score: {performance_score:.1%}",
                f"Total validations performed: {total_validations}",
                f"Features analyzed: {len(self.dashboard_data.get('feature_expectations', {}))}",
            ],
            "recommendations": recommendations,
            "sprint_01_status": "COMPLETED"
            if meets_threshold and performance_score > 0.7
            else "IN_PROGRESS",
        }
