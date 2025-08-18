"""
AIrsenal Performance Dashboard

Real-time performance monitoring and historical analysis dashboard.
Provides web interface for viewing benchmark results, system metrics,
and performance trends.

Usage:
    airsenal_performance_dashboard --port 8080
    airsenal_performance_dashboard --data-dir benchmarks/reports
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import psutil

mpl.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus client not available - metrics collection disabled")


class PerformanceMetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self):
        self.metrics_history = []
        self.alerts = []

        if PROMETHEUS_AVAILABLE:
            # Prometheus metrics
            self.cpu_usage = Gauge("airsenal_cpu_usage_percent", "CPU usage percentage")
            self.memory_usage = Gauge("airsenal_memory_usage_mb", "Memory usage in MB")
            self.prediction_latency = Histogram(
                "airsenal_prediction_latency_seconds", "Prediction latency"
            )
            self.cache_hit_rate = Gauge("airsenal_cache_hit_rate", "Cache hit rate")
            self.active_users = Gauge("airsenal_active_users", "Number of active users")
            self.requests_total = Counter(
                "airsenal_requests_total", "Total requests", ["endpoint", "method"]
            )

    def collect_system_metrics(self) -> dict[str, float]:
        """Collect current system metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "load_average": os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
        }

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.cpu_usage.set(metrics["cpu_percent"])
            self.memory_usage.set(metrics["memory_mb"])

        self.metrics_history.append(metrics)

        # Keep only last 1000 metrics (about 16 minutes at 1s intervals)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        return metrics

    def check_alerts(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Check for performance alerts."""
        new_alerts = []

        # CPU usage alert
        if metrics["cpu_percent"] > 80:
            new_alerts.append(
                {
                    "type": "cpu_high",
                    "message": f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                    "severity": "warning",
                    "timestamp": metrics["timestamp"],
                }
            )

        # Memory usage alert
        if metrics["memory_percent"] > 85:
            new_alerts.append(
                {
                    "type": "memory_high",
                    "message": f"High memory usage: {metrics['memory_percent']:.1f}%",
                    "severity": "warning",
                    "timestamp": metrics["timestamp"],
                }
            )

        # Load average alert
        if metrics["load_average"] > psutil.cpu_count():
            new_alerts.append(
                {
                    "type": "load_high",
                    "message": f"High load average: {metrics['load_average']:.2f}",
                    "severity": "critical",
                    "timestamp": metrics["timestamp"],
                }
            )

        self.alerts.extend(new_alerts)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        return new_alerts


class BenchmarkResultsAnalyzer:
    """Analyzes benchmark results and generates reports."""

    def __init__(self, data_dir: str = "benchmarks/reports"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_benchmark_results(self) -> list[dict[str, Any]]:
        """Load all benchmark result files."""
        results = []

        for result_file in self.data_dir.glob("benchmark_results_*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    data["filename"] = result_file.name
                    results.append(data)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")

        return sorted(results, key=lambda x: x.get("timestamp", 0), reverse=True)

    def generate_performance_trends(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate performance trend analysis."""
        if not results:
            return {"error": "No benchmark results found"}

        trends = {
            "summary": {
                "total_runs": len(results),
                "date_range": {
                    "start": min(r.get("timestamp", 0) for r in results),
                    "end": max(r.get("timestamp", 0) for r in results),
                },
            },
            "metrics": {},
        }

        # Analyze key metrics over time
        metric_data = {}
        for result in results:
            timestamp = result.get("timestamp", 0)
            summary = result.get("summary", {})

            for metric in ["avg_duration", "max_memory_mb", "total_tests"]:
                if metric in summary:
                    if metric not in metric_data:
                        metric_data[metric] = []
                    metric_data[metric].append(
                        {"timestamp": timestamp, "value": summary[metric]}
                    )

        # Calculate trends
        for metric, data in metric_data.items():
            if len(data) >= 2:
                values = [d["value"] for d in data]
                trends["metrics"][metric] = {
                    "current": values[-1],
                    "previous": values[-2],
                    "change_percent": ((values[-1] - values[-2]) / values[-2] * 100)
                    if values[-2] != 0
                    else 0,
                    "trend": "improving" if values[-1] < values[-2] else "degrading",
                    "data_points": len(data),
                }

        return trends

    def create_performance_plots(self, results: list[dict[str, Any]]) -> list[str]:
        """Create performance visualization plots."""
        plot_files = []

        if not results:
            return plot_files

        # Set up matplotlib style
        plt.style.use("seaborn-v0_8")

        # 1. Duration trends over time
        fig, ax = plt.subplots(figsize=(12, 6))

        timestamps = []
        avg_durations = []
        max_durations = []

        for result in results:
            if "summary" in result and "avg_duration" in result["summary"]:
                timestamps.append(datetime.fromtimestamp(result.get("timestamp", 0)))
                avg_durations.append(result["summary"]["avg_duration"])
                max_durations.append(result["summary"].get("max_duration", 0))

        if timestamps:
            ax.plot(timestamps, avg_durations, label="Average Duration", marker="o")
            ax.plot(
                timestamps, max_durations, label="Max Duration", marker="s", alpha=0.7
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Duration (seconds)")
            ax.set_title("Performance Duration Trends")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            duration_plot = self.data_dir / "performance_duration_trends.png"
            plt.savefig(duration_plot, dpi=150, bbox_inches="tight")
            plot_files.append(str(duration_plot))
            plt.close()

        # 2. Memory usage trends
        fig, ax = plt.subplots(figsize=(12, 6))

        memory_usage = []
        for result in results:
            if "summary" in result and "max_memory_mb" in result["summary"]:
                memory_usage.append(result["summary"]["max_memory_mb"])

        if memory_usage and timestamps:
            ax.plot(
                timestamps[-len(memory_usage) :],
                memory_usage,
                label="Peak Memory Usage",
                marker="o",
                color="red",
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Memory Usage (MB)")
            ax.set_title("Memory Usage Trends")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            memory_plot = self.data_dir / "performance_memory_trends.png"
            plt.savefig(memory_plot, dpi=150, bbox_inches="tight")
            plot_files.append(str(memory_plot))
            plt.close()

        # 3. Test execution summary
        if results:
            latest_result = results[0]
            if "results" in latest_result:
                test_results = latest_result["results"]

                # Group by test category
                categories = {}
                for test in test_results:
                    test_name = test.get("test_name", "unknown")
                    category = test_name.split("_")[0] if "_" in test_name else "other"

                    if category not in categories:
                        categories[category] = []
                    categories[category].append(test.get("duration", 0))

                if categories:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    category_names = list(categories.keys())
                    avg_durations = [
                        sum(durations) / len(durations)
                        for durations in categories.values()
                    ]

                    ax.bar(category_names, avg_durations)
                    ax.set_xlabel("Test Category")
                    ax.set_ylabel("Average Duration (seconds)")
                    ax.set_title("Latest Benchmark Results by Category")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    category_plot = self.data_dir / "performance_categories.png"
                    plt.savefig(category_plot, dpi=150, bbox_inches="tight")
                    plot_files.append(str(category_plot))
                    plt.close()

        return plot_files


class PerformanceDashboard:
    """Main performance dashboard application."""

    def __init__(self, data_dir: str = "benchmarks/reports", port: int = 8080):
        self.data_dir = data_dir
        self.port = port
        self.metrics_collector = PerformanceMetricsCollector()
        self.results_analyzer = BenchmarkResultsAnalyzer(data_dir)

        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.port + 1)  # Metrics on port+1
                print(
                    f"Prometheus metrics available at http://localhost:{self.port + 1}/metrics"
                )
            except Exception as e:
                print(f"Failed to start Prometheus server: {e}")

    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard."""
        # Collect current metrics
        current_metrics = self.metrics_collector.collect_system_metrics()
        self.metrics_collector.check_alerts(current_metrics)

        # Load benchmark results
        benchmark_results = self.results_analyzer.load_benchmark_results()
        trends = self.results_analyzer.generate_performance_trends(benchmark_results)
        plot_files = self.results_analyzer.create_performance_plots(benchmark_results)

        # Generate HTML
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>AIrsenal Performance Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-top: 10px; }}
        .alert {{ padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .alert-warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
        .alert-critical {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .trend-up {{ color: #dc3545; }}
        .trend-down {{ color: #28a745; }}
    </style>
    <script>
        function refreshPage() {{
            window.location.reload();
        }}
        setInterval(refreshPage, 30000); // Refresh every 30 seconds
    </script>
</head>
<body>
    <div class="container">
        <h1>AIrsenal Performance Dashboard</h1>
        <p class="timestamp">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="card">
            <h2>Current System Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{current_metrics["cpu_percent"]:.1f}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{current_metrics["memory_mb"]:.0f} MB</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{current_metrics["memory_percent"]:.1f}%</div>
                    <div class="metric-label">Memory %</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{current_metrics["load_average"]:.2f}</div>
                    <div class="metric-label">Load Average</div>
                </div>
            </div>
        </div>

        {self._generate_alerts_html()}

        <div class="card">
            <h2>Performance Trends</h2>
            {self._generate_trends_html(trends)}
        </div>

        <div class="card">
            <h2>Performance Plots</h2>
            {self._generate_plots_html(plot_files)}
        </div>

        <div class="card">
            <h2>Recent Benchmark Results</h2>
            {self._generate_results_table_html(benchmark_results[:10])}
        </div>
    </div>
</body>
</html>
"""

    def _generate_alerts_html(self) -> str:
        """Generate alerts section HTML."""
        if not self.metrics_collector.alerts:
            return '<div class="card"><h2>Alerts</h2><p>No active alerts</p></div>'

        alerts_html = '<div class="card"><h2>Active Alerts</h2>'

        for alert in self.metrics_collector.alerts[-5:]:  # Show last 5 alerts
            severity_class = f"alert-{alert['severity']}"
            timestamp = datetime.fromtimestamp(alert["timestamp"]).strftime("%H:%M:%S")
            alerts_html += f"""
            <div class="alert {severity_class}">
                <strong>{alert["type"].upper()}</strong>: {alert["message"]}
                <span style="float: right;">{timestamp}</span>
            </div>
            """

        alerts_html += "</div>"
        return alerts_html

    def _generate_trends_html(self, trends: dict[str, Any]) -> str:
        """Generate trends section HTML."""
        if "error" in trends:
            return f"<p>{trends['error']}</p>"

        html = f"<p>Based on {trends['summary']['total_runs']} benchmark runs</p>"
        html += '<div class="metrics-grid">'

        for metric, data in trends.get("metrics", {}).items():
            trend_class = "trend-down" if data["trend"] == "improving" else "trend-up"
            change_symbol = "↓" if data["trend"] == "improving" else "↑"

            html += f"""
            <div class="metric">
                <div class="metric-value {trend_class}">{data["current"]:.3f}</div>
                <div class="metric-label">{metric.replace("_", " ").title()}</div>
                <div class="metric-label {trend_class}">
                    {change_symbol} {abs(data["change_percent"]):.1f}%
                </div>
            </div>
            """

        html += "</div>"
        return html

    def _generate_plots_html(self, plot_files: list[str]) -> str:
        """Generate plots section HTML."""
        if not plot_files:
            return "<p>No performance plots available</p>"

        html = ""
        for plot_file in plot_files:
            plot_name = Path(plot_file).name
            # Convert absolute path to relative path for web serving
            relative_path = f"reports/{plot_name}"
            html += f'''
            <div class="plot">
                <h3>{plot_name.replace("_", " ").replace(".png", "").title()}</h3>
                <img src="{relative_path}" alt="{plot_name}">
            </div>
            '''

        return html

    def _generate_results_table_html(self, results: list[dict[str, Any]]) -> str:
        """Generate results table HTML."""
        if not results:
            return "<p>No benchmark results found</p>"

        html = """
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Total Tests</th>
                    <th>Avg Duration (s)</th>
                    <th>Max Memory (MB)</th>
                    <th>Failed Tests</th>
                </tr>
            </thead>
            <tbody>
        """

        for result in results:
            summary = result.get("summary", {})
            timestamp = datetime.fromtimestamp(result.get("timestamp", 0)).strftime(
                "%Y-%m-%d %H:%M"
            )

            html += f"""
            <tr>
                <td>{timestamp}</td>
                <td>{summary.get("total_tests", "N/A")}</td>
                <td>{summary.get("avg_duration", 0):.3f}</td>
                <td>{summary.get("max_memory_mb", 0):.1f}</td>
                <td>{summary.get("failed_tests", 0)}</td>
            </tr>
            """

        html += "</tbody></table>"
        return html

    def save_dashboard(self) -> str:
        """Save dashboard to HTML file."""
        html_content = self.generate_html_dashboard()
        dashboard_file = Path(self.data_dir) / "performance_dashboard.html"

        with open(dashboard_file, "w") as f:
            f.write(html_content)

        return str(dashboard_file)

    def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        print("Starting performance monitoring...")
        print(f"Dashboard will be saved to: {self.data_dir}/performance_dashboard.html")

        try:
            while True:
                # Update dashboard
                dashboard_file = self.save_dashboard()
                print(f"Dashboard updated: {dashboard_file}")

                # Wait before next update
                time.sleep(30)  # Update every 30 seconds

        except KeyboardInterrupt:
            print("\nMonitoring stopped")


def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description="AIrsenal Performance Dashboard")

    parser.add_argument(
        "--data-dir",
        default="benchmarks/reports",
        help="Directory containing benchmark results",
    )

    parser.add_argument(
        "--port", type=int, default=8080, help="Port for metrics server"
    )

    parser.add_argument(
        "--generate-only", action="store_true", help="Generate dashboard once and exit"
    )

    parser.add_argument(
        "--monitor", action="store_true", help="Run continuous monitoring"
    )

    args = parser.parse_args()

    # Create dashboard
    dashboard = PerformanceDashboard(args.data_dir, args.port)

    if args.generate_only:
        dashboard_file = dashboard.save_dashboard()
        print(f"Dashboard generated: {dashboard_file}")
    elif args.monitor:
        dashboard.run_monitoring_loop()
    else:
        # Generate once and show info
        dashboard_file = dashboard.save_dashboard()
        print(f"Dashboard generated: {dashboard_file}")
        print("To run continuous monitoring: airsenal_performance_dashboard --monitor")
        if PROMETHEUS_AVAILABLE:
            print(
                f"Prometheus metrics available at: http://localhost:{args.port + 1}/metrics"
            )


if __name__ == "__main__":
    main()
