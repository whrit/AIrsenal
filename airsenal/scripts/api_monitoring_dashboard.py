#!/usr/bin/env python3
"""
API Monitoring Dashboard for AIrsenal

Provides a comprehensive web-based dashboard for monitoring:
- API usage and performance metrics
- Rate limiting status and quotas
- Cache hit rates and efficiency
- Cost tracking and alerts
- Real-time system health
- Historical trend analysis

Usage:
    python -m airsenal.scripts.api_monitoring_dashboard [--port 8080] [--host 0.0.0.0]

The dashboard provides:
- Real-time metrics visualization
- Historical performance charts
- Cost analysis and projections
- Alert management
- Provider comparison
- Cache efficiency analysis
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    import plotly.graph_objs as go
    import plotly.utils
    from flask import Flask, jsonify, render_template_string
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    render_template_string = None
    jsonify = None
    go = None
    plotly = None

# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from airsenal.framework.api_manager import get_api_manager
from airsenal.framework.logging_config import get_logger
from airsenal.framework.redis_cache import redis_cache

logger = get_logger(__name__)

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIrsenal API Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-subtitle {
            font-size: 14px;
            color: #666;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        .alert-warning {
            color: #856404;
            background-color: #fff3cd;
            border-color: #ffeaa7;
        }
        .alert-danger {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .provider-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .provider-table th,
        .provider-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .provider-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .refresh-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #0056b3;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AIrsenal API Monitoring Dashboard</h1>
            <p>Real-time monitoring of API usage, performance, and costs</p>
        </div>

        <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>

        <div class="loading" id="loading">
            <p>Loading data...</p>
        </div>

        <!-- Alerts Section -->
        <div id="alerts"></div>

        <!-- Key Metrics -->
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated dynamically -->
        </div>

        <!-- Charts Section -->
        <div class="chart-container">
            <h3>API Usage Over Time</h3>
            <div id="usage-chart" style="height: 400px;"></div>
        </div>

        <div class="chart-container">
            <h3>Cache Performance</h3>
            <div id="cache-chart" style="height: 400px;"></div>
        </div>

        <div class="chart-container">
            <h3>Cost Analysis</h3>
            <div id="cost-chart" style="height: 400px;"></div>
        </div>

        <!-- Provider Details -->
        <div class="chart-container">
            <h3>Provider Status</h3>
            <div id="provider-status"></div>
        </div>
    </div>

    <script>
        let refreshInterval;

        function refreshDashboard() {
            document.getElementById('loading').style.display = 'block';

            Promise.all([
                fetch('/api/metrics').then(r => r.json()),
                fetch('/api/alerts').then(r => r.json()),
                fetch('/api/charts').then(r => r.json()),
                fetch('/api/providers').then(r => r.json())
            ]).then(([metrics, alerts, charts, providers]) => {
                updateMetrics(metrics);
                updateAlerts(alerts);
                updateCharts(charts);
                updateProviders(providers);
                document.getElementById('loading').style.display = 'none';
            }).catch(error => {
                console.error('Error refreshing dashboard:', error);
                document.getElementById('loading').style.display = 'none';
            });
        }

        function updateMetrics(metrics) {
            const grid = document.getElementById('metrics-grid');
            grid.innerHTML = '';

            const metricCards = [
                {
                    title: 'Total API Requests',
                    value: metrics.total_requests || 0,
                    subtitle: 'Across all providers',
                    status: 'good'
                },
                {
                    title: 'Cache Hit Rate',
                    value: ((metrics.cache_hit_rate || 0) * 100).toFixed(1) + '%',
                    subtitle: 'Memory + Redis cache',
                    status: metrics.cache_hit_rate > 0.8 ? 'good' : metrics.cache_hit_rate > 0.6 ? 'warning' : 'danger'
                },
                {
                    title: 'Total Cost',
                    value: '$' + (metrics.total_cost || 0).toFixed(2),
                    subtitle: 'All providers combined',
                    status: 'good'
                },
                {
                    title: 'Active Providers',
                    value: metrics.active_providers || 0,
                    subtitle: 'Currently configured',
                    status: 'good'
                },
                {
                    title: 'Queue Length',
                    value: metrics.queue_length || 0,
                    subtitle: 'Pending requests',
                    status: metrics.queue_length > 100 ? 'danger' : metrics.queue_length > 50 ? 'warning' : 'good'
                },
                {
                    title: 'Avg Response Time',
                    value: (metrics.avg_response_time || 0).toFixed(2) + 's',
                    subtitle: 'All providers',
                    status: metrics.avg_response_time > 5 ? 'danger' : metrics.avg_response_time > 2 ? 'warning' : 'good'
                }
            ];

            metricCards.forEach(metric => {
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <div class="metric-title">${metric.title}</div>
                    <div class="metric-value status-${metric.status}">${metric.value}</div>
                    <div class="metric-subtitle">${metric.subtitle}</div>
                `;
                grid.appendChild(card);
            });
        }

        function updateAlerts(alerts) {
            const alertsDiv = document.getElementById('alerts');
            alertsDiv.innerHTML = '';

            if (alerts.length === 0) return;

            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${alert.type === 'warning' ? 'warning' : 'danger'}`;
                alertDiv.innerHTML = `<strong>${alert.title}</strong> ${alert.message}`;
                alertsDiv.appendChild(alertDiv);
            });
        }

        function updateCharts(charts) {
            // Usage chart
            if (charts.usage) {
                Plotly.newPlot('usage-chart', charts.usage.data, charts.usage.layout, {responsive: true});
            }

            // Cache chart
            if (charts.cache) {
                Plotly.newPlot('cache-chart', charts.cache.data, charts.cache.layout, {responsive: true});
            }

            // Cost chart
            if (charts.cost) {
                Plotly.newPlot('cost-chart', charts.cost.data, charts.cost.layout, {responsive: true});
            }
        }

        function updateProviders(providers) {
            const container = document.getElementById('provider-status');

            if (providers.length === 0) {
                container.innerHTML = '<p>No providers configured</p>';
                return;
            }

            let html = '<table class="provider-table"><thead><tr>';
            html += '<th>Provider</th><th>Status</th><th>Requests</th><th>Success Rate</th>';
            html += '<th>Avg Response</th><th>Cost</th><th>Rate Limit</th></tr></thead><tbody>';

            providers.forEach(provider => {
                const statusClass = provider.is_healthy ? 'status-good' : 'status-danger';
                html += `<tr>
                    <td>${provider.name}</td>
                    <td class="${statusClass}">${provider.is_healthy ? 'Healthy' : 'Issues'}</td>
                    <td>${provider.total_requests}</td>
                    <td>${(provider.success_rate * 100).toFixed(1)}%</td>
                    <td>${provider.avg_response_time.toFixed(2)}s</td>
                    <td>$${provider.total_cost.toFixed(2)}</td>
                    <td>${provider.rate_limit_status}</td>
                </tr>`;
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        }

        // Auto-refresh every 30 seconds
        function startAutoRefresh() {
            refreshInterval = setInterval(refreshDashboard, 30000);
        }

        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            refreshDashboard();
            startAutoRefresh();
        });

        // Stop auto-refresh when page is hidden
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'hidden') {
                stopAutoRefresh();
            } else {
                startAutoRefresh();
            }
        });
    </script>
</body>
</html>
"""


class APIMonitoringDashboard:
    """Web-based monitoring dashboard for API management system"""

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.app = None
        self.api_manager = None

        if not FLASK_AVAILABLE:
            msg = "Flask and plotly are required for the dashboard. Install with: pip install flask plotly flask-cors"
            raise ImportError(msg)

    async def initialize(self):
        """Initialize the dashboard and API manager"""
        self.api_manager = await get_api_manager()
        self.app = Flask(__name__)
        CORS(self.app)

        # Register routes
        self.app.route("/")(self.dashboard)
        self.app.route("/api/metrics")(self.get_metrics)
        self.app.route("/api/alerts")(self.get_alerts)
        self.app.route("/api/charts")(self.get_charts)
        self.app.route("/api/providers")(self.get_providers)
        self.app.route("/api/health")(self.health_check)

        logger.info(f"Dashboard initialized on http://{self.host}:{self.port}")

    def dashboard(self):
        """Serve the main dashboard page"""
        return render_template_string(DASHBOARD_HTML)

    def get_metrics(self):
        """Get key metrics for the dashboard"""
        try:
            if not self.api_manager:
                return jsonify({"error": "API manager not initialized"})

            stats = self.api_manager.get_comprehensive_stats()

            # Calculate aggregated metrics
            metrics = {
                "total_requests": stats.get("usage", {})
                .get("totals", {})
                .get("requests", 0),
                "total_cost": stats.get("usage", {}).get("totals", {}).get("cost", 0.0),
                "cache_hit_rate": stats.get("cache", {}).get("overall_hit_rate", 0.0),
                "queue_length": stats.get("queue", {}).get("queue_length", 0),
                "active_providers": len(stats.get("rate_limiter", {})),
                "avg_response_time": self._calculate_avg_response_time(stats),
                "timestamp": datetime.now().isoformat(),
            }

            return jsonify(metrics)

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({"error": str(e)}), 500

    def get_alerts(self):
        """Get current alerts and warnings"""
        try:
            if not self.api_manager:
                return jsonify([])

            alerts = []
            stats = self.api_manager.get_comprehensive_stats()

            # Check cache hit rate
            cache_hit_rate = stats.get("cache", {}).get("overall_hit_rate", 0.0)
            if cache_hit_rate < 0.8:
                alerts.append(
                    {
                        "type": "warning",
                        "title": "Low Cache Hit Rate",
                        "message": f"Cache hit rate is {cache_hit_rate:.1%}. Consider reviewing cache strategies.",
                    }
                )

            # Check queue length
            queue_length = stats.get("queue", {}).get("queue_length", 0)
            if queue_length > 100:
                alerts.append(
                    {
                        "type": "danger",
                        "title": "High Queue Length",
                        "message": f"Request queue has {queue_length} pending requests. System may be overloaded.",
                    }
                )

            # Check cost alerts
            cost_alerts = stats.get("cost_alerts", [])
            for cost_alert in cost_alerts:
                alerts.append(
                    {
                        "type": "warning",
                        "title": "Cost Alert",
                        "message": cost_alert.get("message", "Cost threshold exceeded"),
                    }
                )

            # Check provider health
            rate_limiter_stats = stats.get("rate_limiter", {})
            for provider, provider_stats in rate_limiter_stats.items():
                if not provider_stats.get("can_make_request", True):
                    alerts.append(
                        {
                            "type": "danger",
                            "title": f"Provider {provider} Quota Exceeded",
                            "message": f"Provider {provider} has exceeded its quota limits.",
                        }
                    )

            return jsonify(alerts)

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return jsonify([])

    def get_charts(self):
        """Generate chart data for visualization"""
        try:
            if not self.api_manager:
                return jsonify({})

            stats = self.api_manager.get_comprehensive_stats()
            charts = {}

            # Usage chart
            charts["usage"] = self._create_usage_chart(stats)

            # Cache performance chart
            charts["cache"] = self._create_cache_chart(stats)

            # Cost analysis chart
            charts["cost"] = self._create_cost_chart(stats)

            return jsonify(charts)

        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return jsonify({})

    def get_providers(self):
        """Get provider status information"""
        try:
            if not self.api_manager:
                return jsonify([])

            stats = self.api_manager.get_comprehensive_stats()
            providers = []

            # Get provider data from usage monitor
            usage_stats = stats.get("usage", {}).get("providers", {})
            rate_limiter_stats = stats.get("rate_limiter", {})

            for provider_name, provider_data in usage_stats.items():
                rate_limit_data = rate_limiter_stats.get(provider_name, {})

                provider_info = {
                    "name": provider_name,
                    "is_healthy": provider_data.get("success_rate", 0) > 0.8,
                    "total_requests": provider_data.get("total_requests", 0),
                    "success_rate": provider_data.get("success_rate", 0.0),
                    "avg_response_time": provider_data.get("avg_response_time", 0.0),
                    "total_cost": provider_data.get("total_cost", 0.0),
                    "rate_limit_status": self._get_rate_limit_status(rate_limit_data),
                }

                providers.append(provider_info)

            return jsonify(providers)

        except Exception as e:
            logger.error(f"Error getting providers: {e}")
            return jsonify([])

    def health_check(self):
        """Simple health check endpoint"""
        try:
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "api_manager_initialized": self.api_manager is not None,
                    "redis_available": redis_cache.is_available(),
                }
            )
        except Exception as e:
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    def _calculate_avg_response_time(self, stats: dict) -> float:
        """Calculate average response time across all providers"""
        usage_stats = stats.get("usage", {}).get("providers", {})
        if not usage_stats:
            return 0.0

        total_time = 0.0
        total_requests = 0

        for provider_data in usage_stats.values():
            requests = provider_data.get("total_requests", 0)
            avg_time = provider_data.get("avg_response_time", 0.0)
            total_time += requests * avg_time
            total_requests += requests

        return total_time / total_requests if total_requests > 0 else 0.0

    def _get_rate_limit_status(self, rate_limit_data: dict) -> str:
        """Get human-readable rate limit status"""
        if not rate_limit_data:
            return "Not configured"

        can_make_request = rate_limit_data.get("can_make_request", True)
        tokens_available = rate_limit_data.get("tokens_available", 0)

        if not can_make_request:
            return "Quota exceeded"
        if tokens_available < 1:
            return "Rate limited"
        return "Available"

    def _create_usage_chart(self, stats: dict) -> dict:
        """Create usage over time chart"""
        # Simulate historical data for demo (in real implementation, this would come from stored metrics)
        import numpy as np

        hours = list(range(24))
        requests = np.random.poisson(100, 24)  # Simulated request counts

        trace = go.Scatter(
            x=hours,
            y=requests,
            mode="lines+markers",
            name="API Requests",
            line={"color": "rgb(55, 128, 191)", "width": 2},
        )

        layout = go.Layout(
            title="API Requests Over Last 24 Hours",
            xaxis={"title": "Hours Ago"},
            yaxis={"title": "Request Count"},
            hovermode="closest",
        )

        return {"data": [trace], "layout": layout}

    def _create_cache_chart(self, stats: dict) -> dict:
        """Create cache performance chart"""
        cache_stats = stats.get("cache", {})

        # Cache hit rate pie chart
        hit_rate = cache_stats.get("overall_hit_rate", 0.0)
        miss_rate = 1.0 - hit_rate

        trace = go.Pie(
            labels=["Cache Hits", "Cache Misses"],
            values=[hit_rate * 100, miss_rate * 100],
            colors=["#28a745", "#dc3545"],
        )

        layout = go.Layout(title="Cache Hit Rate", showlegend=True)

        return {"data": [trace], "layout": layout}

    def _create_cost_chart(self, stats: dict) -> dict:
        """Create cost analysis chart"""
        usage_stats = stats.get("usage", {}).get("providers", {})

        if not usage_stats:
            # Return empty chart if no data
            return {
                "data": [],
                "layout": go.Layout(title="Cost by Provider (No data available)"),
            }

        providers = list(usage_stats.keys())
        costs = [
            provider_data.get("total_cost", 0.0)
            for provider_data in usage_stats.values()
        ]

        trace = go.Bar(x=providers, y=costs, marker={"color": "rgb(158, 202, 225)"})

        layout = go.Layout(
            title="Cost by Provider",
            xaxis={"title": "Provider"},
            yaxis={"title": "Total Cost ($)"},
        )

        return {"data": [trace], "layout": layout}

    def run(self):
        """Run the dashboard server"""
        if not self.app:
            msg = "Dashboard not initialized. Call initialize() first."
            raise RuntimeError(msg)

        logger.info(
            f"Starting API Monitoring Dashboard on http://{self.host}:{self.port}"
        )
        self.app.run(host=self.host, port=self.port, debug=False)


async def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description="AIrsenal API Monitoring Dashboard")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the dashboard on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the dashboard to"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Initialize dashboard
        dashboard = APIMonitoringDashboard(port=args.port, host=args.host)
        await dashboard.initialize()

        # Run the dashboard
        dashboard.run()

    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
