#!/usr/bin/env python
"""
xG/xA Data Provider Monitoring Dashboard

Provides web-based monitoring dashboard for xG data providers with:
- Real-time health status
- API cost tracking
- Performance metrics
- Alert configuration
"""

import json
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from airsenal.framework.logging_config import get_logger
from airsenal.framework.xg_data_provider import create_xg_manager
from airsenal.framework.env import AIRSENAL_HOME

logger = get_logger(__name__)

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AIrsenal xG Data Provider Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .card { 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .status-healthy { color: #28a745; }
        .status-unhealthy { color: #dc3545; }
        .metric { 
            display: inline-block; 
            margin: 10px; 
            padding: 10px; 
            background: #f8f9fa; 
            border-radius: 4px; 
        }
        .metric-label { 
            font-weight: bold; 
            color: #666; 
        }
        .metric-value { 
            font-size: 1.2em; 
            color: #333; 
        }
        .recommendation { 
            background: #fff3cd; 
            padding: 10px; 
            margin: 5px 0; 
            border-left: 4px solid #ffc107; 
            border-radius: 4px; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .refresh-button { 
            background: #007bff; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 4px; 
            cursor: pointer; 
        }
        .auto-refresh { 
            margin-left: 10px; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 10px 0; 
        }
        th, td { 
            padding: 8px; 
            text-align: left; 
            border-bottom: 1px solid #ddd; 
        }
        th { 
            background-color: #f8f9fa; 
        }
    </style>
    <script>
        let autoRefresh = false;
        let refreshInterval;
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const button = document.getElementById('autoRefreshBtn');
            
            if (autoRefresh) {
                button.textContent = 'Stop Auto Refresh';
                refreshInterval = setInterval(refreshData, 30000); // 30 seconds
            } else {
                button.textContent = 'Start Auto Refresh';
                clearInterval(refreshInterval);
            }
        }
        
        function refreshData() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error refreshing data:', error));
        }
        
        function updateDashboard(data) {
            // Update timestamp
            document.getElementById('lastUpdate').textContent = 
                new Date(data.health_status.timestamp).toLocaleString();
            
            // Update provider status
            const providersDiv = document.getElementById('providers');
            providersDiv.innerHTML = '';
            
            for (const [name, status] of Object.entries(data.health_status.providers)) {
                const providerDiv = document.createElement('div');
                providerDiv.className = 'card';
                providerDiv.innerHTML = `
                    <h3>${name.toUpperCase()} Provider</h3>
                    <div class="metric">
                        <div class="metric-label">Status</div>
                        <div class="metric-value ${status.is_healthy ? 'status-healthy' : 'status-unhealthy'}">
                            ${status.is_healthy ? '🟢 Healthy' : '🔴 Unhealthy'}
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Success Rate</div>
                        <div class="metric-value">${(status.success_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Requests</div>
                        <div class="metric-value">${status.total_requests}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Average Response Time</div>
                        <div class="metric-value">${status.average_response_time.toFixed(2)}s</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Circuit Breaker</div>
                        <div class="metric-value">${status.circuit_breaker_state}</div>
                    </div>
                `;
                providersDiv.appendChild(providerDiv);
            }
            
            // Update cost summary
            const costDiv = document.getElementById('costSummary');
            costDiv.innerHTML = `
                <div class="metric">
                    <div class="metric-label">Total Cost</div>
                    <div class="metric-value">$${data.cost_summary.total_cost.toFixed(4)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Requests</div>
                    <div class="metric-value">${data.cost_summary.total_requests}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Average Cost/Request</div>
                    <div class="metric-value">$${data.cost_summary.average_cost_per_request.toFixed(4)}</div>
                </div>
            `;
            
            // Update recommendations
            const recDiv = document.getElementById('recommendations');
            if (data.recommendations.length > 0) {
                recDiv.innerHTML = data.recommendations.map(rec => 
                    `<div class="recommendation">${rec.provider} - ${rec.type}: ${rec.message}</div>`
                ).join('');
            } else {
                recDiv.innerHTML = '<p>No recommendations at this time.</p>';
            }
        }
        
        // Auto-refresh on page load
        document.addEventListener('DOMContentLoaded', function() {
            refreshData();
        });
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 AIrsenal xG Data Provider Dashboard</h1>
            <button class="refresh-button" onclick="refreshData()">Refresh Now</button>
            <button id="autoRefreshBtn" class="refresh-button auto-refresh" onclick="toggleAutoRefresh()">
                Start Auto Refresh
            </button>
            <p>Last updated: <span id="lastUpdate">Loading...</span></p>
        </div>
        
        <div class="card">
            <h2>📊 Provider Health Status</h2>
            <div id="providers">Loading...</div>
        </div>
        
        <div class="card">
            <h2>💰 Cost Summary</h2>
            <div id="costSummary">Loading...</div>
        </div>
        
        <div class="card">
            <h2>💡 Recommendations</h2>
            <div id="recommendations">Loading...</div>
        </div>
        
        <div class="card">
            <h2>📈 Historical Metrics</h2>
            <p>Historical metrics tracking will be available in future versions.</p>
            <p>Consider integrating with tools like Grafana for advanced visualization.</p>
        </div>
    </div>
</body>
</html>
"""


class XGMonitoringDashboard:
    """Web-based monitoring dashboard for xG data providers"""
    
    def __init__(self):
        self.manager = create_xg_manager()
        self.metrics_file = Path(AIRSENAL_HOME) / "xg_metrics.json"
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self._setup_routes()
        else:
            logger.warning("Flask not available. Web dashboard disabled.")
            self.app = None
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/dashboard')
        def api_dashboard():
            """API endpoint for dashboard data"""
            try:
                data = self.manager.get_health_dashboard()
                return jsonify(data)
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/health')
        def api_health():
            """Health check endpoint"""
            try:
                health = self.manager.monitor.get_health_status()
                return jsonify(health)
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/costs')
        def api_costs():
            """Cost summary endpoint"""
            try:
                costs = self.manager.monitor.get_cost_summary()
                return jsonify(costs)
            except Exception as e:
                logger.error(f"Error getting cost summary: {e}")
                return jsonify({"error": str(e)}), 500
    
    def save_metrics_snapshot(self):
        """Save current metrics to file for historical tracking"""
        try:
            dashboard_data = self.manager.get_health_dashboard()
            dashboard_data["snapshot_time"] = datetime.now().isoformat()
            
            # Load existing snapshots
            snapshots = []
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        snapshots = existing_data.get("snapshots", [])
                    except json.JSONDecodeError:
                        logger.warning("Invalid metrics file, starting fresh")
            
            # Add new snapshot
            snapshots.append(dashboard_data)
            
            # Keep only last 100 snapshots
            if len(snapshots) > 100:
                snapshots = snapshots[-100:]
            
            # Save back to file
            with open(self.metrics_file, 'w') as f:
                json.dump({"snapshots": snapshots}, f, indent=2)
            
            logger.info(f"Saved metrics snapshot to {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics snapshot: {e}")
    
    def run_web_dashboard(self, host='127.0.0.1', port=5000, debug=False):
        """Run the web dashboard"""
        if not self.app:
            logger.error("Flask not available. Cannot run web dashboard.")
            return False
        
        try:
            logger.info(f"Starting xG monitoring dashboard on http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
            return True
        except Exception as e:
            logger.error(f"Error running web dashboard: {e}")
            return False
    
    def print_console_dashboard(self):
        """Print dashboard data to console"""
        try:
            dashboard = self.manager.get_health_dashboard()
            
            print("\n" + "="*60)
            print("🎯 AIrsenal xG Data Provider Dashboard")
            print("="*60)
            
            # Health Status
            print("\n📊 Provider Health Status:")
            print("-" * 40)
            
            health = dashboard["health_status"]
            for provider_name, status in health["providers"].items():
                status_icon = "🟢" if status["is_healthy"] else "🔴"
                print(f"\n{provider_name.upper()} Provider {status_icon}")
                print(f"  Circuit Breaker: {status['circuit_breaker_state']}")
                print(f"  Success Rate: {status['success_rate']:.1%}")
                print(f"  Total Requests: {status['total_requests']}")
                print(f"  Failed Requests: {status['failed_requests']}")
                print(f"  Avg Response Time: {status['average_response_time']:.2f}s")
                print(f"  Rate Limit Hits: {status['rate_limit_hits']}")
                print(f"  Tier: {status['tier']}")
            
            # Cost Summary
            print("\n💰 Cost Summary:")
            print("-" * 40)
            
            costs = dashboard["cost_summary"]
            print(f"Total Cost: ${costs['total_cost']:.4f}")
            print(f"Total Requests: {costs['total_requests']}")
            
            if costs['total_requests'] > 0:
                print(f"Average Cost/Request: ${costs['average_cost_per_request']:.4f}")
            
            print("\nProvider Breakdown:")
            for provider, cost_data in costs["provider_breakdown"].items():
                print(f"  {provider}: ${cost_data['total_cost']:.4f} ({cost_data['requests']} requests)")
            
            # Recommendations
            recommendations = dashboard.get("recommendations", [])
            if recommendations:
                print("\n💡 Recommendations:")
                print("-" * 40)
                for rec in recommendations:
                    print(f"  • {rec['provider']} - {rec['type'].title()}: {rec['message']}")
            else:
                print("\n💡 No recommendations at this time.")
            
            print(f"\n⏰ Last updated: {health['timestamp']}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing console dashboard: {e}")
    
    def generate_alert_config(self, output_file: str = "xg_alerts.json"):
        """Generate alert configuration for monitoring systems"""
        alert_config = {
            "alerts": [
                {
                    "name": "xg_provider_unhealthy",
                    "description": "xG data provider is unhealthy",
                    "condition": "health_status.providers.*.is_healthy == false",
                    "severity": "high",
                    "notification_channels": ["email", "slack"]
                },
                {
                    "name": "xg_circuit_breaker_open",
                    "description": "Circuit breaker is open for xG provider",
                    "condition": "health_status.providers.*.circuit_breaker_state == 'open'",
                    "severity": "critical",
                    "notification_channels": ["email", "slack", "pagerduty"]
                },
                {
                    "name": "xg_low_success_rate",
                    "description": "Low success rate for xG API requests",
                    "condition": "health_status.providers.*.success_rate < 0.9",
                    "severity": "medium",
                    "notification_channels": ["email"]
                },
                {
                    "name": "xg_high_response_time",
                    "description": "High response time for xG API",
                    "condition": "health_status.providers.*.average_response_time > 5.0",
                    "severity": "medium",
                    "notification_channels": ["email"]
                },
                {
                    "name": "xg_rate_limit_exceeded",
                    "description": "Rate limit frequently exceeded",
                    "condition": "health_status.providers.*.rate_limit_hits > 10",
                    "severity": "low",
                    "notification_channels": ["email"]
                },
                {
                    "name": "xg_high_api_costs",
                    "description": "API costs are high",
                    "condition": "cost_summary.total_cost > 10.0",
                    "severity": "medium",
                    "notification_channels": ["email"]
                }
            ],
            "notification_channels": {
                "email": {
                    "type": "email",
                    "recipients": ["admin@example.com"]
                },
                "slack": {
                    "type": "slack",
                    "webhook_url": "https://hooks.slack.com/services/...",
                    "channel": "#airsenal-alerts"
                },
                "pagerduty": {
                    "type": "pagerduty",
                    "integration_key": "your-pagerduty-key"
                }
            },
            "check_interval_seconds": 300,
            "dashboard_url": "http://localhost:5000"
        }
        
        with open(output_file, 'w') as f:
            json.dump(alert_config, f, indent=2)
        
        logger.info(f"Generated alert configuration: {output_file}")
        return alert_config


def main():
    """Main function for monitoring dashboard"""
    parser = argparse.ArgumentParser(
        description="xG Data Provider Monitoring Dashboard"
    )
    
    parser.add_argument(
        "--mode",
        choices=["web", "console", "snapshot", "alerts"],
        default="console",
        help="Dashboard mode (default: console)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for web dashboard (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web dashboard (default: 5000)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for web dashboard"
    )
    
    args = parser.parse_args()
    
    try:
        dashboard = XGMonitoringDashboard()
        
        if args.mode == "web":
            if not FLASK_AVAILABLE:
                logger.error("Flask not installed. Install with: pip install flask")
                return 1
            success = dashboard.run_web_dashboard(args.host, args.port, args.debug)
            return 0 if success else 1
            
        elif args.mode == "console":
            dashboard.print_console_dashboard()
            return 0
            
        elif args.mode == "snapshot":
            dashboard.save_metrics_snapshot()
            print("Metrics snapshot saved successfully")
            return 0
            
        elif args.mode == "alerts":
            config = dashboard.generate_alert_config()
            print("Alert configuration generated successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Error running monitoring dashboard: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())