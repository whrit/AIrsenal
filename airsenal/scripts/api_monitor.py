#!/usr/bin/env python3
"""
CLI API Monitor for AIrsenal

A command-line monitoring tool for API usage, performance, and costs.
Provides real-time metrics and alerts without requiring web dependencies.

Usage:
    python -m airsenal.scripts.api_monitor [command] [options]

Commands:
    status      Show current API status and metrics
    monitor     Continuous monitoring with real-time updates
    alerts      Show current alerts and warnings
    costs       Show cost analysis and projections
    providers   Show detailed provider information
    cache       Show cache performance metrics
    export      Export metrics to file (JSON/CSV)

Examples:
    python -m airsenal.scripts.api_monitor status
    python -m airsenal.scripts.api_monitor monitor --interval 10
    python -m airsenal.scripts.api_monitor costs --provider sportmonks
    python -m airsenal.scripts.api_monitor export --format json --output metrics.json
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from airsenal.framework.api_manager import get_api_manager
from airsenal.framework.logging_config import get_logger
from airsenal.framework.redis_cache import redis_cache

logger = get_logger(__name__)


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class APIMonitor:
    """CLI-based API monitoring tool"""

    def __init__(self):
        self.api_manager = None
        self.use_colors = sys.stdout.isatty()  # Use colors only in terminal

    async def initialize(self):
        """Initialize the API manager"""
        try:
            self.api_manager = await get_api_manager()
            logger.info("API Monitor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API manager: {e}")
            raise

    def colorize(self, text: str, color: str) -> str:
        """Add color to text if terminal supports it"""
        if self.use_colors:
            return f"{color}{text}{Colors.ENDC}"
        return text

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def format_size(self, bytes_val: float) -> str:
        """Format size in human-readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"

    async def show_status(self):
        """Show current API status and key metrics"""
        if not self.api_manager:
            await self.initialize()

        print(self.colorize("=" * 60, Colors.HEADER))
        print(self.colorize("AIrsenal API Status Dashboard", Colors.HEADER))
        print(self.colorize("=" * 60, Colors.HEADER))
        print()

        try:
            stats = self.api_manager.get_comprehensive_stats()

            # Overall health status
            self._show_health_status(stats)
            print()

            # Key metrics
            self._show_key_metrics(stats)
            print()

            # Provider summary
            self._show_provider_summary(stats)

        except Exception as e:
            print(self.colorize(f"Error retrieving status: {e}", Colors.FAIL))

    def _show_health_status(self, stats: dict):
        """Show overall system health"""
        print(self.colorize("System Health", Colors.BOLD))
        print("-" * 20)

        # Calculate overall health
        cache_hit_rate = stats.get("cache", {}).get("overall_hit_rate", 0.0)
        queue_length = stats.get("queue", {}).get("queue_length", 0)
        total_requests = stats.get("usage", {}).get("totals", {}).get("requests", 0)

        # Determine health status
        issues = []
        if cache_hit_rate < 0.6:
            issues.append("Low cache hit rate")
        if queue_length > 100:
            issues.append("High queue length")

        if not issues:
            health_status = self.colorize("HEALTHY", Colors.OKGREEN)
        elif len(issues) == 1:
            health_status = self.colorize("WARNING", Colors.WARNING)
        else:
            health_status = self.colorize("CRITICAL", Colors.FAIL)

        print(f"Overall Status: {health_status}")
        print(
            f"Redis Available: {self.colorize('Yes' if redis_cache.is_available() else 'No', Colors.OKGREEN if redis_cache.is_available() else Colors.FAIL)}"
        )
        print(f"Total Requests: {total_requests:,}")

        if issues:
            print(f"Issues: {', '.join(issues)}")

    def _show_key_metrics(self, stats: dict):
        """Show key performance metrics"""
        print(self.colorize("Key Metrics", Colors.BOLD))
        print("-" * 20)

        # Cache metrics
        cache_stats = stats.get("cache", {})
        hit_rate = cache_stats.get("overall_hit_rate", 0.0) * 100
        memory_entries = cache_stats.get("memory_entries", 0)
        memory_size = cache_stats.get("memory_size_mb", 0.0)

        hit_rate_color = (
            Colors.OKGREEN
            if hit_rate > 80
            else Colors.WARNING
            if hit_rate > 60
            else Colors.FAIL
        )
        print(f"Cache Hit Rate: {self.colorize(f'{hit_rate:.1f}%', hit_rate_color)}")
        print(f"Memory Cache: {memory_entries:,} entries, {memory_size:.1f}MB")

        # Queue metrics
        queue_stats = stats.get("queue", {})
        queue_length = queue_stats.get("queue_length", 0)
        active_requests = queue_stats.get("active_requests", 0)

        queue_color = (
            Colors.OKGREEN
            if queue_length < 10
            else Colors.WARNING
            if queue_length < 50
            else Colors.FAIL
        )
        print(
            f"Request Queue: {self.colorize(str(queue_length), queue_color)} pending, {active_requests} active"
        )

        # Cost metrics
        usage_stats = stats.get("usage", {})
        total_cost = usage_stats.get("totals", {}).get("cost", 0.0)
        print(f"Total Cost: ${total_cost:.2f}")

    def _show_provider_summary(self, stats: dict):
        """Show summary of all providers"""
        print(self.colorize("Provider Summary", Colors.BOLD))
        print("-" * 20)

        usage_stats = stats.get("usage", {}).get("providers", {})
        rate_limiter_stats = stats.get("rate_limiter", {})

        if not usage_stats:
            print("No providers configured")
            return

        # Header
        print(
            f"{'Provider':<15} {'Status':<10} {'Requests':<10} {'Success':<8} {'Cost':<10}"
        )
        print("-" * 60)

        for provider_name, provider_data in usage_stats.items():
            requests = provider_data.get("total_requests", 0)
            success_rate = provider_data.get("success_rate", 0.0) * 100
            cost = provider_data.get("total_cost", 0.0)

            # Determine status
            rate_limit_data = rate_limiter_stats.get(provider_name, {})
            can_make_request = rate_limit_data.get("can_make_request", True)

            if success_rate > 95 and can_make_request:
                status = self.colorize("OK", Colors.OKGREEN)
            elif success_rate > 80:
                status = self.colorize("WARNING", Colors.WARNING)
            else:
                status = self.colorize("ERROR", Colors.FAIL)

            print(
                f"{provider_name:<15} {status:<15} {requests:<10,} {success_rate:<7.1f}% ${cost:<9.2f}"
            )

    async def monitor_continuous(self, interval: int = 10):
        """Continuous monitoring with regular updates"""
        if not self.api_manager:
            await self.initialize()

        print(
            self.colorize(
                "Starting continuous monitoring (Ctrl+C to stop)", Colors.OKBLUE
            )
        )
        print(f"Update interval: {interval} seconds")
        print()

        try:
            while True:
                # Clear screen (platform independent)
                os.system("cls" if os.name == "nt" else "clear")

                # Show status
                await self.show_status()

                # Show timestamp
                print()
                print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Next update in {interval} seconds...")

                # Wait for next update
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

    async def show_alerts(self):
        """Show current alerts and warnings"""
        if not self.api_manager:
            await self.initialize()

        print(self.colorize("API Alerts and Warnings", Colors.HEADER))
        print("=" * 40)
        print()

        try:
            stats = self.api_manager.get_comprehensive_stats()
            alerts = self._generate_alerts(stats)

            if not alerts:
                print(self.colorize("No alerts - system is healthy! ✓", Colors.OKGREEN))
                return

            for alert in alerts:
                if alert["level"] == "warning":
                    color = Colors.WARNING
                    icon = "⚠️"
                else:
                    color = Colors.FAIL
                    icon = "🚨"

                print(f"{icon} {self.colorize(alert['message'], color)}")

        except Exception as e:
            print(self.colorize(f"Error retrieving alerts: {e}", Colors.FAIL))

    def _generate_alerts(self, stats: dict) -> list[dict]:
        """Generate alerts based on current metrics"""
        alerts = []

        # Cache performance alerts
        cache_hit_rate = stats.get("cache", {}).get("overall_hit_rate", 0.0)
        if cache_hit_rate < 0.6:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Cache hit rate is critically low ({cache_hit_rate:.1%})",
                }
            )
        elif cache_hit_rate < 0.8:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Cache hit rate is below optimal ({cache_hit_rate:.1%})",
                }
            )

        # Queue length alerts
        queue_length = stats.get("queue", {}).get("queue_length", 0)
        if queue_length > 100:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Request queue is very long ({queue_length} requests)",
                }
            )
        elif queue_length > 50:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Request queue is getting long ({queue_length} requests)",
                }
            )

        # Provider health alerts
        usage_stats = stats.get("usage", {}).get("providers", {})
        for provider_name, provider_data in usage_stats.items():
            success_rate = provider_data.get("success_rate", 0.0)
            if success_rate < 0.8:
                alerts.append(
                    {
                        "level": "critical",
                        "message": f"Provider {provider_name} has low success rate ({success_rate:.1%})",
                    }
                )

        # Cost alerts
        cost_alerts = stats.get("cost_alerts", [])
        for cost_alert in cost_alerts:
            alerts.append(
                {
                    "level": "warning",
                    "message": cost_alert.get("message", "Cost threshold exceeded"),
                }
            )

        return alerts

    async def show_costs(self, provider: str | None = None):
        """Show cost analysis and projections"""
        if not self.api_manager:
            await self.initialize()

        print(self.colorize("Cost Analysis", Colors.HEADER))
        print("=" * 40)
        print()

        try:
            stats = self.api_manager.get_comprehensive_stats()
            usage_stats = stats.get("usage", {})

            if provider:
                # Show specific provider costs
                provider_data = usage_stats.get("providers", {}).get(provider)
                if not provider_data:
                    print(f"Provider '{provider}' not found")
                    return

                self._show_provider_costs(provider, provider_data)
            else:
                # Show all provider costs
                self._show_all_costs(usage_stats)

        except Exception as e:
            print(self.colorize(f"Error retrieving cost data: {e}", Colors.FAIL))

    def _show_provider_costs(self, provider: str, data: dict):
        """Show costs for a specific provider"""
        print(f"Provider: {self.colorize(provider, Colors.BOLD)}")
        print()

        total_cost = data.get("total_cost", 0.0)
        total_requests = data.get("total_requests", 0)
        cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0

        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Total Requests: {total_requests:,}")
        print(f"Cost per Request: ${cost_per_request:.4f}")

        # Projections
        hourly_data = data.get("hourly_data", [])
        if hourly_data and len(hourly_data) > 0:
            recent_hour = hourly_data[-1]
            hourly_cost = recent_hour.get("cost", 0.0)

            daily_projection = hourly_cost * 24
            monthly_projection = daily_projection * 30

            print()
            print("Projections (based on last hour):")
            print(f"Daily: ${daily_projection:.2f}")
            print(f"Monthly: ${monthly_projection:.2f}")

    def _show_all_costs(self, usage_stats: dict):
        """Show costs for all providers"""
        totals = usage_stats.get("totals", {})
        providers = usage_stats.get("providers", {})

        cost_text = f"${totals.get('cost', 0.0):.2f}"
        print(f"Total Cost: {self.colorize(cost_text, Colors.BOLD)}")
        print(f"Total Requests: {totals.get('requests', 0):,}")
        print()

        if providers:
            print("Cost by Provider:")
            print(f"{'Provider':<15} {'Cost':<10} {'Requests':<10} {'Avg Cost':<10}")
            print("-" * 50)

            for provider_name, provider_data in providers.items():
                cost = provider_data.get("total_cost", 0.0)
                requests = provider_data.get("total_requests", 0)
                avg_cost = cost / requests if requests > 0 else 0.0

                print(
                    f"{provider_name:<15} ${cost:<9.2f} {requests:<10,} ${avg_cost:<9.4f}"
                )

    async def show_cache_stats(self):
        """Show detailed cache performance metrics"""
        if not self.api_manager:
            await self.initialize()

        print(self.colorize("Cache Performance", Colors.HEADER))
        print("=" * 40)
        print()

        try:
            stats = self.api_manager.get_comprehensive_stats()
            cache_stats = stats.get("cache", {})

            # Hit rates
            print(self.colorize("Hit Rates", Colors.BOLD))
            print("-" * 15)
            memory_hit_rate = cache_stats.get("memory_hit_rate", 0.0) * 100
            redis_hit_rate = cache_stats.get("redis_hit_rate", 0.0) * 100
            overall_hit_rate = cache_stats.get("overall_hit_rate", 0.0) * 100

            print(f"Memory Cache: {memory_hit_rate:.1f}%")
            print(f"Redis Cache: {redis_hit_rate:.1f}%")
            print(
                f"Overall: {self.colorize(f'{overall_hit_rate:.1f}%', Colors.OKGREEN if overall_hit_rate > 80 else Colors.WARNING)}"
            )
            print()

            # Cache sizes
            print(self.colorize("Cache Sizes", Colors.BOLD))
            print("-" * 15)
            memory_entries = cache_stats.get("memory_entries", 0)
            memory_size_mb = cache_stats.get("memory_size_mb", 0.0)

            print(f"Memory Entries: {memory_entries:,}")
            print(f"Memory Size: {memory_size_mb:.1f} MB")
            print()

            # Performance metrics
            print(self.colorize("Performance", Colors.BOLD))
            print("-" * 15)
            evictions = cache_stats.get("evictions", 0)
            total_requests = cache_stats.get("total_requests", 0)

            print(f"Total Requests: {total_requests:,}")
            print(f"Evictions: {evictions:,}")

            if total_requests > 0:
                eviction_rate = evictions / total_requests * 100
                print(f"Eviction Rate: {eviction_rate:.2f}%")

        except Exception as e:
            print(self.colorize(f"Error retrieving cache stats: {e}", Colors.FAIL))

    async def export_metrics(
        self, format_type: str = "json", output_file: str | None = None
    ):
        """Export metrics to file"""
        if not self.api_manager:
            await self.initialize()

        try:
            stats = self.api_manager.get_comprehensive_stats()

            # Add timestamp
            stats["export_timestamp"] = datetime.now().isoformat()

            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"api_metrics_{timestamp}.{format_type}"

            if format_type.lower() == "json":
                with open(output_file, "w") as f:
                    json.dump(stats, f, indent=2, default=str)

            elif format_type.lower() == "csv":
                # Flatten stats for CSV export
                flattened = self._flatten_dict(stats)

                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    for key, value in flattened.items():
                        writer.writerow([key, value])

            else:
                msg = f"Unsupported format: {format_type}"
                raise ValueError(msg)

            print(f"Metrics exported to: {self.colorize(output_file, Colors.OKGREEN)}")

        except Exception as e:
            print(self.colorize(f"Error exporting metrics: {e}", Colors.FAIL))

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="AIrsenal API Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    subparsers.add_parser("status", help="Show current API status and metrics")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuous monitoring")
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds (default: 10)",
    )

    # Alerts command
    subparsers.add_parser("alerts", help="Show current alerts and warnings")

    # Costs command
    costs_parser = subparsers.add_parser("costs", help="Show cost analysis")
    costs_parser.add_argument(
        "--provider", type=str, help="Show costs for specific provider"
    )

    # Providers command
    subparsers.add_parser("providers", help="Show detailed provider information")

    # Cache command
    subparsers.add_parser("cache", help="Show cache performance metrics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics to file")
    export_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    monitor = APIMonitor()

    try:
        if args.command == "status":
            await monitor.show_status()

        elif args.command == "monitor":
            await monitor.monitor_continuous(args.interval)

        elif args.command == "alerts":
            await monitor.show_alerts()

        elif args.command == "costs":
            await monitor.show_costs(getattr(args, "provider", None))

        elif args.command == "providers":
            await monitor.show_status()  # Providers are shown in status

        elif args.command == "cache":
            await monitor.show_cache_stats()

        elif args.command == "export":
            await monitor.export_metrics(args.format, getattr(args, "output", None))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
