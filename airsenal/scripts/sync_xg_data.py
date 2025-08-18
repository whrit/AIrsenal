#!/usr/bin/env python
"""
Sync xG/xA data from external providers into AIrsenal database.

This script provides a command-line interface for syncing xG/xA data
from providers like Sportmonks into the AIrsenal PlayerScore table.
"""

import argparse
import sys

from airsenal.framework.logging_config import get_logger
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.xg_data_provider import create_xg_manager

logger = get_logger(__name__)


def main():
    """Main function for xG data synchronization"""
    parser = argparse.ArgumentParser(
        description="Sync xG/xA data from external providers"
    )

    parser.add_argument(
        "--season",
        type=str,
        default=CURRENT_SEASON,
        help=f"Season to sync data for (default: {CURRENT_SEASON})",
    )

    parser.add_argument(
        "--gameweek",
        type=int,
        help="Specific gameweek to sync (if not provided, syncs all recent gameweeks)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="sportmonks",
        choices=["sportmonks"],
        help="xG data provider to use (default: sportmonks)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes",
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check health status of xG data providers",
    )

    parser.add_argument(
        "--cost-summary", action="store_true", help="Show API cost summary"
    )

    args = parser.parse_args()

    try:
        # Initialize xG data manager
        manager = create_xg_manager()

        if args.health_check:
            dashboard = manager.get_health_dashboard()
            print_health_status(dashboard)
            return 0

        if args.cost_summary:
            dashboard = manager.get_health_dashboard()
            print_cost_summary(dashboard["cost_summary"])
            return 0

        if not manager.providers:
            logger.error("No xG data providers available. Check API key configuration.")
            return 1

        if args.gameweek:
            # Sync specific gameweek
            logger.info(f"Syncing xG data for {args.season} GW{args.gameweek}")

            if args.dry_run:
                logger.info("DRY RUN: Would sync xG data for specific gameweek")
                return 0

            updated_count = manager.sync_xg_data_for_gameweek(
                args.season, args.gameweek
            )
            logger.info(f"Updated {updated_count} player scores with xG data")

        else:
            logger.error(
                "Please specify --gameweek or use --health-check/--cost-summary"
            )
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error syncing xG data: {e}")
        return 1


def print_health_status(dashboard):
    """Print formatted health status"""
    print("\n=== xG Data Provider Health Status ===")

    health = dashboard["health_status"]
    print(f"Last updated: {health['timestamp']}")

    for provider_name, status in health["providers"].items():
        print(f"\n{provider_name.upper()} Provider:")
        print(f"  Status: {'🟢 Healthy' if status['is_healthy'] else '🔴 Unhealthy'}")
        print(f"  Circuit Breaker: {status['circuit_breaker_state']}")
        print(f"  Success Rate: {status['success_rate']:.1%}")
        print(f"  Total Requests: {status['total_requests']}")
        print(f"  Failed Requests: {status['failed_requests']}")
        print(f"  Average Response Time: {status['average_response_time']:.2f}s")
        print(f"  Rate Limit Hits: {status['rate_limit_hits']}")
        print(f"  Tier: {status['tier']}")

    # Show recommendations
    recommendations = dashboard.get("recommendations", [])
    if recommendations:
        print("\n=== Recommendations ===")
        for rec in recommendations:
            print(f"  {rec['provider']} - {rec['type'].title()}: {rec['message']}")


def print_cost_summary(cost_summary):
    """Print formatted cost summary"""
    print("\n=== API Cost Summary ===")
    print(f"Total Cost: ${cost_summary['total_cost']:.4f}")
    print(f"Total Requests: {cost_summary['total_requests']}")

    if cost_summary["total_requests"] > 0:
        print(
            f"Average Cost per Request: ${cost_summary['average_cost_per_request']:.4f}"
        )

    print("\n--- Provider Breakdown ---")
    for provider, costs in cost_summary["provider_breakdown"].items():
        print(f"{provider}:")
        print(f"  Total Cost: ${costs['total_cost']:.4f}")
        print(f"  Requests: {costs['requests']}")
        print(f"  Cost per Request: ${costs['cost_per_request']:.4f}")
        print(f"  Estimated Monthly: ${costs['estimated_monthly_cost']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
