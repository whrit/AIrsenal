#!/usr/bin/env python
"""
Historical xG/xA Data Backfilling Script for AIrsenal

This script provides comprehensive backfilling capabilities for historical xG/xA data
from external providers like Sportmonks. Supports season-wide backfills, specific
gameweeks, incremental updates, and resume functionality.

Usage:
    # Backfill complete seasons
    airsenal_backfill_xg_data --seasons 2122 2223 2324

    # Backfill specific gameweeks
    airsenal_backfill_xg_data --season 2324 --gameweeks 1 2 3

    # Incremental backfill (last 7 days)
    airsenal_backfill_xg_data --season 2324 --incremental

    # Resume interrupted backfill
    airsenal_backfill_xg_data --resume season_2324_1641234567

    # Generate missing data report
    airsenal_backfill_xg_data --season 2324 --missing-data-report

    # Validation only
    airsenal_backfill_xg_data --season 2324 --validate-only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from airsenal.framework.data_backfill import (
    HistoricalDataBackfiller, BackfillStatus, ValidationSeverity
)
from airsenal.framework.env import AIRSENAL_HOME
from airsenal.framework.logging_config import get_logger
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.xg_data_provider import create_xg_manager

logger = get_logger(__name__)


def main():
    """Main function for xG data backfilling"""
    parser = argparse.ArgumentParser(
        description="Backfill historical xG/xA data from external providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main operation modes
    operation_group = parser.add_mutually_exclusive_group()
    operation_group.add_argument(
        "--seasons",
        nargs="+",
        help="Seasons to backfill (e.g., 2122 2223 2324)"
    )
    
    operation_group.add_argument(
        "--resume",
        type=str,
        help="Resume backfill from session ID"
    )
    
    operation_group.add_argument(
        "--incremental",
        action="store_true",
        help="Perform incremental backfill for recent data"
    )
    
    operation_group.add_argument(
        "--missing-data-report",
        action="store_true",
        help="Generate missing data report without backfilling"
    )
    
    operation_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only without backfilling"
    )
    
    # Season and gameweek selection
    parser.add_argument(
        "--season",
        type=str,
        default=CURRENT_SEASON,
        help=f"Specific season for operations (default: {CURRENT_SEASON})"
    )
    
    parser.add_argument(
        "--gameweeks",
        nargs="+",
        type=int,
        help="Specific gameweeks to backfill"
    )
    
    # Configuration options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts for failed operations (default: 3)"
    )
    
    parser.add_argument(
        "--incremental-days",
        type=int,
        default=7,
        help="Number of days back for incremental backfill (default: 7)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(AIRSENAL_HOME) / "backfill_reports"),
        help="Directory for output reports"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force backfill even if data already exists"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging level
        if args.verbose:
            logger.setLevel("DEBUG")
        
        # Initialize backfiller
        xg_manager = create_xg_manager()
        backfiller = HistoricalDataBackfiller(xg_manager)
        
        # Configure backfiller
        backfiller.configure(
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            retry_attempts=args.retry_attempts
        )
        
        if not xg_manager.providers:
            logger.error("No xG data providers available. Check API key configuration.")
            print("\n❌ No xG data providers configured!")
            print("Please ensure you have a valid API key for Sportmonks or other providers.")
            return 1
        
        # Execute the requested operation
        if args.seasons:
            return handle_seasons_backfill(backfiller, args.seasons, args, output_dir)
        elif args.resume:
            return handle_resume_backfill(backfiller, args.resume, args, output_dir)
        elif args.incremental:
            return handle_incremental_backfill(backfiller, args.season, args, output_dir)
        elif args.missing_data_report:
            return handle_missing_data_report(backfiller, args.season, output_dir)
        elif args.validate_only:
            return handle_validation_only(backfiller, args.season, args.gameweeks, output_dir)
        elif args.gameweeks:
            return handle_gameweeks_backfill(backfiller, args.season, args.gameweeks, args, output_dir)
        else:
            # Default: backfill current season
            return handle_seasons_backfill(backfiller, [args.season], args, output_dir)
        
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user")
        print("\n⏸️  Backfill interrupted. You can resume using --resume with the session ID.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Backfill failed with exception: {e}")
        print(f"\n❌ Backfill failed: {e}")
        return 1


def handle_seasons_backfill(backfiller: HistoricalDataBackfiller, seasons: List[str], 
                           args, output_dir: Path) -> int:
    """Handle complete season backfills"""
    logger.info(f"Starting backfill for seasons: {seasons}")
    print(f"\n🚀 Starting backfill for seasons: {', '.join(seasons)}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        for season in seasons:
            print(f"  Would backfill season {season}")
        return 0
    
    results = {}
    total_seasons = len(seasons)
    
    for i, season in enumerate(seasons, 1):
        print(f"\n📅 Processing season {season} ({i}/{total_seasons})")
        
        start_time = time.time()
        result = backfiller.backfill_season(season)
        duration = time.time() - start_time
        
        results[season] = result
        
        # Print progress
        status_emoji = "✅" if result.status == BackfillStatus.COMPLETED else "❌"
        print(f"{status_emoji} Season {season}: {result.status.value} "
              f"({result.progress.completed_items}/{result.progress.total_items} gameweeks, "
              f"{duration:.1f}s)")
        
        if result.validation_issues:
            critical_count = len([i for i in result.validation_issues if i.severity == ValidationSeverity.CRITICAL])
            warning_count = len([i for i in result.validation_issues if i.severity == ValidationSeverity.WARNING])
            
            if critical_count > 0:
                print(f"  ⚠️  {critical_count} critical issues found")
            if warning_count > 0:
                print(f"  ⚡ {warning_count} warnings found")
    
    # Generate summary report
    generate_summary_report(results, output_dir, "seasons_backfill")
    
    # Determine overall success
    failed_seasons = [s for s, r in results.items() if r.status == BackfillStatus.FAILED]
    
    if failed_seasons:
        print(f"\n❌ {len(failed_seasons)} seasons failed: {', '.join(failed_seasons)}")
        return 1
    else:
        print(f"\n✅ All {total_seasons} seasons completed successfully!")
        return 0


def handle_gameweeks_backfill(backfiller: HistoricalDataBackfiller, season: str, 
                             gameweeks: List[int], args, output_dir: Path) -> int:
    """Handle specific gameweeks backfill"""
    logger.info(f"Starting backfill for {season} gameweeks: {gameweeks}")
    print(f"\n🎯 Backfilling {season} gameweeks: {', '.join(map(str, gameweeks))}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print(f"  Would backfill {len(gameweeks)} gameweeks")
        return 0
    
    start_time = time.time()
    result = backfiller.backfill_gameweeks(season, gameweeks)
    duration = time.time() - start_time
    
    # Print results
    status_emoji = "✅" if result.status == BackfillStatus.COMPLETED else "❌"
    print(f"{status_emoji} Gameweeks backfill: {result.status.value} "
          f"({result.progress.completed_items}/{result.progress.total_items} gameweeks, "
          f"{duration:.1f}s)")
    
    # Save detailed report
    report_path = output_dir / f"gameweeks_backfill_{season}_{int(time.time())}.json"
    save_result_report(result, report_path, {"season": season, "gameweeks": gameweeks})
    print(f"📄 Detailed report saved to: {report_path}")
    
    return 0 if result.status == BackfillStatus.COMPLETED else 1


def handle_resume_backfill(backfiller: HistoricalDataBackfiller, session_id: str, 
                          args, output_dir: Path) -> int:
    """Handle resume backfill from session ID"""
    logger.info(f"Resuming backfill session: {session_id}")
    print(f"\n🔄 Resuming backfill session: {session_id}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Would resume backfill session")
        return 0
    
    # Parse session ID to determine operation type
    if session_id.startswith("season_"):
        parts = session_id.split("_")
        if len(parts) >= 3:
            season = parts[1]
            print(f"📅 Resuming season {season} backfill...")
            
            start_time = time.time()
            result = backfiller.backfill_season(season, resume_session_id=session_id)
            duration = time.time() - start_time
            
            status_emoji = "✅" if result.status == BackfillStatus.COMPLETED else "❌"
            print(f"{status_emoji} Resumed backfill: {result.status.value} ({duration:.1f}s)")
            
            # Save report
            report_path = output_dir / f"resumed_backfill_{session_id}.json"
            save_result_report(result, report_path, {"session_id": session_id, "season": season})
            
            return 0 if result.status == BackfillStatus.COMPLETED else 1
    
    print(f"❌ Unable to parse session ID: {session_id}")
    return 1


def handle_incremental_backfill(backfiller: HistoricalDataBackfiller, season: str, 
                               args, output_dir: Path) -> int:
    """Handle incremental backfill"""
    logger.info(f"Starting incremental backfill for {season}")
    print(f"\n⚡ Incremental backfill for {season} (last {args.incremental_days} days)")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Would perform incremental backfill")
        return 0
    
    start_time = time.time()
    result = backfiller.incremental_backfill(season, days_back=args.incremental_days)
    duration = time.time() - start_time
    
    status_emoji = "✅" if result.status == BackfillStatus.COMPLETED else "❌"
    print(f"{status_emoji} Incremental backfill: {result.status.value} ({duration:.1f}s)")
    
    # Save report
    report_path = output_dir / f"incremental_backfill_{season}_{int(time.time())}.json"
    save_result_report(result, report_path, {
        "season": season, 
        "incremental_days": args.incremental_days
    })
    
    return 0 if result.status == BackfillStatus.COMPLETED else 1


def handle_missing_data_report(backfiller: HistoricalDataBackfiller, season: str, 
                              output_dir: Path) -> int:
    """Generate missing data report"""
    logger.info(f"Generating missing data report for {season}")
    print(f"\n📊 Generating missing data report for {season}")
    
    report = backfiller.get_missing_data_report(season)
    
    # Save report
    report_path = output_dir / f"missing_data_report_{season}_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Missing data report saved to: {report_path}")
    
    # Print summary
    summary = report["summary"]
    print(f"\n📈 Summary for {season}:")
    print(f"  Overall xG coverage: {summary['overall_coverage']:.1f}%")
    print(f"  Total issues found: {summary['total_issues']}")
    print(f"  Critical issues: {summary['critical_issues']}")
    print(f"  High priority gaps: {summary['high_priority_gaps']}")
    
    # Show top issues
    if report["missing_data_issues"]:
        print("\n🔍 Top issues:")
        for issue in report["missing_data_issues"][:5]:
            severity_emoji = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            emoji = severity_emoji.get(issue["severity"], "")
            print(f"  {emoji} {issue['description']}")
    
    return 0


def handle_validation_only(backfiller: HistoricalDataBackfiller, season: str, 
                          gameweeks: Optional[List[int]], output_dir: Path) -> int:
    """Run validation only"""
    logger.info(f"Running validation for {season}")
    print(f"\n🔍 Running validation for {season}")
    
    if gameweeks:
        print(f"  Gameweeks: {', '.join(map(str, gameweeks))}")
    
    validation_issues = backfiller.validator.validate_backfilled_data(season, gameweeks)
    
    # Categorize issues
    issues_by_severity = {}
    for issue in validation_issues:
        if issue.severity not in issues_by_severity:
            issues_by_severity[issue.severity] = []
        issues_by_severity[issue.severity].append(issue)
    
    # Print summary
    print(f"\n📊 Validation Results:")
    print(f"  Total issues: {len(validation_issues)}")
    
    for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                    ValidationSeverity.WARNING, ValidationSeverity.INFO]:
        count = len(issues_by_severity.get(severity, []))
        if count > 0:
            severity_emoji = {"CRITICAL": "🚨", "ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
            emoji = severity_emoji.get(severity.name, "")
            print(f"  {emoji} {severity.name}: {count}")
    
    # Save validation report
    validation_report = {
        "season": season,
        "gameweeks": gameweeks,
        "timestamp": datetime.now().isoformat(),
        "total_issues": len(validation_issues),
        "issues_by_severity": {
            severity.name: [
                {
                    "category": issue.category,
                    "description": issue.description,
                    "affected_entity": issue.affected_entity,
                    "entity_id": issue.entity_id,
                    "suggested_action": issue.suggested_action,
                    "metadata": issue.metadata
                }
                for issue in issues
            ]
            for severity, issues in issues_by_severity.items()
        }
    }
    
    report_path = output_dir / f"validation_report_{season}_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"📄 Validation report saved to: {report_path}")
    
    # Show critical issues
    critical_issues = issues_by_severity.get(ValidationSeverity.CRITICAL, [])
    if critical_issues:
        print("\n🚨 Critical Issues:")
        for issue in critical_issues[:5]:
            print(f"  • {issue.description}")
    
    return 1 if critical_issues else 0


def generate_summary_report(results: Dict[str, any], output_dir: Path, operation_type: str):
    """Generate summary report for multiple operations"""
    summary = {
        "operation_type": operation_type,
        "timestamp": datetime.now().isoformat(),
        "total_operations": len(results),
        "successful_operations": len([r for r in results.values() if r.status == BackfillStatus.COMPLETED]),
        "failed_operations": len([r for r in results.values() if r.status == BackfillStatus.FAILED]),
        "results": {}
    }
    
    for key, result in results.items():
        summary["results"][key] = {
            "status": result.status.value,
            "completed_items": result.progress.completed_items,
            "total_items": result.progress.total_items,
            "progress_percentage": result.progress.progress_percentage,
            "validation_issues_count": len(result.validation_issues),
            "critical_issues_count": len([i for i in result.validation_issues if i.severity == ValidationSeverity.CRITICAL])
        }
    
    report_path = output_dir / f"summary_{operation_type}_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary report saved to: {report_path}")


def save_result_report(result, report_path: Path, metadata: Dict = None):
    """Save detailed result report"""
    report = {
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "status": result.status.value,
        "progress": {
            "total_items": result.progress.total_items,
            "completed_items": result.progress.completed_items,
            "failed_items": result.progress.failed_items,
            "skipped_items": result.progress.skipped_items,
            "progress_percentage": result.progress.progress_percentage,
            "success_rate": result.progress.success_rate,
        },
        "validation_issues": [
            {
                "severity": issue.severity.value,
                "category": issue.category,
                "description": issue.description,
                "affected_entity": issue.affected_entity,
                "entity_id": issue.entity_id,
                "suggested_action": issue.suggested_action,
                "metadata": issue.metadata
            }
            for issue in result.validation_issues
        ],
        "reconciliation_report": result.reconciliation_report,
        "performance_metrics": result.performance_metrics
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())