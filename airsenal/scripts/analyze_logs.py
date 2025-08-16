#!/usr/bin/env python3
"""
AIrsenal Log Analysis CLI Tool

Provides command-line access to log analysis functionality.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from airsenal.framework.log_analysis import LogAnalyzer, analyze_correlation_chains, search_logs


def format_json_output(data, indent=2):
    """Format data as pretty JSON."""
    return json.dumps(data, indent=indent, default=str)


def analyze_performance(hours_back: int, output_format: str = "json"):
    """Analyze performance metrics."""
    analyzer = LogAnalyzer()
    metrics = analyzer.get_performance_metrics(hours_back)
    
    if output_format == "json":
        print(format_json_output(metrics))
    else:
        print(f"Performance Analysis (Last {hours_back} hours)")
        print("=" * 50)
        
        summary = metrics.get("summary", {})
        print(f"Total Operations: {summary.get('total_operations', 0)}")
        
        # Prediction metrics
        if metrics.get("prediction_count", 0) > 0:
            print(f"\nPredictions:")
            print(f"  Count: {metrics.get('prediction_count', 0)}")
            print(f"  Mean Time: {metrics.get('prediction_mean', 0):.3f}s")
            print(f"  P95 Time: {metrics.get('prediction_p95', 0):.3f}s")
        
        # API metrics
        if metrics.get("api_count", 0) > 0:
            print(f"\nAPI Calls:")
            print(f"  Count: {metrics.get('api_count', 0)}")
            print(f"  Mean Time: {metrics.get('api_mean', 0):.3f}s")
            print(f"  P95 Time: {metrics.get('api_p95', 0):.3f}s")
        
        # Database metrics
        if metrics.get("database_count", 0) > 0:
            print(f"\nDatabase Operations:")
            print(f"  Count: {metrics.get('database_count', 0)}")
            print(f"  Mean Time: {metrics.get('database_mean', 0):.3f}s")
            print(f"  P95 Time: {metrics.get('database_p95', 0):.3f}s")


def analyze_errors(hours_back: int, output_format: str = "json"):
    """Analyze error patterns."""
    analyzer = LogAnalyzer()
    errors = analyzer.get_error_analysis(hours_back)
    
    if output_format == "json":
        print(format_json_output(errors))
    else:
        print(f"Error Analysis (Last {hours_back} hours)")
        print("=" * 50)
        
        summary = errors.get("summary", {})
        print(f"Total Errors: {summary.get('total_errors', 0)}")
        print(f"Unique Error Types: {summary.get('unique_error_types', 0)}")
        print(f"Functions with Errors: {summary.get('unique_functions_with_errors', 0)}")
        
        # Top error types
        top_errors = errors.get("top_error_types", {})
        if top_errors:
            print(f"\nTop Error Types:")
            for error_type, count in list(top_errors.items())[:5]:
                print(f"  {error_type}: {count}")
        
        # Top error functions
        top_functions = errors.get("top_error_functions", {})
        if top_functions:
            print(f"\nTop Error Functions:")
            for function, count in list(top_functions.items())[:5]:
                print(f"  {function}: {count}")


def analyze_predictions(hours_back: int, output_format: str = "json"):
    """Analyze prediction operations."""
    analyzer = LogAnalyzer()
    predictions = analyzer.get_prediction_analysis(hours_back)
    
    if output_format == "json":
        print(format_json_output(predictions))
    else:
        print(f"Prediction Analysis (Last {hours_back} hours)")
        print("=" * 50)
        
        summary = predictions.get("summary", {})
        print(f"Total Prediction Operations: {summary.get('total_prediction_operations', 0)}")
        print(f"Unique Models: {summary.get('unique_models', 0)}")
        print(f"Unique Players: {summary.get('unique_players', 0)}")
        
        # Model performance
        model_perf = predictions.get("model_performance", {})
        if model_perf:
            print(f"\nModel Performance:")
            for model, perf in model_perf.items():
                print(f"  {model}:")
                print(f"    Predictions: {perf.get('total_predictions', 0)}")
                print(f"    Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"    Avg Time: {perf.get('avg_time', 0):.3f}s")


def analyze_features(hours_back: int, output_format: str = "json"):
    """Analyze feature operations."""
    analyzer = LogAnalyzer()
    features = analyzer.get_feature_analysis(hours_back)
    
    if output_format == "json":
        print(format_json_output(features))
    else:
        print(f"Feature Analysis (Last {hours_back} hours)")
        print("=" * 50)
        
        summary = features.get("summary", {})
        print(f"Total Feature Operations: {summary.get('total_feature_operations', 0)}")
        print(f"Unique Features: {summary.get('unique_features', 0)}")
        
        # Feature performance
        feature_perf = features.get("feature_performance", {})
        if feature_perf:
            print(f"\nFeature Performance:")
            for feature, perf in list(feature_perf.items())[:10]:  # Top 10
                print(f"  {feature}:")
                print(f"    Operations: {perf.get('total_operations', 0)}")
                print(f"    Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"    Avg Time: {perf.get('avg_computation_time', 0):.3f}s")


def analyze_api_calls(hours_back: int, output_format: str = "json"):
    """Analyze API call patterns."""
    analyzer = LogAnalyzer()
    api_data = analyzer.get_api_analysis(hours_back)
    
    if output_format == "json":
        print(format_json_output(api_data))
    else:
        print(f"API Analysis (Last {hours_back} hours)")
        print("=" * 50)
        
        summary = api_data.get("summary", {})
        print(f"Total API Calls: {summary.get('total_api_calls', 0)}")
        print(f"Unique Services: {summary.get('unique_services', 0)}")
        
        # Service performance
        service_perf = api_data.get("service_performance", {})
        if service_perf:
            print(f"\nService Performance:")
            for service, perf in service_perf.items():
                print(f"  {service}:")
                print(f"    Calls: {perf.get('total_calls', 0)}")
                print(f"    Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"    Avg Response Time: {perf.get('avg_response_time', 0):.3f}s")
        
        # Status codes
        status_codes = api_data.get("status_code_distribution", {})
        if status_codes:
            print(f"\nStatus Code Distribution:")
            for code, count in sorted(status_codes.items()):
                print(f"  {code}: {count}")


def trace_correlation(correlation_id: str, hours_back: int, output_format: str = "json"):
    """Trace all operations for a correlation ID."""
    analysis = analyze_correlation_chains(correlation_id, hours_back)
    
    if output_format == "json":
        print(format_json_output(analysis))
    else:
        print(f"Correlation Trace: {correlation_id}")
        print("=" * 50)
        
        summary = analysis.get("summary", {})
        print(f"Total Entries: {summary.get('total_entries', 0)}")
        print(f"Time Span: {summary.get('time_span', 0):.1f}s")
        print(f"Total Execution Time: {summary.get('total_execution_time', 0):.3f}s")
        
        # Operation distribution
        operations = summary.get("operation_distribution", {})
        if operations:
            print(f"\nOperation Distribution:")
            for operation, count in operations.items():
                print(f"  {operation}: {count}")
        
        # Log entries
        entries = analysis.get("entries", [])
        if entries and len(entries) <= 20:  # Show entries if reasonable number
            print(f"\nLog Entries:")
            for entry in entries:
                timestamp = entry.get("timestamp", "")
                level = entry.get("level", "").upper()
                event = entry.get("event", "")
                print(f"  [{timestamp}] {level}: {event}")


def search_log_entries(pattern: str, hours_back: int, level: Optional[str], 
                      event_type: Optional[str], output_format: str = "json"):
    """Search log entries by pattern."""
    results = search_logs(
        pattern=pattern,
        hours_back=hours_back,
        log_level=level,
        event_type=event_type
    )
    
    if output_format == "json":
        print(format_json_output(results))
    else:
        print(f"Search Results for '{pattern}' (Last {hours_back} hours)")
        print("=" * 50)
        print(f"Found {len(results)} matching entries")
        
        for i, entry in enumerate(results[:20]):  # Show first 20
            timestamp = entry.get("timestamp", "")
            level = entry.get("level", "").upper()
            event = entry.get("event", "")
            print(f"\n{i+1}. [{timestamp}] {level}: {event}")
            
            # Show relevant fields
            for key, value in entry.items():
                if key not in ["timestamp", "level", "event", "logger"] and value:
                    print(f"   {key}: {value}")
        
        if len(results) > 20:
            print(f"\n... and {len(results) - 20} more entries")


def export_analysis(hours_back: int, output_dir: str):
    """Export all analyses to CSV files."""
    analyzer = LogAnalyzer()
    output_path = Path(output_dir)
    
    try:
        file_paths = analyzer.export_to_csv(output_path, hours_back)
        
        print(f"Analysis exported to {output_path}")
        print("Files created:")
        for analysis_type, file_path in file_paths.items():
            print(f"  {analysis_type}: {file_path}")
        
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze AIrsenal structured logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s performance --hours 24
  %(prog)s errors --hours 6 --format table
  %(prog)s trace abc-123-def --hours 12
  %(prog)s search "prediction.*failed" --level ERROR
  %(prog)s export ./analysis --hours 168
        """
    )
    
    parser.add_argument(
        "--hours", "-t",
        type=int,
        default=24,
        help="Hours to look back (default: 24)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Analysis commands")
    
    # Performance analysis
    perf_parser = subparsers.add_parser("performance", help="Analyze performance metrics")
    
    # Error analysis
    error_parser = subparsers.add_parser("errors", help="Analyze error patterns")
    
    # Prediction analysis
    pred_parser = subparsers.add_parser("predictions", help="Analyze prediction operations")
    
    # Feature analysis
    feature_parser = subparsers.add_parser("features", help="Analyze feature operations")
    
    # API analysis
    api_parser = subparsers.add_parser("api", help="Analyze API call patterns")
    
    # Correlation tracing
    trace_parser = subparsers.add_parser("trace", help="Trace correlation ID")
    trace_parser.add_argument("correlation_id", help="Correlation ID to trace")
    
    # Log searching
    search_parser = subparsers.add_parser("search", help="Search log entries")
    search_parser.add_argument("pattern", help="Regex pattern to search for")
    search_parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                              help="Filter by log level")
    search_parser.add_argument("--event-type", help="Filter by event type")
    
    # Export
    export_parser = subparsers.add_parser("export", help="Export analysis to CSV")
    export_parser.add_argument("output_dir", help="Output directory for CSV files")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "performance":
            analyze_performance(args.hours, args.format)
        elif args.command == "errors":
            analyze_errors(args.hours, args.format)
        elif args.command == "predictions":
            analyze_predictions(args.hours, args.format)
        elif args.command == "features":
            analyze_features(args.hours, args.format)
        elif args.command == "api":
            analyze_api_calls(args.hours, args.format)
        elif args.command == "trace":
            trace_correlation(args.correlation_id, args.hours, args.format)
        elif args.command == "search":
            search_log_entries(args.pattern, args.hours, args.level, 
                             args.event_type, args.format)
        elif args.command == "export":
            export_analysis(args.hours, args.output_dir)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()