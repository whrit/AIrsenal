"""
Log analysis utilities for AIrsenal structured logs.

Provides tools for analyzing performance, errors, and patterns in log data.
"""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_config import get_log_directory


class LogAnalyzer:
    """Analyzer for structured log files."""

    def __init__(self, log_directory: Path | None = None):
        """
        Initialize log analyzer.

        Args:
            log_directory: Directory containing log files. If None, uses default.
        """
        self.log_directory = log_directory or get_log_directory()

    def _read_json_logs(
        self,
        log_file: Path,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Read JSON log entries from a file.

        Args:
            log_file: Path to log file
            start_time: Optional start time filter
            end_time: Optional end time filter

        Yields:
            Parsed log entries as dictionaries
        """
        if not log_file.exists():
            return

        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)

                    # Filter by time if specified
                    if start_time or end_time:
                        entry_time = datetime.fromisoformat(
                            entry.get("timestamp", "").replace("Z", "+00:00")
                        )
                        if start_time and entry_time < start_time:
                            continue
                        if end_time and entry_time > end_time:
                            continue

                    yield entry

                except (json.JSONDecodeError, ValueError):
                    # Skip malformed lines
                    continue

    def get_performance_metrics(self, hours_back: int = 24) -> dict[str, Any]:
        """
        Analyze performance metrics from logs.

        Args:
            hours_back: Number of hours to look back from now

        Returns:
            Dictionary containing performance analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        # Collect timing data
        prediction_times = []
        feature_times = []
        db_times = []
        api_times = []
        optimization_times = []

        for log_file in self.log_directory.glob("*.log"):
            for entry in self._read_json_logs(log_file, start_time, end_time):
                event_type = entry.get("event_type", "")
                execution_time = entry.get("execution_time")

                if execution_time is None:
                    continue

                if "prediction" in event_type:
                    prediction_times.append(execution_time)
                elif "feature" in event_type:
                    feature_times.append(execution_time)
                elif "database" in event_type:
                    db_times.append(execution_time)
                elif "api" in event_type:
                    api_times.append(execution_time)
                elif "optimization" in event_type:
                    optimization_times.append(execution_time)

        def _analyze_times(times: list[float], name: str) -> dict[str, Any]:
            if not times:
                return {f"{name}_count": 0}

            return {
                f"{name}_count": len(times),
                f"{name}_mean": sum(times) / len(times),
                f"{name}_median": sorted(times)[len(times) // 2],
                f"{name}_min": min(times),
                f"{name}_max": max(times),
                f"{name}_p95": sorted(times)[int(len(times) * 0.95)]
                if len(times) > 0
                else 0,
                f"{name}_p99": sorted(times)[int(len(times) * 0.99)]
                if len(times) > 0
                else 0,
            }

        metrics = {
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_operations": len(prediction_times)
                + len(feature_times)
                + len(db_times)
                + len(api_times)
                + len(optimization_times),
            },
        }

        metrics.update(_analyze_times(prediction_times, "prediction"))
        metrics.update(_analyze_times(feature_times, "feature"))
        metrics.update(_analyze_times(db_times, "database"))
        metrics.update(_analyze_times(api_times, "api"))
        metrics.update(_analyze_times(optimization_times, "optimization"))

        return metrics

    def get_error_analysis(self, hours_back: int = 24) -> dict[str, Any]:
        """
        Analyze errors and failures from logs.

        Args:
            hours_back: Number of hours to look back from now

        Returns:
            Dictionary containing error analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        errors = []
        error_types: Counter[str] = Counter()
        error_functions: Counter[str] = Counter()
        error_models: Counter[str] = Counter()
        correlation_errors = defaultdict(list)

        for log_file in self.log_directory.glob("*.log"):
            for entry in self._read_json_logs(log_file, start_time, end_time):
                level = entry.get("level", "").upper()
                event_type = entry.get("event_type", "")

                if level in ["ERROR", "CRITICAL"] or "error" in event_type:
                    errors.append(entry)

                    error_type = entry.get("error_type", "Unknown")
                    error_types[error_type] += 1

                    function_name = entry.get("function", "Unknown")
                    error_functions[function_name] += 1

                    model_name = entry.get("model_name")
                    if model_name:
                        error_models[model_name] += 1

                    correlation_id = entry.get("correlation_id")
                    if correlation_id:
                        correlation_errors[correlation_id].append(entry)

        # Find error patterns by correlation ID
        error_chains = {}
        for correlation_id, error_list in correlation_errors.items():
            if len(error_list) > 1:
                error_chains[correlation_id] = {
                    "count": len(error_list),
                    "time_span": self._calculate_time_span(error_list),
                    "error_types": [e.get("error_type", "Unknown") for e in error_list],
                }

        return {
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_errors": len(errors),
                "unique_error_types": len(error_types),
                "unique_functions_with_errors": len(error_functions),
                "error_chains": len(error_chains),
            },
            "top_error_types": dict(error_types.most_common(10)),
            "top_error_functions": dict(error_functions.most_common(10)),
            "top_error_models": dict(error_models.most_common(10)),
            "error_chains": error_chains,
        }

    def get_prediction_analysis(self, hours_back: int = 24) -> dict[str, Any]:
        """
        Analyze prediction operations from logs.

        Args:
            hours_back: Number of hours to look back from now

        Returns:
            Dictionary containing prediction analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        predictions = []
        model_stats: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "times": [], "errors": 0}
        )
        player_stats: defaultdict[str, int] = defaultdict(int)
        gameweek_stats: defaultdict[str, int] = defaultdict(int)

        for log_file in self.log_directory.glob("*.log"):
            for entry in self._read_json_logs(log_file, start_time, end_time):
                event_type = entry.get("event_type", "")

                if "prediction" in event_type:
                    predictions.append(entry)

                    model_name = entry.get("model_name", "Unknown")
                    execution_time = entry.get("execution_time")

                    if "error" in event_type:
                        model_stats[model_name]["errors"] += 1
                    else:
                        model_stats[model_name]["count"] += 1
                        if execution_time:
                            model_stats[model_name]["times"].append(execution_time)

                    player_id = entry.get("player_id")
                    if player_id:
                        player_stats[player_id] += 1

                    gameweek = entry.get("gameweek")
                    if gameweek:
                        gameweek_stats[gameweek] += 1

        # Calculate model performance
        model_performance = {}
        for model, stats in model_stats.items():
            times = stats["times"]
            model_performance[model] = {
                "total_predictions": stats["count"],
                "total_errors": stats["errors"],
                "success_rate": stats["count"] / (stats["count"] + stats["errors"])
                if (stats["count"] + stats["errors"]) > 0
                else 0,
                "avg_time": sum(times) / len(times) if times else 0,
                "median_time": sorted(times)[len(times) // 2] if times else 0,
            }

        return {
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_prediction_operations": len(predictions),
                "unique_models": len(model_stats),
                "unique_players": len(player_stats),
                "unique_gameweeks": len(gameweek_stats),
            },
            "model_performance": model_performance,
            "top_players_predicted": dict(Counter(player_stats).most_common(10)),
            "gameweek_distribution": dict(gameweek_stats),
        }

    def get_feature_analysis(self, hours_back: int = 24) -> dict[str, Any]:
        """
        Analyze feature operations from logs.

        Args:
            hours_back: Number of hours to look back from now

        Returns:
            Dictionary containing feature analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        feature_ops = []
        feature_stats: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "computations": 0,
                "times": [],
                "errors": 0,
                "validations_failed": 0,
            }
        )
        entity_stats: defaultdict[str, int] = defaultdict(int)

        for log_file in self.log_directory.glob("*.log"):
            for entry in self._read_json_logs(log_file, start_time, end_time):
                event_type = entry.get("event_type", "")

                if "feature" in event_type:
                    feature_ops.append(entry)

                    feature_name = entry.get("feature_name", "Unknown")
                    execution_time = entry.get("execution_time")

                    if "error" in event_type:
                        feature_stats[feature_name]["errors"] += 1
                    elif "validation" in event_type:
                        feature_stats[feature_name]["validations_failed"] += 1
                    else:
                        feature_stats[feature_name]["computations"] += 1
                        if execution_time:
                            feature_stats[feature_name]["times"].append(execution_time)

                    entity_type = entry.get("entity_type")
                    if entity_type:
                        entity_stats[entity_type] += 1

        # Calculate feature performance
        feature_performance = {}
        for feature, stats in feature_stats.items():
            times = stats["times"]
            total_ops = (
                stats["computations"] + stats["errors"] + stats["validations_failed"]
            )

            feature_performance[feature] = {
                "total_operations": total_ops,
                "successful_computations": stats["computations"],
                "errors": stats["errors"],
                "validation_failures": stats["validations_failed"],
                "success_rate": stats["computations"] / total_ops
                if total_ops > 0
                else 0,
                "avg_computation_time": sum(times) / len(times) if times else 0,
            }

        return {
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_feature_operations": len(feature_ops),
                "unique_features": len(feature_stats),
                "entity_type_distribution": dict(entity_stats),
            },
            "feature_performance": feature_performance,
        }

    def get_api_analysis(self, hours_back: int = 24) -> dict[str, Any]:
        """
        Analyze API call patterns from logs.

        Args:
            hours_back: Number of hours to look back from now

        Returns:
            Dictionary containing API analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        api_calls = []
        service_stats: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "times": []}
        )
        status_codes: Counter[str] = Counter()

        for log_file in self.log_directory.glob("*.log"):
            for entry in self._read_json_logs(log_file, start_time, end_time):
                event_type = entry.get("event_type", "")

                if "api" in event_type:
                    api_calls.append(entry)

                    service = entry.get("service", "Unknown")
                    response_time = entry.get("response_time")
                    status_code = entry.get("status_code")

                    if "error" in event_type or (status_code and status_code >= 400):
                        service_stats[service]["errors"] += 1
                    else:
                        service_stats[service]["calls"] += 1
                        if response_time:
                            service_stats[service]["times"].append(response_time)

                    if status_code:
                        status_codes[status_code] += 1

        # Calculate service performance
        service_performance = {}
        for service, stats in service_stats.items():
            times = stats["times"]
            total_calls = stats["calls"] + stats["errors"]

            service_performance[service] = {
                "total_calls": total_calls,
                "successful_calls": stats["calls"],
                "errors": stats["errors"],
                "success_rate": stats["calls"] / total_calls if total_calls > 0 else 0,
                "avg_response_time": sum(times) / len(times) if times else 0,
            }

        return {
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_api_calls": len(api_calls),
                "unique_services": len(service_stats),
            },
            "service_performance": service_performance,
            "status_code_distribution": dict(status_codes),
        }

    def _calculate_time_span(self, entries: list[dict[str, Any]]) -> float:
        """Calculate time span between first and last entry."""
        if len(entries) < 2:
            return 0.0

        timestamps = []
        for entry in entries:
            timestamp_str = entry.get("timestamp", "")
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except ValueError:
                    continue

        if len(timestamps) < 2:
            return 0.0

        return (max(timestamps) - min(timestamps)).total_seconds()

    def export_to_csv(self, output_dir: Path, hours_back: int = 24) -> dict[str, Path]:
        """
        Export analysis results to CSV files.

        Args:
            output_dir: Directory to save CSV files
            hours_back: Number of hours to analyze

        Returns:
            Dictionary mapping analysis type to file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        analyses = {
            "performance": self.get_performance_metrics(hours_back),
            "errors": self.get_error_analysis(hours_back),
            "predictions": self.get_prediction_analysis(hours_back),
            "features": self.get_feature_analysis(hours_back),
            "api": self.get_api_analysis(hours_back),
        }

        file_paths = {}

        for analysis_type, data in analyses.items():
            file_path = output_dir / f"{analysis_type}_analysis.csv"

            # Flatten the nested dictionary for CSV export
            flattened_data = self._flatten_dict(data)
            df = pd.DataFrame([flattened_data])
            df.to_csv(file_path, index=False)

            file_paths[analysis_type] = file_path

        return file_paths

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> dict[str, Any]:
        """Flatten a nested dictionary."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def analyze_correlation_chains(
    correlation_id: str, hours_back: int = 24
) -> dict[str, Any]:
    """
    Analyze all log entries for a specific correlation ID.

    Args:
        correlation_id: Correlation ID to trace
        hours_back: Number of hours to look back

    Returns:
        Dictionary containing the full trace analysis
    """
    analyzer = LogAnalyzer()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    entries = []

    for log_file in analyzer.log_directory.glob("*.log"):
        for entry in analyzer._read_json_logs(log_file, start_time, end_time):
            if entry.get("correlation_id") == correlation_id:
                entries.append(entry)

    if not entries:
        return {
            "correlation_id": correlation_id,
            "entries": [],
            "summary": {"total_entries": 0},
        }

    # Sort by timestamp
    entries.sort(key=lambda x: x.get("timestamp", ""))

    # Analyze the chain
    operations: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    functions: Counter[str] = Counter()
    total_time = 0

    for entry in entries:
        operations[entry.get("event_type", "Unknown")] += 1
        levels[entry.get("level", "Unknown")] += 1

        function = entry.get("function")
        if function:
            functions[function] += 1

        execution_time = entry.get("execution_time")
        if execution_time:
            total_time += execution_time

    return {
        "correlation_id": correlation_id,
        "entries": entries,
        "summary": {
            "total_entries": len(entries),
            "time_span": analyzer._calculate_time_span(entries),
            "total_execution_time": total_time,
            "operation_distribution": dict(operations),
            "level_distribution": dict(levels),
            "function_distribution": dict(functions),
        },
    }


def search_logs(
    pattern: str,
    hours_back: int = 24,
    log_level: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search logs for entries matching criteria.

    Args:
        pattern: Regex pattern to search for in log messages
        hours_back: Number of hours to look back
        log_level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        event_type: Filter by event type

    Returns:
        List of matching log entries
    """
    analyzer = LogAnalyzer()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    regex = re.compile(pattern, re.IGNORECASE)
    matching_entries = []

    for log_file in analyzer.log_directory.glob("*.log"):
        for entry in analyzer._read_json_logs(log_file, start_time, end_time):
            # Apply filters
            if log_level and entry.get("level", "").upper() != log_level.upper():
                continue

            if event_type and entry.get("event_type") != event_type:
                continue

            # Search in the entry content
            entry_str = json.dumps(entry)
            if regex.search(entry_str):
                matching_entries.append(entry)

    return matching_entries


# Export main functions
__all__ = [
    "LogAnalyzer",
    "analyze_correlation_chains",
    "search_logs",
]
