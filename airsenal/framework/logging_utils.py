"""
Domain-specific logging utilities for AIrsenal.

Provides specialized decorators and logging helpers for ML operations,
predictions, database queries, API calls, and feature engineering.
"""

import functools
import os
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine

from .logging_config import get_logger, set_context


class PredictionLogger:
    """Logger specialized for ML prediction operations."""

    def __init__(self, logger_name: str = "airsenal.predictions"):
        self.logger = get_logger(logger_name)

    def log_prediction_start(
        self,
        model_name: str,
        player_ids: list[int],
        gameweek_range: tuple,
        features: dict[str, Any] | None = None,
    ) -> None:
        """Log the start of a prediction operation."""
        self.logger.info(
            "Prediction started",
            event_type="prediction_start",
            model_name=model_name,
            player_count=len(player_ids),
            gameweek_range=gameweek_range,
            features=list(features.keys()) if features else None,
        )

    def log_prediction_result(
        self,
        model_name: str,
        player_id: int,
        gameweek: int,
        predicted_points: float,
        confidence: float | None = None,
        features_used: list[str] | None = None,
        execution_time: float | None = None,
    ) -> None:
        """Log individual prediction results."""
        log_data = {
            "event_type": "prediction_result",
            "model_name": model_name,
            "player_id": player_id,
            "gameweek": gameweek,
            "predicted_points": predicted_points,
        }

        if confidence is not None:
            log_data["confidence"] = confidence
        if features_used:
            log_data["features_used"] = features_used
        if execution_time is not None:
            log_data["execution_time"] = execution_time

        self.logger.info("Prediction completed", **log_data)

    def log_prediction_batch_summary(
        self,
        model_name: str,
        total_predictions: int,
        successful_predictions: int,
        failed_predictions: int,
        total_time: float,
        avg_prediction_time: float,
    ) -> None:
        """Log summary of batch prediction operation."""
        self.logger.info(
            "Prediction batch completed",
            event_type="prediction_batch_summary",
            model_name=model_name,
            total_predictions=total_predictions,
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            success_rate=successful_predictions / total_predictions
            if total_predictions > 0
            else 0,
            total_time=total_time,
            avg_prediction_time=avg_prediction_time,
        )

    def log_model_performance(
        self,
        model_name: str,
        metrics: dict[str, float],
        validation_period: tuple,
        player_count: int,
    ) -> None:
        """Log model performance metrics."""
        self.logger.info(
            "Model performance evaluation",
            event_type="model_performance",
            model_name=model_name,
            metrics=metrics,
            validation_period=validation_period,
            player_count=player_count,
        )


class FeatureLogger:
    """Logger specialized for feature engineering operations."""

    def __init__(self, logger_name: str = "airsenal.features"):
        self.logger = get_logger(logger_name)

    def log_feature_computation(
        self,
        feature_name: str,
        entity_type: str,
        entity_id: int,
        value: Any,
        computation_time: float,
        data_sources: list[str] | None = None,
    ) -> None:
        """Log feature computation."""
        self.logger.info(
            "Feature computed",
            event_type="feature_computation",
            feature_name=feature_name,
            entity_type=entity_type,
            entity_id=entity_id,
            value=value,
            computation_time=computation_time,
            data_sources=data_sources,
        )

    def log_feature_validation_error(
        self,
        feature_name: str,
        entity_type: str,
        entity_id: int,
        expected_type: str,
        actual_type: str,
        value: Any,
    ) -> None:
        """Log feature validation errors."""
        self.logger.warning(
            "Feature validation failed",
            event_type="feature_validation_error",
            feature_name=feature_name,
            entity_type=entity_type,
            entity_id=entity_id,
            expected_type=expected_type,
            actual_type=actual_type,
            value=value,
        )

    def log_feature_store_operation(
        self,
        operation: str,
        feature_names: list[str],
        entity_count: int,
        cache_hits: int,
        cache_misses: int,
        execution_time: float,
    ) -> None:
        """Log feature store operations."""
        self.logger.info(
            "Feature store operation",
            event_type="feature_store_operation",
            operation=operation,
            feature_names=feature_names,
            entity_count=entity_count,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=cache_hits / (cache_hits + cache_misses)
            if (cache_hits + cache_misses) > 0
            else 0,
            execution_time=execution_time,
        )


class DatabaseLogger:
    """Logger for database operations."""

    def __init__(self, logger_name: str = "airsenal.database"):
        self.logger = get_logger(logger_name)
        self._setup_sqlalchemy_logging()

    def _setup_sqlalchemy_logging(self) -> None:
        """Set up SQLAlchemy event listeners for query logging."""

        @event.listens_for(Engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            context._query_start_time = time.time()

        @event.listens_for(Engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            total = time.time() - context._query_start_time

            # Only log slow queries by default (> 1 second)
            threshold = float(os.getenv("AIRSENAL_SLOW_QUERY_THRESHOLD", "1.0"))
            if total > threshold:
                self.logger.warning(
                    "Slow query detected",
                    event_type="slow_query",
                    execution_time=total,
                    statement=statement[:500] + "..."
                    if len(statement) > 500
                    else statement,
                    parameters=str(parameters)[:200] + "..."
                    if len(str(parameters)) > 200
                    else str(parameters),
                )

    def log_query(
        self,
        operation: str,
        table: str,
        execution_time: float,
        row_count: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Log database query operations."""
        log_data = {
            "event_type": "database_query",
            "operation": operation,
            "table": table,
            "execution_time": execution_time,
        }

        if row_count is not None:
            log_data["row_count"] = row_count
        if filters:
            log_data["filters"] = filters

        self.logger.info("Database query executed", **log_data)

    def log_bulk_operation(
        self,
        operation: str,
        table: str,
        record_count: int,
        execution_time: float,
        batch_size: int | None = None,
    ) -> None:
        """Log bulk database operations."""
        self.logger.info(
            "Bulk database operation",
            event_type="database_bulk_operation",
            operation=operation,
            table=table,
            record_count=record_count,
            execution_time=execution_time,
            batch_size=batch_size,
        )


class APILogger:
    """Logger for external API calls."""

    def __init__(self, logger_name: str = "airsenal.api"):
        self.logger = get_logger(logger_name)

    def log_api_call(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_size: int | None = None,
        response_size: int | None = None,
        rate_limit_remaining: int | None = None,
    ) -> None:
        """Log external API calls."""
        log_data = {
            "event_type": "api_call",
            "service": service,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
        }

        if request_size is not None:
            log_data["request_size"] = request_size
        if response_size is not None:
            log_data["response_size"] = response_size
        if rate_limit_remaining is not None:
            log_data["rate_limit_remaining"] = rate_limit_remaining

        level = "error" if status_code >= 400 else "info"
        getattr(self.logger, level)("API call completed", **log_data)

    def log_api_error(
        self,
        service: str,
        endpoint: str,
        error: str,
        retry_count: int,
        will_retry: bool,
    ) -> None:
        """Log API errors and retry attempts."""
        self.logger.error(
            "API call failed",
            event_type="api_error",
            service=service,
            endpoint=endpoint,
            error=error,
            retry_count=retry_count,
            will_retry=will_retry,
        )


class OptimizationLogger:
    """Logger for optimization operations."""

    def __init__(self, logger_name: str = "airsenal.optimization"):
        self.logger = get_logger(logger_name)

    def log_optimization_start(
        self,
        optimization_type: str,
        parameters: dict[str, Any],
        constraints: dict[str, Any],
    ) -> None:
        """Log the start of an optimization operation."""
        self.logger.info(
            "Optimization started",
            event_type="optimization_start",
            optimization_type=optimization_type,
            parameters=parameters,
            constraints=constraints,
        )

    def log_optimization_progress(
        self,
        optimization_type: str,
        generation: int,
        best_score: float,
        population_size: int,
        convergence_metric: float | None = None,
    ) -> None:
        """Log optimization progress."""
        log_data = {
            "event_type": "optimization_progress",
            "optimization_type": optimization_type,
            "generation": generation,
            "best_score": best_score,
            "population_size": population_size,
        }

        if convergence_metric is not None:
            log_data["convergence_metric"] = convergence_metric

        self.logger.info("Optimization progress", **log_data)

    def log_optimization_result(
        self,
        optimization_type: str,
        final_score: float,
        total_generations: int,
        execution_time: float,
        solution: dict[str, Any],
    ) -> None:
        """Log optimization results."""
        self.logger.info(
            "Optimization completed",
            event_type="optimization_result",
            optimization_type=optimization_type,
            final_score=final_score,
            total_generations=total_generations,
            execution_time=execution_time,
            solution=solution,
        )


# Decorator functions for common logging patterns
def log_prediction(
    model_name: str,
    include_inputs: bool = True,
    include_outputs: bool = True,
    logger: PredictionLogger | None = None,
) -> Callable:
    """
    Decorator for prediction functions.

    Args:
        model_name: Name of the prediction model
        include_inputs: Whether to log input parameters
        include_outputs: Whether to log prediction outputs
        logger: PredictionLogger instance to use

    Returns:
        Decorator function
    """
    if logger is None:
        logger = PredictionLogger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Set context for this prediction
            set_context(model_name=model_name, operation="prediction")

            try:
                if include_inputs:
                    logger.logger.info(
                        "Prediction function called",
                        event_type="prediction_input",
                        function=func.__name__,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys()),
                    )

                result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                if include_outputs:
                    logger.logger.info(
                        "Prediction function completed",
                        event_type="prediction_output",
                        function=func.__name__,
                        execution_time=execution_time,
                        result_type=type(result).__name__,
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.logger.error(
                    "Prediction function failed",
                    event_type="prediction_error",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


def log_feature_computation(
    feature_name: str,
    entity_type: str = "player",
    logger: FeatureLogger | None = None,
) -> Callable:
    """
    Decorator for feature computation functions.

    Args:
        feature_name: Name of the feature being computed
        entity_type: Type of entity (player, team, etc.)
        logger: FeatureLogger instance to use

    Returns:
        Decorator function
    """
    if logger is None:
        logger = FeatureLogger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Set context for this feature computation
            set_context(
                feature_name=feature_name,
                entity_type=entity_type,
                operation="feature_computation",
            )

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log successful computation
                if hasattr(result, "__len__") and not isinstance(result, str):
                    result_size = len(result)
                else:
                    result_size = 1

                logger.logger.info(
                    "Feature computation completed",
                    event_type="feature_computation_success",
                    feature_name=feature_name,
                    entity_type=entity_type,
                    function=func.__name__,
                    execution_time=execution_time,
                    result_size=result_size,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.logger.error(
                    "Feature computation failed",
                    event_type="feature_computation_error",
                    feature_name=feature_name,
                    entity_type=entity_type,
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


def log_database_operation(
    operation: str,
    table: str,
    logger: DatabaseLogger | None = None,
) -> Callable:
    """
    Decorator for database operations.

    Args:
        operation: Type of database operation (select, insert, update, delete)
        table: Database table name
        logger: DatabaseLogger instance to use

    Returns:
        Decorator function
    """
    if logger is None:
        logger = DatabaseLogger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Set context for this database operation
            set_context(operation=operation, table=table, operation_type="database")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Try to determine row count from result
                row_count = None
                if hasattr(result, "rowcount"):
                    row_count = result.rowcount
                elif hasattr(result, "__len__") and not isinstance(result, str):
                    row_count = len(result)

                logger.log_query(
                    operation=operation,
                    table=table,
                    execution_time=execution_time,
                    row_count=row_count,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.logger.error(
                    "Database operation failed",
                    event_type="database_error",
                    operation=operation,
                    table=table,
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


def log_api_call(
    service: str,
    endpoint: str,
    logger: APILogger | None = None,
) -> Callable:
    """
    Decorator for API calls.

    Args:
        service: Name of the external service
        endpoint: API endpoint being called
        logger: APILogger instance to use

    Returns:
        Decorator function
    """
    if logger is None:
        logger = APILogger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Set context for this API call
            set_context(service=service, endpoint=endpoint, operation="api_call")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Extract status code if available
                status_code = 200  # Default success
                if hasattr(result, "status_code"):
                    status_code = result.status_code
                elif isinstance(result, dict) and "status_code" in result:
                    status_code = result["status_code"]

                logger.log_api_call(
                    service=service,
                    endpoint=endpoint,
                    method="GET",  # Default, could be enhanced to detect method
                    status_code=status_code,
                    response_time=execution_time,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.log_api_error(
                    service=service,
                    endpoint=endpoint,
                    error=str(e),
                    retry_count=0,
                    will_retry=False,
                )
                raise

        return wrapper

    return decorator


# Create global logger instances for easy access
prediction_logger = PredictionLogger()
feature_logger = FeatureLogger()
database_logger = DatabaseLogger()
api_logger = APILogger()
optimization_logger = OptimizationLogger()

# Export main classes and functions
__all__ = [
    "APILogger",
    "DatabaseLogger",
    "FeatureLogger",
    "OptimizationLogger",
    "PredictionLogger",
    "api_logger",
    "database_logger",
    "feature_logger",
    "log_api_call",
    "log_database_operation",
    "log_feature_computation",
    "log_prediction",
    "optimization_logger",
    "prediction_logger",
]
