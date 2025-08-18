"""
Structured logging configuration for AIrsenal.

Provides centralized logging setup with correlation IDs, performance timing,
context propagation, and configurable output formats.
"""

import contextvars
import functools
import logging
import logging.handlers
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import orjson
import structlog
from platformdirs import user_data_dir

# Context variables for correlation tracking
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)
context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "context", default={}
)

# Sensitive data patterns to mask in logs
SENSITIVE_PATTERNS = [
    "password",
    "token",
    "key",
    "secret",
    "auth",
    "credential",
    "fpl_password",
    "fpl_login",
]


def mask_sensitive_data(data: Any) -> Any:
    """
    Recursively mask sensitive data in log records.

    Args:
        data: Data structure to mask

    Returns:
        Data structure with sensitive fields masked
    """
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if any(pattern in key.lower() for pattern in SENSITIVE_PATTERNS):
                masked[key] = "***MASKED***"
            else:
                masked[key] = mask_sensitive_data(value)
        return masked
    if isinstance(data, list | tuple):
        return type(data)(mask_sensitive_data(item) for item in data)
    return data


def add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add correlation ID to log records."""
    correlation_id = correlation_id_var.get("")
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add context variables to log records."""
    context = context_var.get({})
    if context:
        event_dict["context"] = context
    return event_dict


def mask_sensitive_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor to mask sensitive data in log records."""
    return mask_sensitive_data(event_dict)


def json_serializer(obj: Any) -> str:
    """Custom JSON serializer using orjson for performance."""
    return orjson.dumps(obj, option=orjson.OPT_UTC_Z).decode("utf-8")


def get_log_level() -> int:
    """Get log level from environment variable."""
    level_name = os.getenv("AIRSENAL_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_log_directory() -> Path:
    """Get the directory for log files."""
    airsenal_home = os.getenv("AIRSENAL_HOME")
    if airsenal_home:
        log_dir = Path(airsenal_home) / "logs"
    else:
        log_dir = Path(user_data_dir("airsenal", "airsenal")) / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_file_handlers() -> list[logging.Handler]:
    """Set up rotating file handlers for different log levels."""
    log_dir = get_log_directory()
    handlers: list[logging.Handler] = []

    # Main application log (all levels)
    main_handler = logging.handlers.RotatingFileHandler(
        log_dir / "airsenal.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    main_handler.setLevel(logging.DEBUG)
    handlers.append(main_handler)

    # Error log (errors and above only)
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "airsenal_errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
    )
    error_handler.setLevel(logging.ERROR)
    handlers.append(error_handler)

    # Performance log (for timing and metrics)
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "airsenal_performance.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.addFilter(
        lambda record: hasattr(record, "event_type")
        and record.event_type in ["performance", "timing", "metrics"]
    )
    handlers.append(perf_handler)

    return handlers


def configure_structlog(
    level: str | int | None = None,
    use_colors: bool | None = None,
    json_logs: bool | None = None,
    enable_file_logging: bool = True,
) -> None:
    """
    Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_colors: Whether to use colored output for console
        json_logs: Whether to output JSON format
        enable_file_logging: Whether to enable file logging
    """
    if level is None:
        level = get_log_level()
    elif isinstance(level, str):
        level = getattr(logging, level.upper())

    if use_colors is None:
        use_colors = sys.stderr.isatty() and os.getenv("NO_COLOR") != "1"

    if json_logs is None:
        json_logs = os.getenv("AIRSENAL_JSON_LOGS", "false").lower() == "true"

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=level,
        force=True,  # Force reconfiguration
    )

    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handlers if enabled
    if enable_file_logging:
        for handler in setup_file_handlers():
            root_logger.addHandler(handler)

    # Configure processors
    processors = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        add_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        mask_sensitive_processor,
    ]

    # Choose renderer based on configuration
    if json_logs:
        processors.append(structlog.processors.JSONRenderer(serializer=json_serializer))
    else:
        if use_colors:
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            processors.append(structlog.processors.KeyValueRenderer())

    # Configure structlog
    structlog.configure(
        processors=cast(Any, processors),
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: str | None = None) -> str:
    """
    Set correlation ID for the current context.

    Args:
        correlation_id: Correlation ID to set. If None, generates a new UUID4.

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id_var.get("")


def set_context(**kwargs: Any) -> None:
    """
    Set context variables for the current execution.

    Args:
        **kwargs: Key-value pairs to add to context
    """
    current_context = context_var.get({}).copy()
    current_context.update(kwargs)
    context_var.set(current_context)


def clear_context() -> None:
    """Clear all context variables."""
    context_var.set({})


def with_correlation_id(correlation_id: str | None = None) -> Callable:
    """
    Decorator to set correlation ID for a function.

    Args:
        correlation_id: Correlation ID to use. If None, generates a new one.

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save current correlation ID
            old_correlation_id = correlation_id_var.get("")

            # Set new correlation ID
            set_correlation_id(correlation_id)

            try:
                return func(*args, **kwargs)
            finally:
                # Restore old correlation ID
                correlation_id_var.set(old_correlation_id)

        return wrapper

    return decorator


def with_context(**context_kwargs: Any) -> Callable:
    """
    Decorator to add context to a function.

    Args:
        **context_kwargs: Context key-value pairs to add

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save current context
            old_context = context_var.get({}).copy()

            # Add new context
            current_context = old_context.copy()
            current_context.update(context_kwargs)
            context_var.set(current_context)

            try:
                return func(*args, **kwargs)
            finally:
                # Restore old context
                context_var.set(old_context)

        return wrapper

    return decorator


def logged_function(
    logger: structlog.stdlib.BoundLogger | None = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
    mask_args: bool = True,
) -> Callable:
    """
    Decorator to add logging to function calls.

    Args:
        logger: Logger to use. If None, creates one based on function module.
        level: Log level to use
        include_args: Whether to include function arguments in logs
        include_result: Whether to include function result in logs
        mask_args: Whether to mask sensitive arguments

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()

            # Prepare log context
            log_context: dict[str, Any] = {
                "function": func_name,
                "event_type": "function_call",
            }

            if include_args:
                func_args: dict[str, Any] = {"args": args, "kwargs": kwargs}
                if mask_args:
                    func_args = mask_sensitive_data(func_args)
                log_context["arguments"] = func_args

            logger.log(level, "Function started", **log_context)

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                success_context = {
                    "function": func_name,
                    "event_type": "function_success",
                    "execution_time": execution_time,
                }

                if include_result and result is not None:
                    success_context["result"] = (
                        mask_sensitive_data(result) if mask_args else result
                    )

                logger.log(level, "Function completed", **success_context)
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                error_context = {
                    "function": func_name,
                    "event_type": "function_error",
                    "execution_time": execution_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

                logger.error("Function failed", **error_context)
                raise

        return wrapper

    return decorator


def timed(
    logger: structlog.stdlib.BoundLogger | None = None,
    operation: str | None = None,
    threshold: float = 0.0,
) -> Callable:
    """
    Decorator to time function execution and log if above threshold.

    Args:
        logger: Logger to use. If None, creates one based on function module.
        operation: Operation name for logging. If None, uses function name.
        threshold: Minimum execution time (seconds) to log

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger, operation
        if logger is None:
            logger = get_logger(func.__module__)
        if operation is None:
            operation = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                if execution_time >= threshold:
                    logger.info(
                        "Operation timing",
                        operation=operation,
                        execution_time=execution_time,
                        event_type="timing",
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Operation failed",
                    operation=operation,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                    event_type="timing",
                )
                raise

        return wrapper

    return decorator


# Initialize logging on module import
configure_structlog()

# Export main functions
__all__ = [
    "clear_context",
    "configure_structlog",
    "get_correlation_id",
    "get_logger",
    "logged_function",
    "mask_sensitive_data",
    "set_context",
    "set_correlation_id",
    "timed",
    "with_context",
    "with_correlation_id",
]
