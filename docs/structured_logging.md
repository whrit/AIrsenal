# AIrsenal Structured Logging Framework

This document describes the structured logging framework implemented in AIrsenal, which provides comprehensive logging for ML operations, API calls, database interactions, and system events.

## Overview

The structured logging framework uses `structlog` to provide:
- **JSON-formatted logs** for easy parsing and analysis
- **Correlation IDs** for tracking requests across the system
- **Context propagation** for maintaining request-specific information
- **Performance timing** and metrics collection
- **Sensitive data masking** for security
- **Log aggregation** and rotation
- **Domain-specific loggers** for different components

## Quick Start

### Basic Usage

```python
from airsenal.framework.logging_config import get_logger

logger = get_logger(__name__)

# Simple logging
logger.info("Starting player analysis", player_id=123, gameweek=5)

# With structured data
logger.info(
    "Prediction completed",
    player_id=123,
    predicted_points=8.5,
    confidence=0.85,
    model="team_player_model"
)
```

### Using Decorators

```python
from airsenal.framework.logging_utils import log_prediction, timed

@log_prediction(model_name="my_model", include_inputs=True)
@timed(operation="player_prediction", threshold=1.0)
def predict_player_points(player_id, gameweek):
    # Your prediction logic here
    return predicted_points
```

### Setting Context and Correlation IDs

```python
from airsenal.framework.logging_config import set_correlation_id, set_context

# Set correlation ID for tracking requests
correlation_id = set_correlation_id()

# Add context that will be included in all subsequent logs
set_context(
    user_id=123,
    operation="squad_optimization",
    gameweek=5
)
```

## Configuration

### Environment Variables

Configure logging behavior with environment variables:

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export AIRSENAL_LOG_LEVEL=INFO

# Enable JSON output (default: false)
export AIRSENAL_JSON_LOGS=true

# Log directory (default: platform-specific user data dir)
export AIRSENAL_HOME=/path/to/airsenal/data

# Disable colored output
export NO_COLOR=1

# Slow query threshold in seconds (default: 1.0)
export AIRSENAL_SLOW_QUERY_THRESHOLD=0.5
```

### Programmatic Configuration

```python
from airsenal.framework.logging_config import configure_structlog

# Configure with custom settings
configure_structlog(
    level="DEBUG",
    use_colors=True,
    json_logs=False,
    enable_file_logging=True
)
```

## Log Files

Logs are automatically rotated and stored in the following files:

- **`airsenal.log`**: Main application log (all levels)
- **`airsenal_errors.log`**: Errors and critical messages only
- **`airsenal_performance.log`**: Performance and timing data

Log files are rotated when they reach 10MB (main log) or 5MB (error log), with 5 and 3 backup files retained respectively.

## Domain-Specific Loggers

### Prediction Logger

For ML model predictions and scoring:

```python
from airsenal.framework.logging_utils import prediction_logger

# Log prediction start
prediction_logger.log_prediction_start(
    model_name="team_player_model",
    player_ids=[123, 456, 789],
    gameweek_range=(5, 7),
    features={"form": True, "fixtures": True}
)

# Log individual prediction
prediction_logger.log_prediction_result(
    model_name="team_player_model",
    player_id=123,
    gameweek=5,
    predicted_points=8.5,
    confidence=0.85,
    features_used=["form", "minutes", "fixtures"],
    execution_time=0.15
)

# Log batch summary
prediction_logger.log_prediction_batch_summary(
    model_name="team_player_model",
    total_predictions=100,
    successful_predictions=98,
    failed_predictions=2,
    total_time=15.2,
    avg_prediction_time=0.152
)
```

### Feature Logger

For feature engineering and computation:

```python
from airsenal.framework.logging_utils import feature_logger

# Log feature computation
feature_logger.log_feature_computation(
    feature_name="player_form",
    entity_type="player",
    entity_id=123,
    value=0.75,
    computation_time=0.05,
    data_sources=["player_scores", "fixtures"]
)

# Log feature store operations
feature_logger.log_feature_store_operation(
    operation="batch_compute",
    feature_names=["form", "difficulty", "minutes"],
    entity_count=50,
    cache_hits=35,
    cache_misses=15,
    execution_time=2.3
)
```

### API Logger

For external API calls:

```python
from airsenal.framework.logging_utils import api_logger

# This is automatically called by the decorated _get_request method
# but you can use it directly for other API calls
api_logger.log_api_call(
    service="FPL_API",
    endpoint="/api/bootstrap-static/",
    method="GET",
    status_code=200,
    response_time=0.45,
    response_size=1024768
)
```

### Database Logger

For database operations:

```python
from airsenal.framework.logging_utils import database_logger

# Use the decorator for automatic logging
@log_database_operation(operation="select", table="player_scores")
def get_player_scores(player_id, season):
    # Your database query here
    return results
```

### Optimization Logger

For genetic algorithms and optimization:

```python
from airsenal.framework.logging_utils import optimization_logger

# Log optimization start
optimization_logger.log_optimization_start(
    optimization_type="transfer_optimization",
    parameters={"budget": 100.0, "transfers": 2},
    constraints={"formation": "3-5-2"}
)

# Log optimization progress
optimization_logger.log_optimization_progress(
    optimization_type="transfer_optimization",
    generation=50,
    best_score=85.2,
    population_size=100,
    convergence_metric=0.001
)
```

## Decorators

### @log_prediction

Automatically logs prediction function calls:

```python
@log_prediction(
    model_name="my_model",
    include_inputs=True,      # Log function arguments
    include_outputs=True,     # Log function results
    logger=my_prediction_logger  # Optional custom logger
)
def my_prediction_function(player_id, gameweek):
    return predicted_points
```

### @timed

Logs execution time for functions:

```python
@timed(
    operation="database_query",  # Operation name for logs
    threshold=0.5               # Only log if execution time > threshold
)
def slow_database_query():
    # Your code here
    pass
```

### @log_feature_computation

Automatically logs feature computation:

```python
@log_feature_computation(
    feature_name="player_form",
    entity_type="player"
)
def compute_player_form(player_id, gameweeks):
    return form_score
```

### @with_correlation_id

Ensures a correlation ID is set for the function:

```python
@with_correlation_id()  # Generates new ID if none exists
def process_gameweek_predictions():
    # All logs in this function will have the same correlation ID
    pass

@with_correlation_id("custom-id-123")  # Use specific ID
def process_with_custom_id():
    pass
```

### @with_context

Adds context to all logs within a function:

```python
@with_context(operation="squad_optimization", gameweek=5)
def optimize_squad():
    # All logs will include operation and gameweek context
    pass
```

## Log Analysis

### Using the LogAnalyzer

```python
from airsenal.framework.log_analysis import LogAnalyzer

analyzer = LogAnalyzer()

# Get performance metrics for the last 24 hours
metrics = analyzer.get_performance_metrics(hours_back=24)
print(f"Average prediction time: {metrics['prediction_mean']:.3f}s")

# Analyze errors
error_analysis = analyzer.get_error_analysis(hours_back=24)
print(f"Total errors: {error_analysis['summary']['total_errors']}")

# Analyze predictions
pred_analysis = analyzer.get_prediction_analysis(hours_back=24)
for model, perf in pred_analysis['model_performance'].items():
    print(f"{model}: {perf['success_rate']:.1%} success rate")
```

### Correlation Chain Analysis

Track all operations for a specific correlation ID:

```python
from airsenal.framework.log_analysis import analyze_correlation_chains

# Analyze a specific correlation chain
chain_analysis = analyze_correlation_chains("abc-123-def", hours_back=24)
print(f"Total operations: {chain_analysis['summary']['total_entries']}")
print(f"Time span: {chain_analysis['summary']['time_span']:.1f} seconds")
```

### Searching Logs

```python
from airsenal.framework.log_analysis import search_logs

# Search for specific patterns
results = search_logs(
    pattern="prediction.*failed",
    hours_back=24,
    log_level="ERROR",
    event_type="prediction_error"
)

for result in results:
    print(f"Error: {result['error']}")
```

### Export Analysis to CSV

```python
from pathlib import Path

# Export all analyses to CSV files
file_paths = analyzer.export_to_csv(
    output_dir=Path("./log_analysis"),
    hours_back=24
)

print("Analysis exported to:")
for analysis_type, file_path in file_paths.items():
    print(f"  {analysis_type}: {file_path}")
```

## Log Format

### Standard Log Entry

```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "level": "info",
  "logger": "airsenal.framework.prediction_utils",
  "event": "Prediction completed",
  "correlation_id": "abc-123-def-456",
  "context": {
    "player_id": 123,
    "season": "2324",
    "operation": "prediction"
  },
  "player_id": 123,
  "gameweek": 5,
  "predicted_points": 8.5,
  "confidence": 0.85,
  "execution_time": 0.152,
  "event_type": "prediction_result"
}
```

### Error Log Entry

```json
{
  "timestamp": "2024-01-15T10:31:02.789012Z",
  "level": "error",
  "logger": "airsenal.framework.data_fetcher",
  "event": "FPL API request HTTP error",
  "correlation_id": "abc-123-def-456",
  "url": "https://fantasy.premierleague.com/api/bootstrap-static/",
  "status_code": 429,
  "total_time": 5.2,
  "error": "Too Many Requests",
  "response_content": "{\"detail\":\"Request was throttled...\"}",
  "event_type": "api_request_http_error"
}
```

## Best Practices

### 1. Use Correlation IDs

Always set correlation IDs for operations that span multiple functions:

```python
from airsenal.framework.logging_config import set_correlation_id

def process_gameweek():
    correlation_id = set_correlation_id()
    logger.info("Starting gameweek processing", correlation_id=correlation_id)
    
    # All subsequent operations will use this correlation ID
    update_player_data()
    run_predictions()
    optimize_transfers()
```

### 2. Add Meaningful Context

Use context to provide additional information:

```python
set_context(
    gameweek=current_gameweek,
    operation="squad_optimization",
    user_id=123
)
```

### 3. Log at Appropriate Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about operations
- **WARNING**: Something unexpected but not critical
- **ERROR**: Error conditions that need attention
- **CRITICAL**: Serious errors that may cause system failure

### 4. Include Relevant Metrics

Always include performance and business metrics:

```python
logger.info(
    "Optimization completed",
    execution_time=15.2,
    total_players=100,
    success_rate=0.95,
    final_score=85.2
)
```

### 5. Use Event Types

Consistently use event types for easier log analysis:

```python
logger.info(
    "Database query completed",
    event_type="database_query",
    table="player_scores",
    row_count=150,
    execution_time=0.25
)
```

## Integration Examples

### Prediction Pipeline

```python
from airsenal.framework.logging_config import set_correlation_id, set_context
from airsenal.framework.logging_utils import prediction_logger

def run_prediction_pipeline(gameweek_range):
    # Set correlation ID for the entire pipeline
    correlation_id = set_correlation_id()
    
    # Set context
    set_context(
        operation="prediction_pipeline",
        gameweek_range=gameweek_range
    )
    
    logger.info("Starting prediction pipeline", gameweek_range=gameweek_range)
    
    try:
        # Update data
        update_player_data()
        
        # Run predictions
        players = get_all_players()
        prediction_logger.log_prediction_start(
            model_name="team_player_model",
            player_ids=[p.id for p in players],
            gameweek_range=gameweek_range
        )
        
        for player in players:
            predictions = predict_player_points(player, gameweek_range)
            # Predictions are automatically logged by the decorator
        
        logger.info("Prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error(
            "Prediction pipeline failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### Custom Analysis

```python
def analyze_model_performance():
    """Example of using the logging framework for model evaluation."""
    from airsenal.framework.log_analysis import LogAnalyzer
    
    analyzer = LogAnalyzer()
    
    # Get prediction analysis for the last week
    analysis = analyzer.get_prediction_analysis(hours_back=168)  # 7 days
    
    logger.info("Model performance analysis", **analysis['summary'])
    
    # Alert on poor performance
    for model, perf in analysis['model_performance'].items():
        if perf['success_rate'] < 0.9:
            logger.warning(
                "Model performance below threshold",
                model=model,
                success_rate=perf['success_rate'],
                threshold=0.9
            )
```

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check `AIRSENAL_LOG_LEVEL` environment variable
2. **Large log files**: Verify log rotation is working and adjust rotation settings
3. **Performance impact**: Use appropriate log levels and thresholds for timing decorators
4. **Missing correlation IDs**: Ensure `set_correlation_id()` is called at the start of operations

### Debugging

Enable debug logging to see detailed information:

```bash
export AIRSENAL_LOG_LEVEL=DEBUG
```

Check log files location:

```python
from airsenal.framework.logging_config import get_log_directory
print(f"Logs directory: {get_log_directory()}")
```

## Future Enhancements

Potential improvements to the logging framework:

1. **ELK Stack Integration**: Send logs to Elasticsearch for advanced analysis
2. **Metrics Export**: Export metrics to Prometheus for monitoring
3. **Alerting**: Integration with PagerDuty or Slack for critical errors
4. **Sampling**: Implement log sampling for high-volume operations
5. **Distributed Tracing**: Add OpenTelemetry support for distributed tracing

## API Reference

See the docstrings in the following modules for detailed API documentation:

- `airsenal.framework.logging_config`: Core logging configuration
- `airsenal.framework.logging_utils`: Domain-specific loggers and decorators
- `airsenal.framework.log_analysis`: Log analysis utilities