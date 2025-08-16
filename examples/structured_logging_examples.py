"""
Examples demonstrating the AIrsenal structured logging framework.

This file shows practical examples of how to use the logging framework
in different scenarios within AIrsenal.
"""

import time
from datetime import datetime
from typing import List, Dict, Any

# Import the logging framework
from airsenal.framework.logging_config import (
    get_logger, 
    set_correlation_id, 
    set_context,
    with_correlation_id,
    with_context
)
from airsenal.framework.logging_utils import (
    log_prediction,
    log_feature_computation,
    log_api_call,
    timed,
    prediction_logger,
    feature_logger,
    api_logger,
    optimization_logger
)


# Example 1: Basic logging with structured data
def example_basic_logging():
    """Demonstrate basic structured logging."""
    logger = get_logger(__name__)
    
    logger.info("Starting basic logging example")
    
    # Log with structured data
    logger.info(
        "Player analysis started",
        player_id=123,
        player_name="Mohamed Salah",
        gameweek=5,
        season="2324",
        event_type="analysis_start"
    )
    
    # Simulate some work
    time.sleep(0.1)
    
    # Log completion with metrics
    logger.info(
        "Player analysis completed",
        player_id=123,
        execution_time=0.1,
        result="success",
        points_predicted=8.5,
        event_type="analysis_complete"
    )


# Example 2: Using correlation IDs to track requests
def example_correlation_tracking():
    """Demonstrate correlation ID usage for request tracking."""
    logger = get_logger(__name__)
    
    # Set a correlation ID for this operation
    correlation_id = set_correlation_id()
    logger.info("Starting correlated operation", correlation_id=correlation_id)
    
    # All subsequent operations will use this correlation ID
    process_player_data(123)
    run_prediction_model(123)
    store_results(123)
    
    logger.info("Correlated operation completed")


def process_player_data(player_id: int):
    """Simulate processing player data."""
    logger = get_logger(__name__)
    logger.info("Processing player data", player_id=player_id)
    time.sleep(0.05)


def run_prediction_model(player_id: int):
    """Simulate running a prediction model."""
    logger = get_logger(__name__)
    logger.info("Running prediction model", player_id=player_id)
    time.sleep(0.1)


def store_results(player_id: int):
    """Simulate storing prediction results."""
    logger = get_logger(__name__)
    logger.info("Storing prediction results", player_id=player_id)
    time.sleep(0.02)


# Example 3: Using decorators for automatic logging
@log_prediction(model_name="example_model", include_inputs=True, include_outputs=True)
@timed(operation="player_points_prediction", threshold=0.5)
def predict_player_points(player_id: int, gameweek: int, features: Dict[str, Any]) -> float:
    """
    Example prediction function with automatic logging.
    
    The decorators will automatically log:
    - Function inputs and outputs
    - Execution time (if > 0.5 seconds)
    - Prediction-specific metadata
    """
    logger = get_logger(__name__)
    
    # Simulate feature processing
    logger.info("Processing features", feature_count=len(features))
    time.sleep(0.1)
    
    # Simulate model inference
    logger.info("Running model inference")
    predicted_points = 8.5  # Simulate prediction
    time.sleep(0.2)
    
    # Log prediction details
    prediction_logger.log_prediction_result(
        model_name="example_model",
        player_id=player_id,
        gameweek=gameweek,
        predicted_points=predicted_points,
        confidence=0.85,
        features_used=list(features.keys()),
        execution_time=0.3
    )
    
    return predicted_points


# Example 4: Feature computation with logging
@log_feature_computation(feature_name="player_form", entity_type="player")
@timed(operation="form_computation", threshold=0.1)
def compute_player_form(player_id: int, recent_gameweeks: int = 5) -> float:
    """
    Example feature computation with automatic logging.
    """
    logger = get_logger(__name__)
    
    # Set context for this computation
    set_context(
        feature_name="player_form",
        player_id=player_id,
        lookback_gameweeks=recent_gameweeks
    )
    
    logger.info("Starting form computation")
    
    # Simulate data retrieval
    time.sleep(0.05)
    logger.info("Retrieved historical data", gameweeks=recent_gameweeks)
    
    # Simulate computation
    form_score = 0.75  # Simulate calculated form
    time.sleep(0.03)
    
    # Log the computed feature
    feature_logger.log_feature_computation(
        feature_name="player_form",
        entity_type="player",
        entity_id=player_id,
        value=form_score,
        computation_time=0.08,
        data_sources=["player_scores", "fixtures"]
    )
    
    return form_score


# Example 5: API call logging
@log_api_call(service="FPL_API", endpoint="/bootstrap-static/")
@timed(operation="fpl_api_call", threshold=1.0)
def fetch_fpl_data(endpoint: str) -> Dict[str, Any]:
    """
    Example API call with automatic logging.
    """
    logger = get_logger(__name__)
    
    logger.info("Starting FPL API call", endpoint=endpoint)
    
    # Simulate API call
    time.sleep(0.5)
    
    # Simulate response
    response_data = {"players": [], "teams": [], "events": []}
    
    # Log API call details
    api_logger.log_api_call(
        service="FPL_API",
        endpoint=endpoint,
        method="GET",
        status_code=200,
        response_time=0.5,
        response_size=1024000,  # 1MB
    )
    
    return response_data


# Example 6: Context propagation
@with_context(operation="squad_optimization", season="2324")
def example_context_propagation():
    """Demonstrate context propagation across function calls."""
    logger = get_logger(__name__)
    
    # Set additional context
    set_context(gameweek=5, budget=100.0)
    
    logger.info("Starting squad optimization")
    
    # These functions will inherit the context
    analyze_player_pool()
    run_optimization_algorithm()
    validate_solution()
    
    logger.info("Squad optimization completed")


def analyze_player_pool():
    """Analyze available players."""
    logger = get_logger(__name__)
    logger.info("Analyzing player pool")  # Will include inherited context
    time.sleep(0.1)


def run_optimization_algorithm():
    """Run the optimization algorithm."""
    logger = get_logger(__name__)
    logger.info("Running optimization algorithm")  # Will include inherited context
    
    # Log optimization progress
    optimization_logger.log_optimization_progress(
        optimization_type="squad_optimization",
        generation=50,
        best_score=85.2,
        population_size=100,
        convergence_metric=0.001
    )
    
    time.sleep(0.5)


def validate_solution():
    """Validate the optimization solution."""
    logger = get_logger(__name__)
    logger.info("Validating solution")  # Will include inherited context
    time.sleep(0.05)


# Example 7: Error handling and logging
def example_error_handling():
    """Demonstrate error logging with full context."""
    logger = get_logger(__name__)
    correlation_id = set_correlation_id()
    
    set_context(
        operation="error_example",
        player_id=123,
        gameweek=5
    )
    
    try:
        logger.info("Starting operation that might fail")
        
        # Simulate an error
        raise ValueError("Example error for demonstration")
        
    except ValueError as e:
        logger.error(
            "Operation failed with ValueError",
            error=str(e),
            error_type=type(e).__name__,
            event_type="operation_error"
        )
        
        # Re-raise the error
        raise
    
    except Exception as e:
        logger.critical(
            "Operation failed with unexpected error",
            error=str(e),
            error_type=type(e).__name__,
            event_type="operation_critical_error"
        )
        raise


# Example 8: Batch processing with progress logging
def example_batch_processing():
    """Demonstrate logging for batch processing operations."""
    logger = get_logger(__name__)
    correlation_id = set_correlation_id()
    
    player_ids = [123, 456, 789, 101, 112]  # Example player IDs
    gameweek = 5
    
    set_context(
        operation="batch_prediction",
        gameweek=gameweek,
        batch_size=len(player_ids)
    )
    
    logger.info("Starting batch prediction", player_count=len(player_ids))
    
    # Log batch start
    prediction_logger.log_prediction_start(
        model_name="batch_model",
        player_ids=player_ids,
        gameweek_range=(gameweek, gameweek),
        features={"form": True, "fixtures": True, "minutes": True}
    )
    
    successful_predictions = 0
    failed_predictions = 0
    total_time = 0
    
    for i, player_id in enumerate(player_ids):
        start_time = time.time()
        
        try:
            # Simulate prediction
            predicted_points = predict_player_points(
                player_id, 
                gameweek, 
                {"form": 0.75, "fixture_difficulty": 3}
            )
            
            execution_time = time.time() - start_time
            total_time += execution_time
            successful_predictions += 1
            
            # Log progress
            if (i + 1) % 2 == 0:  # Log every 2 predictions
                logger.info(
                    "Batch progress",
                    completed=i + 1,
                    total=len(player_ids),
                    success_rate=successful_predictions / (i + 1)
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            total_time += execution_time
            failed_predictions += 1
            
            logger.error(
                "Prediction failed for player",
                player_id=player_id,
                error=str(e),
                event_type="batch_prediction_error"
            )
    
    # Log batch summary
    avg_time = total_time / len(player_ids) if player_ids else 0
    
    prediction_logger.log_prediction_batch_summary(
        model_name="batch_model",
        total_predictions=len(player_ids),
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions,
        total_time=total_time,
        avg_prediction_time=avg_time
    )
    
    logger.info("Batch prediction completed")


# Example 9: Performance monitoring
@timed(operation="performance_example", threshold=0.0)  # Log all executions
def example_performance_monitoring():
    """Demonstrate performance monitoring and logging."""
    logger = get_logger(__name__)
    
    # Simulate different phases of processing
    phases = [
        ("data_loading", 0.1),
        ("feature_engineering", 0.2),
        ("model_inference", 0.3),
        ("result_processing", 0.05)
    ]
    
    total_start = time.time()
    
    for phase_name, duration in phases:
        phase_start = time.time()
        
        logger.info(f"Starting {phase_name}")
        time.sleep(duration)  # Simulate work
        
        phase_time = time.time() - phase_start
        logger.info(
            f"Completed {phase_name}",
            phase=phase_name,
            execution_time=phase_time,
            event_type="phase_timing"
        )
    
    total_time = time.time() - total_start
    logger.info(
        "All phases completed",
        total_execution_time=total_time,
        phase_count=len(phases),
        event_type="performance_summary"
    )


# Example 10: Complex workflow with nested operations
@with_correlation_id()
def example_complex_workflow():
    """Demonstrate a complex workflow with nested operations and logging."""
    logger = get_logger(__name__)
    
    set_context(
        workflow="gameweek_processing",
        gameweek=5,
        season="2324"
    )
    
    logger.info("Starting complex workflow")
    
    try:
        # Phase 1: Data preparation
        logger.info("Phase 1: Data preparation")
        prepare_gameweek_data()
        
        # Phase 2: Feature computation
        logger.info("Phase 2: Feature computation")
        compute_all_features()
        
        # Phase 3: Model predictions
        logger.info("Phase 3: Model predictions")
        run_all_predictions()
        
        # Phase 4: Optimization
        logger.info("Phase 4: Squad optimization")
        optimize_squads()
        
        logger.info("Complex workflow completed successfully")
        
    except Exception as e:
        logger.error(
            "Complex workflow failed",
            error=str(e),
            error_type=type(e).__name__,
            event_type="workflow_error"
        )
        raise


def prepare_gameweek_data():
    """Simulate data preparation."""
    logger = get_logger(__name__)
    logger.info("Preparing gameweek data")
    time.sleep(0.1)


def compute_all_features():
    """Simulate feature computation."""
    logger = get_logger(__name__)
    logger.info("Computing features for all players")
    
    # Simulate computing different features
    features = ["form", "fixture_difficulty", "minutes_expected"]
    for feature in features:
        logger.info(f"Computing {feature}")
        time.sleep(0.05)


def run_all_predictions():
    """Simulate running predictions."""
    logger = get_logger(__name__)
    logger.info("Running predictions for all players")
    time.sleep(0.2)


def optimize_squads():
    """Simulate squad optimization."""
    logger = get_logger(__name__)
    logger.info("Optimizing squad selection")
    
    # Log optimization start
    optimization_logger.log_optimization_start(
        optimization_type="squad_selection",
        parameters={"budget": 100.0, "formation": "3-5-2"},
        constraints={"max_players_per_team": 3}
    )
    
    time.sleep(0.3)
    
    # Log optimization result
    optimization_logger.log_optimization_result(
        optimization_type="squad_selection",
        final_score=85.2,
        total_generations=100,
        execution_time=0.3,
        solution={"formation": "3-5-2", "total_cost": 99.5}
    )


def main():
    """Run all examples."""
    print("Running structured logging examples...")
    
    examples = [
        ("Basic Logging", example_basic_logging),
        ("Correlation Tracking", example_correlation_tracking),
        ("Context Propagation", example_context_propagation),
        ("Batch Processing", example_batch_processing),
        ("Performance Monitoring", example_performance_monitoring),
        ("Complex Workflow", example_complex_workflow),
    ]
    
    for name, example_func in examples:
        print(f"\n--- {name} ---")
        try:
            example_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    # Demonstrate error handling
    print("\n--- Error Handling ---")
    try:
        example_error_handling()
    except ValueError:
        print("✓ Error handling example completed (error was expected)")


if __name__ == "__main__":
    main()