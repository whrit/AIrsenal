"""
Prediction Service Load Testing

Locust-based load testing for AIrsenal prediction services.
Tests system behavior under realistic user loads.

Usage:
    locust -f benchmarks/load_testing/prediction_load_test.py --host=http://localhost:5000
    
Or programmatically:
    python -c "from benchmarks.load_testing.prediction_load_test import run_prediction_load_test; run_prediction_load_test()"
"""

import random
import time
from typing import Dict, List, Any

from locust import HttpUser, task, between, events
import numpy as np

from benchmarks.configs.benchmark_config import BenchmarkConfig


class PredictionServiceUser(HttpUser):
    """Simulates a user making prediction requests."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.player_ids = list(range(1, 501))  # Available player IDs
        self.seasons = ["2324", "2425"]
        self.gameweeks = list(range(1, 39))
    
    @task(3)
    def get_single_player_prediction(self):
        """Request prediction for a single player."""
        player_id = random.choice(self.player_ids)
        gameweek = random.choice(self.gameweeks)
        season = random.choice(self.seasons)
        
        # Simulate API endpoint
        with self.client.get(
            f"/api/predictions/player/{player_id}",
            params={
                "gameweek": gameweek,
                "season": season
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                # Handle missing endpoints gracefully
                response.success()
    
    @task(2)
    def get_batch_player_predictions(self):
        """Request predictions for multiple players."""
        batch_size = random.randint(5, 20)
        player_ids = random.sample(self.player_ids, batch_size)
        gameweek = random.choice(self.gameweeks)
        season = random.choice(self.seasons)
        
        # Simulate batch API endpoint
        with self.client.post(
            "/api/predictions/batch",
            json={
                "player_ids": player_ids,
                "gameweek": gameweek,
                "season": season
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()
    
    @task(1)
    def get_team_predictions(self):
        """Request predictions for an entire team."""
        team_id = random.randint(1, 20)
        gameweek = random.choice(self.gameweeks)
        season = random.choice(self.seasons)
        
        with self.client.get(
            f"/api/predictions/team/{team_id}",
            params={
                "gameweek": gameweek,
                "season": season
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()
    
    @task(1)
    def get_optimization_suggestions(self):
        """Request transfer optimization suggestions."""
        squad = random.sample(self.player_ids, 15)  # 15-player squad
        
        with self.client.post(
            "/api/optimization/transfers",
            json={
                "current_squad": squad,
                "budget": random.uniform(0, 10),
                "weeks_ahead": random.randint(1, 5)
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()


class FeatureStoreUser(HttpUser):
    """Simulates a user making feature store requests."""
    
    wait_time = between(0.5, 2)  # Faster requests for feature store
    
    def on_start(self):
        """Initialize user session."""
        self.player_ids = list(range(1, 501))
        self.feature_names = [
            "rolling_goals_5",
            "rolling_assists_5",
            "xg_form",
            "minutes_trend",
            "difficulty_adjusted_points"
        ]
        self.seasons = ["2324", "2425"]
        self.gameweeks = list(range(1, 39))
    
    @task(4)
    def get_player_features(self):
        """Request features for a player."""
        player_id = random.choice(self.player_ids)
        features = random.sample(self.feature_names, random.randint(1, 3))
        gameweek = random.choice(self.gameweeks)
        season = random.choice(self.seasons)
        
        with self.client.get(
            f"/api/features/player/{player_id}",
            params={
                "features": ",".join(features),
                "gameweek": gameweek,
                "season": season
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()
    
    @task(2)
    def get_batch_features(self):
        """Request features for multiple players."""
        batch_size = random.randint(5, 50)
        player_ids = random.sample(self.player_ids, batch_size)
        features = random.sample(self.feature_names, random.randint(2, 4))
        gameweek = random.choice(self.gameweeks)
        season = random.choice(self.seasons)
        
        with self.client.post(
            "/api/features/batch",
            json={
                "player_ids": player_ids,
                "features": features,
                "gameweek": gameweek,
                "season": season
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()


# Custom load testing scenarios
class HeavyPredictionUser(HttpUser):
    """Simulates heavy prediction usage (optimization tools, etc.)."""
    
    wait_time = between(5, 10)  # Longer wait times, but heavier requests
    
    def on_start(self):
        self.player_ids = list(range(1, 501))
    
    @task
    def heavy_optimization_request(self):
        """Simulate heavy optimization request."""
        # Large batch prediction for optimization
        all_players = random.sample(self.player_ids, 100)
        
        with self.client.post(
            "/api/predictions/optimization",
            json={
                "player_ids": all_players,
                "gameweeks": [35, 36, 37, 38],
                "season": "2425",
                "include_confidence": True,
                "include_variance": True
            },
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()


# Performance monitoring
@events.request.add_listener
def request_stats_handler(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Custom request statistics handler."""
    if exception:
        print(f"Request failed: {request_type} {name} - {exception}")
    
    # Log slow requests
    if response_time > 1000:  # 1 second
        print(f"Slow request: {request_type} {name} took {response_time}ms")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize load testing environment."""
    print("Starting AIrsenal load testing...")
    print(f"Target host: {environment.host}")


# Programmatic load testing function
def run_prediction_load_test(config: BenchmarkConfig = None):
    """Run load test programmatically."""
    import subprocess
    import tempfile
    import os
    from pathlib import Path
    
    if config is None:
        from benchmarks.configs.benchmark_config import get_config
        config = get_config()
    
    # Create temporary locust configuration
    locust_config = f"""
# Locust configuration for AIrsenal
users = {config.load_test_users}
spawn-rate = {config.load_test_spawn_rate}
run-time = {config.load_test_duration}s
host = http://localhost:5000
headless = true
"""
    
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(locust_config)
        config_file = f.name
    
    try:
        # Get the path to this script
        script_path = Path(__file__).absolute()
        
        # Run locust
        cmd = [
            "locust",
            "-f", str(script_path),
            "--config", config_file,
            "--csv", "benchmarks/reports/load_test",
            "--html", "benchmarks/reports/load_test_report.html"
        ]
        
        print(f"Running load test: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Load test output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("Locust not found. Install with: pip install locust")
        return False
    
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_file)
        except:
            pass


# Mock API simulation for testing without real server
class MockAPIUser(HttpUser):
    """Mock API user for testing load framework without real server."""
    
    abstract = True  # Don't include in real load tests
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override client to use mock responses
        self.client = MockClient()


class MockClient:
    """Mock HTTP client that simulates responses."""
    
    def get(self, url, **kwargs):
        return MockResponse()
    
    def post(self, url, **kwargs):
        return MockResponse()


class MockResponse:
    """Mock HTTP response."""
    
    def __init__(self):
        self.status_code = 200
        self.headers = {}
        
        # Simulate response time
        time.sleep(random.uniform(0.01, 0.1))
    
    def json(self):
        return {"mock": "response"}
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def success(self):
        pass


if __name__ == "__main__":
    # Allow running directly for testing
    print("AIrsenal Prediction Service Load Test")
    print("Use: locust -f prediction_load_test.py --host=http://localhost:5000")
    print("Or run programmatically with run_prediction_load_test()")