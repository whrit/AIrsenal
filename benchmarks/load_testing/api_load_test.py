"""
API Load Testing

Locust-based load testing for AIrsenal API endpoints.
Tests general API performance under various load patterns.

Usage:
    locust -f benchmarks/load_testing/api_load_test.py --host=http://localhost:5000
"""

import builtins
import contextlib
import random

from locust import HttpUser, between, task

from benchmarks.configs.benchmark_config import BenchmarkConfig


class GeneralAPIUser(HttpUser):
    """Simulates general API usage patterns."""

    wait_time = between(1, 5)

    def on_start(self):
        """Initialize user session."""
        self.player_ids = list(range(1, 501))
        self.team_ids = list(range(1, 21))
        self.seasons = ["2324", "2425"]
        self.gameweeks = list(range(1, 39))

    @task(3)
    def get_player_info(self):
        """Get basic player information."""
        player_id = random.choice(self.player_ids)

        with self.client.get(
            f"/api/players/{player_id}", catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()

    @task(2)
    def get_team_info(self):
        """Get team information."""
        team_id = random.choice(self.team_ids)

        with self.client.get(f"/api/teams/{team_id}", catch_response=True) as response:
            if response.status_code == 404:
                response.success()

    @task(2)
    def get_fixtures(self):
        """Get fixture information."""
        gameweek = random.choice(self.gameweeks)

        with self.client.get(
            "/api/fixtures", params={"gameweek": gameweek}, catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()

    @task(1)
    def get_league_table(self):
        """Get league table."""
        with self.client.get("/api/league_table", catch_response=True) as response:
            if response.status_code == 404:
                response.success()


class DataAPIUser(HttpUser):
    """Simulates data-intensive API usage."""

    wait_time = between(2, 8)

    def on_start(self):
        self.seasons = ["2324", "2425"]
        self.gameweeks = list(range(1, 39))

    @task(2)
    def get_player_history(self):
        """Get player performance history."""
        player_id = random.randint(1, 500)
        season = random.choice(self.seasons)

        with self.client.get(
            f"/api/players/{player_id}/history",
            params={"season": season},
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()

    @task(1)
    def get_team_stats(self):
        """Get detailed team statistics."""
        team_id = random.randint(1, 20)
        season = random.choice(self.seasons)

        with self.client.get(
            f"/api/teams/{team_id}/stats",
            params={"season": season},
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()

    @task(1)
    def export_data(self):
        """Export data (heavy operation)."""
        with self.client.get(
            "/api/export/players",
            params={"format": "csv", "season": "2425"},
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()


def run_api_load_test(config: BenchmarkConfig = None):
    """Run API load test programmatically."""
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    if config is None:
        from benchmarks.configs.benchmark_config import get_config

        config = get_config()

    # Create locust configuration
    locust_config = f"""
users = {config.load_test_users}
spawn-rate = {config.load_test_spawn_rate}
run-time = {config.load_test_duration}s
host = http://localhost:5000
headless = true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
        f.write(locust_config)
        config_file = f.name

    try:
        script_path = Path(__file__).absolute()

        cmd = [
            "locust",
            "-f",
            str(script_path),
            "--config",
            config_file,
            "--csv",
            "benchmarks/reports/api_load_test",
            "--html",
            "benchmarks/reports/api_load_test_report.html",
        ]

        print(f"Running API load test: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        print("API load test output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        return result.returncode == 0

    except FileNotFoundError:
        print("Locust not found. Install with: pip install locust")
        return False

    finally:
        with contextlib.suppress(builtins.BaseException):
            os.unlink(config_file)


if __name__ == "__main__":
    print("AIrsenal API Load Test")
    print("Use: locust -f api_load_test.py --host=http://localhost:5000")
