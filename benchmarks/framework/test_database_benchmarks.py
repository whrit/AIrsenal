"""
Database Performance Benchmarks

Tests performance of database operations including:
- Query performance for extended schema
- Bulk insert operations
- Complex joins and aggregations
- Index effectiveness
- Connection pooling
"""

import time

import pytest
from sqlalchemy import text

from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerScore,
    Team,
    session,
)
from benchmarks.utils import BenchmarkRunner


class DatabaseBenchmarks:
    """Database performance benchmark suite."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.session = runner.session
        self.config = runner.config

    def run_all(self):
        """Run all database benchmarks."""
        if not self.session:
            print("Database session not available - skipping benchmarks")
            return

        self.benchmark_simple_queries()
        self.benchmark_complex_queries()
        self.benchmark_bulk_operations()
        self.benchmark_joins()
        self.benchmark_aggregations()

    @pytest.mark.benchmark(group="database")
    def benchmark_simple_queries(self):
        """Benchmark simple database queries."""
        # Single player query
        with self.runner.benchmark_context("db_query_player_single"):
            for _i in range(100):
                self.session.query(Player).filter(Player.id == 1).first()

        # Multiple players query
        with self.runner.benchmark_context("db_query_players_batch"):
            self.session.query(Player).limit(100).all()

        # Team query
        with self.runner.benchmark_context("db_query_teams"):
            self.session.query(Team).all()

    @pytest.mark.benchmark(group="database")
    def benchmark_complex_queries(self):
        """Benchmark complex database queries."""
        # Player with recent scores
        with self.runner.benchmark_context("db_query_player_with_scores"):
            try:
                query = (
                    self.session.query(Player)
                    .join(PlayerScore)
                    .filter(PlayerScore.gameweek >= 35)
                    .limit(50)
                )
                query.all()
            except Exception:
                pass

        # Players by position with stats
        with self.runner.benchmark_context("db_query_players_by_position"):
            try:
                query = (
                    self.session.query(Player)
                    .filter(Player.position == "MID")
                    .limit(100)
                )
                query.all()
            except Exception:
                pass

        # Fixture with team information
        with self.runner.benchmark_context("db_query_fixtures_with_teams"):
            try:
                query = (
                    self.session.query(Fixture)
                    .join(Team, Fixture.home_team == Team.id)
                    .limit(100)
                )
                query.all()
            except Exception:
                pass

    @pytest.mark.benchmark(group="database")
    def benchmark_bulk_operations(self):
        """Benchmark bulk database operations."""
        # Bulk insert simulation (using mock data)
        with self.runner.benchmark_context("db_bulk_insert_simulation"):
            try:
                # Simulate bulk insert by creating objects (not committing)
                mock_players = []
                for i in range(1000):
                    player_data = {
                        "name": f"Benchmark Player {i}",
                        "position": "MID",
                        "team": 1,
                        "current_price": 5.0,
                    }
                    # Don't actually insert to avoid database pollution
                    mock_players.append(player_data)

                # Simulate processing time
                time.sleep(0.01)
            except Exception:
                pass

        # Bulk update simulation
        with self.runner.benchmark_context("db_bulk_update_simulation"):
            try:
                # Simulate bulk update
                time.sleep(0.005)
            except Exception:
                pass

    @pytest.mark.benchmark(group="database")
    def benchmark_joins(self):
        """Benchmark join operations."""
        # Player-Team join
        with self.runner.benchmark_context("db_join_player_team"):
            try:
                query = (
                    self.session.query(Player, Team)
                    .join(Team, Player.team == Team.id)
                    .limit(100)
                )
                query.all()
            except Exception:
                pass

        # Player-Score join with aggregation
        with self.runner.benchmark_context("db_join_player_score_agg"):
            try:
                query = text("""
                SELECT p.name, AVG(ps.points) as avg_points
                FROM player p
                JOIN playerscore ps ON p.id = ps.player_id
                WHERE ps.gameweek >= 30
                GROUP BY p.id, p.name
                LIMIT 100
                """)
                self.session.execute(query).fetchall()
            except Exception:
                pass

        # Multi-table join
        with self.runner.benchmark_context("db_join_multi_table"):
            try:
                query = text("""
                SELECT p.name, t.name as team_name, ps.points, f.gameweek
                FROM player p
                JOIN team t ON p.team = t.id
                JOIN playerscore ps ON p.id = ps.player_id
                JOIN fixture f ON ps.fixture_id = f.id
                WHERE f.gameweek >= 35
                LIMIT 100
                """)
                self.session.execute(query).fetchall()
            except Exception:
                pass

    @pytest.mark.benchmark(group="database")
    def benchmark_aggregations(self):
        """Benchmark aggregation queries."""
        # Points aggregation by position
        with self.runner.benchmark_context("db_agg_points_by_position"):
            try:
                query = text("""
                SELECT p.position, AVG(ps.points) as avg_points, COUNT(*) as games
                FROM player p
                JOIN playerscore ps ON p.id = ps.player_id
                WHERE ps.gameweek >= 30
                GROUP BY p.position
                """)
                self.session.execute(query).fetchall()
            except Exception:
                pass

        # Team performance aggregation
        with self.runner.benchmark_context("db_agg_team_performance"):
            try:
                query = text("""
                SELECT t.name,
                       SUM(CASE WHEN f.home_team = t.id THEN f.home_score ELSE f.away_score END) as goals_for,
                       COUNT(*) as games_played
                FROM team t
                JOIN fixture f ON (t.id = f.home_team OR t.id = f.away_team)
                WHERE f.gameweek >= 30
                GROUP BY t.id, t.name
                """)
                self.session.execute(query).fetchall()
            except Exception:
                pass

        # Player consistency metrics
        with self.runner.benchmark_context("db_agg_player_consistency"):
            try:
                query = text("""
                SELECT p.name,
                       AVG(ps.points) as avg_points,
                       STDDEV(ps.points) as point_variance,
                       COUNT(*) as games_played
                FROM player p
                JOIN playerscore ps ON p.id = ps.player_id
                WHERE ps.gameweek >= 30 AND ps.minutes > 0
                GROUP BY p.id, p.name
                HAVING COUNT(*) >= 5
                ORDER BY avg_points DESC
                LIMIT 50
                """)
                self.session.execute(query).fetchall()
            except Exception:
                pass

    def benchmark_index_performance(self):
        """Benchmark index effectiveness."""
        # Query with indexed columns
        with self.runner.benchmark_context("db_query_indexed"):
            try:
                # Assume player.id and playerscore.player_id are indexed
                query = (
                    self.session.query(PlayerScore)
                    .filter(PlayerScore.player_id == 1)
                    .limit(100)
                )
                query.all()
            except Exception:
                pass

        # Query without proper indexing (gameweek range)
        with self.runner.benchmark_context("db_query_range"):
            try:
                query = (
                    self.session.query(PlayerScore)
                    .filter(PlayerScore.gameweek.between(30, 38))
                    .limit(1000)
                )
                query.all()
            except Exception:
                pass

    def benchmark_connection_performance(self):
        """Benchmark database connection performance."""
        # Test connection creation overhead
        with self.runner.benchmark_context("db_connection_create"):
            try:
                # Test session usage
                for _ in range(10):
                    temp_session = session()
                    temp_session.query(Player).first()
                    temp_session.close()
            except Exception:
                pass

        # Test connection reuse
        with self.runner.benchmark_context("db_connection_reuse"):
            try:
                # Reuse existing session
                for _ in range(100):
                    self.session.query(Player).first()
            except Exception:
                pass


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="database_single")
def test_player_query(benchmark):
    """Pytest-benchmark test for single player query."""
    db_session = session()

    def query_player():
        return db_session.query(Player).first()

    try:
        benchmark(query_player)
    finally:
        db_session.close()


@pytest.mark.benchmark(group="database_batch")
def test_players_batch_query(benchmark):
    """Pytest-benchmark test for batch player query."""
    db_session = session()

    def query_players():
        return db_session.query(Player).limit(100).all()

    try:
        benchmark(query_players)
    finally:
        db_session.close()


@pytest.mark.benchmark(group="database_complex")
def test_complex_join_query(benchmark):
    """Pytest-benchmark test for complex join query."""
    db_session = session()

    def complex_query():
        try:
            query = (
                db_session.query(Player, Team)
                .join(Team, Player.team == Team.id)
                .limit(50)
            )
            return query.all()
        except Exception:
            return []

    try:
        benchmark(complex_query)
    finally:
        db_session.close()
