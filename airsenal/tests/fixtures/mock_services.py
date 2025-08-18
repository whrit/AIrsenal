"""
Mock implementations for external services and dependencies.

Provides mock implementations for FPL API, Redis cache, model interfaces,
and other external services to enable isolated testing.
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock

import numpy as np

from .fpl_data import (
    PREMIER_LEAGUE_TEAMS,
    generate_realistic_player_data,
)


class MockFPLDataFetcher:
    """
    Mock implementation of FPL data fetcher for testing.

    Provides realistic FPL API responses without making actual HTTP requests.
    """

    def __init__(self, season: str = "2324"):
        self.session = None
        self.season = season
        self.current_gameweek = random.randint(1, 38)
        self.logged_in = False
        self.fpl_team_id = 742663  # Test team ID

        # Cache for consistent responses
        self._bootstrap_cache = None
        self._fixtures_cache = None
        self._players_cache = None

    def login(self, username: str | None = None, password: str | None = None) -> bool:
        """Mock login functionality."""
        if username and password:
            self.logged_in = True
            return True
        return False

    def get_bootstrap_data(self) -> dict[str, Any]:
        """Mock bootstrap data from FPL API."""
        if self._bootstrap_cache:
            return self._bootstrap_cache

        # Generate realistic bootstrap data
        teams = []
        for i, (code, name) in enumerate(PREMIER_LEAGUE_TEAMS.items(), 1):
            teams.append(
                {
                    "id": i,
                    "name": name,
                    "short_name": code,
                    "code": i,
                    "strength": random.randint(3, 5),
                    "strength_overall_home": random.randint(1000, 1400),
                    "strength_overall_away": random.randint(1000, 1400),
                    "strength_attack_home": random.randint(1000, 1400),
                    "strength_attack_away": random.randint(1000, 1400),
                    "strength_defence_home": random.randint(1000, 1400),
                    "strength_defence_away": random.randint(1000, 1400),
                }
            )

        # Generate players
        elements = []
        for player_id in range(1, 501):  # 500 players
            position = random.choice(["GK", "DEF", "MID", "FWD"])
            team_id = random.randint(1, 20)
            player_data = generate_realistic_player_data(position)

            elements.append(
                {
                    "id": player_id,
                    "web_name": player_data["name"].split()[-1],  # Last name
                    "first_name": player_data["name"].split()[0],
                    "second_name": " ".join(player_data["name"].split()[1:]),
                    "team": team_id,
                    "element_type": {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[position],
                    "now_cost": player_data["price"],
                    "total_points": random.randint(0, 200),
                    "points_per_game": round(random.uniform(0.0, 8.0), 1),
                    "selected_by_percent": round(random.uniform(0.1, 30.0), 1),
                    "form": round(random.uniform(0.0, 10.0), 1),
                    "transfers_in": random.randint(0, 100000),
                    "transfers_out": random.randint(0, 50000),
                    "minutes": random.randint(0, self.current_gameweek * 90),
                    "goals_scored": random.randint(0, 25),
                    "assists": random.randint(0, 20),
                    "clean_sheets": random.randint(0, 15),
                    "goals_conceded": random.randint(0, 40),
                    "own_goals": random.randint(0, 2),
                    "penalties_saved": random.randint(0, 3),
                    "penalties_missed": random.randint(0, 2),
                    "yellow_cards": random.randint(0, 10),
                    "red_cards": random.randint(0, 2),
                    "saves": random.randint(0, 100),
                    "bonus": random.randint(0, 20),
                    "bps": random.randint(0, 500),
                    "influence": round(random.uniform(0.0, 100.0), 1),
                    "creativity": round(random.uniform(0.0, 100.0), 1),
                    "threat": round(random.uniform(0.0, 100.0), 1),
                    "ict_index": round(random.uniform(0.0, 300.0), 1),
                    "chance_of_playing_this_round": random.choice(
                        [None, 0, 25, 50, 75, 100]
                    ),
                    "chance_of_playing_next_round": random.choice(
                        [None, 0, 25, 50, 75, 100]
                    ),
                    "status": random.choice(
                        ["a", "i", "d", "u", "s"]
                    ),  # available, injured, doubtful, unavailable, suspended
                    "news": random.choice(
                        [
                            "",
                            "Knock picked up in training",
                            "Back from injury",
                            "Expected to return soon",
                        ]
                    ),
                    "news_added": datetime.now().isoformat()
                    if random.random() < 0.3
                    else None,
                }
            )

        # Generate element types (positions)
        element_types = [
            {
                "id": 1,
                "plural_name": "Goalkeepers",
                "singular_name": "Goalkeeper",
                "plural_name_short": "GKP",
            },
            {
                "id": 2,
                "plural_name": "Defenders",
                "singular_name": "Defender",
                "plural_name_short": "DEF",
            },
            {
                "id": 3,
                "plural_name": "Midfielders",
                "singular_name": "Midfielder",
                "plural_name_short": "MID",
            },
            {
                "id": 4,
                "plural_name": "Forwards",
                "singular_name": "Forward",
                "plural_name_short": "FWD",
            },
        ]

        # Generate events (gameweeks)
        events = []
        base_date = datetime(2024, 8, 17)  # Season start
        for gw in range(1, 39):
            event_date = base_date + timedelta(weeks=gw - 1)
            events.append(
                {
                    "id": gw,
                    "name": f"Gameweek {gw}",
                    "deadline_time": (
                        event_date + timedelta(days=6, hours=11, minutes=30)
                    ).isoformat(),
                    "average_entry_score": random.randint(35, 70),
                    "finished": gw < self.current_gameweek,
                    "data_checked": gw < self.current_gameweek,
                    "highest_scoring_entry": random.randint(80, 150)
                    if gw < self.current_gameweek
                    else None,
                    "deadline_time_epoch": int(
                        (
                            event_date + timedelta(days=6, hours=11, minutes=30)
                        ).timestamp()
                    ),
                    "deadline_time_game_offset": 0,
                    "highest_score": random.randint(80, 150)
                    if gw < self.current_gameweek
                    else None,
                    "is_previous": gw == self.current_gameweek - 1,
                    "is_current": gw == self.current_gameweek,
                    "is_next": gw == self.current_gameweek + 1,
                    "chip_plays": {
                        "bboost": random.randint(0, 1000000)
                        if gw < self.current_gameweek
                        else 0,
                        "freehit": random.randint(0, 500000)
                        if gw < self.current_gameweek
                        else 0,
                        "wildcard": random.randint(0, 800000)
                        if gw < self.current_gameweek
                        else 0,
                        "3xc": random.randint(0, 300000)
                        if gw < self.current_gameweek
                        else 0,
                    },
                    "most_selected": random.randint(1, 500)
                    if gw < self.current_gameweek
                    else None,
                    "most_transferred_in": random.randint(1, 500)
                    if gw < self.current_gameweek
                    else None,
                    "top_element": random.randint(1, 500)
                    if gw < self.current_gameweek
                    else None,
                    "top_element_info": {
                        "id": random.randint(1, 500),
                        "points": random.randint(10, 25),
                    }
                    if gw < self.current_gameweek
                    else None,
                    "transfers_made": random.randint(2000000, 8000000)
                    if gw < self.current_gameweek
                    else None,
                }
            )

        self._bootstrap_cache = {
            "events": events,
            "game_settings": {
                "league_join_private_max": 20,
                "league_join_public_max": 20,
                "league_max_size_public_classic": 20,
                "league_max_size_public_h2h": 16,
                "league_max_size_private_h2h": 16,
                "league_max_ko_rounds_private_h2h": 3,
                "league_prefix_public": "League",
                "league_points_h2h_win": 3,
                "league_points_h2h_lose": 0,
                "league_points_h2h_draw": 1,
                "league_ko_first_instead_of_random": False,
                "cup_start_event_id": 17,
                "cup_stop_event_id": 38,
                "cup_qualifying_method": "top",
                "cup_type": "knockout",
                "squad_squadplay": 15,
                "squad_squadsize": 15,
                "squad_team_limit": 3,
                "squad_total_spend": 1000,
                "ui_currency_multiplier": 10,
                "ui_use_special_shirts": False,
                "ui_special_shirt_exclusions": [],
                "stats_form_days": 30,
                "sys_vice_captain_enabled": True,
                "transfers_cap": 1,
                "transfers_sell_on_fee": 0.5,
                "league_h2h_tiebreak_stats": ["+goals_scored", "-goals_conceded"],
                "timezone": "UTC",
            },
            "phases": [
                {"id": 1, "name": "Overall", "start_event": 1, "stop_event": 38}
            ],
            "teams": teams,
            "total_players": len(elements),
            "elements": elements,
            "element_stats": [
                {"label": "Minutes played", "name": "minutes"},
                {"label": "Goals scored", "name": "goals_scored"},
                {"label": "Assists", "name": "assists"},
                {"label": "Clean sheets", "name": "clean_sheets"},
                {"label": "Goals conceded", "name": "goals_conceded"},
                {"label": "Own goals", "name": "own_goals"},
                {"label": "Penalties saved", "name": "penalties_saved"},
                {"label": "Penalties missed", "name": "penalties_missed"},
                {"label": "Yellow cards", "name": "yellow_cards"},
                {"label": "Red cards", "name": "red_cards"},
                {"label": "Saves", "name": "saves"},
                {"label": "Bonus", "name": "bonus"},
                {"label": "Bonus Points System", "name": "bps"},
                {"label": "Influence", "name": "influence"},
                {"label": "Creativity", "name": "creativity"},
                {"label": "Threat", "name": "threat"},
                {"label": "ICT Index", "name": "ict_index"},
            ],
            "element_types": element_types,
        }

        return self._bootstrap_cache

    def get_fixtures(self, gameweek: int | None = None) -> list[dict[str, Any]]:
        """Mock fixture data from FPL API."""
        if self._fixtures_cache and not gameweek:
            return self._fixtures_cache

        fixtures = []
        fixture_id = 1

        gameweeks = [gameweek] if gameweek else range(1, 39)

        for gw in gameweeks:
            # 10 fixtures per gameweek
            teams = list(range(1, 21))
            random.shuffle(teams)

            for i in range(0, 20, 2):
                home_team = teams[i]
                away_team = teams[i + 1]

                # Generate fixture date
                base_date = datetime(2024, 8, 17) + timedelta(weeks=gw - 1)
                fixture_date = base_date + timedelta(days=random.randint(0, 6))

                fixture = {
                    "code": fixture_id * 1000 + gw,
                    "event": gw,
                    "finished": gw < self.current_gameweek,
                    "finished_provisional": gw < self.current_gameweek,
                    "id": fixture_id,
                    "kickoff_time": fixture_date.isoformat(),
                    "minutes": 90 if gw < self.current_gameweek else 0,
                    "provisional_start_time": False,
                    "started": gw < self.current_gameweek,
                    "team_a": away_team,
                    "team_a_score": random.randint(0, 4)
                    if gw < self.current_gameweek
                    else None,
                    "team_h": home_team,
                    "team_h_score": random.randint(0, 4)
                    if gw < self.current_gameweek
                    else None,
                    "stats": [],  # Would contain detailed player stats
                    "team_h_difficulty": random.randint(2, 4),
                    "team_a_difficulty": random.randint(2, 4),
                    "pulse_id": fixture_id * 100,
                }

                # Add stats for finished fixtures
                if gw < self.current_gameweek:
                    fixture["stats"] = self._generate_fixture_stats()

                fixtures.append(fixture)
                fixture_id += 1

        if not gameweek:
            self._fixtures_cache = fixtures

        return fixtures

    def get_player_data(self, player_id: int) -> dict[str, Any]:
        """Mock detailed player data."""
        bootstrap = self.get_bootstrap_data()
        player = next((p for p in bootstrap["elements"] if p["id"] == player_id), None)

        if not player:
            msg = f"Player {player_id} not found"
            raise ValueError(msg)

        # Generate detailed player data
        fixtures = []
        history = []

        # Generate fixture list for player
        for gw in range(self.current_gameweek, min(self.current_gameweek + 5, 39)):
            fixture_data = {
                "id": gw * 100 + player_id,
                "code": gw * 1000 + 1,
                "team_h": random.randint(1, 20),
                "team_a": random.randint(1, 20),
                "event": gw,
                "finished": False,
                "minutes": 0,
                "provisional_start_time": False,
                "kickoff_time": (
                    datetime.now() + timedelta(days=(gw - self.current_gameweek) * 7)
                ).isoformat(),
                "event_name": f"Gameweek {gw}",
                "is_home": random.choice([True, False]),
                "difficulty": random.randint(2, 4),
            }
            fixtures.append(fixture_data)

        # Generate historical performance
        for gw in range(1, self.current_gameweek):
            history_entry = {
                "element": player_id,
                "fixture": gw * 100 + player_id,
                "opponent_team": random.randint(1, 20),
                "total_points": random.randint(0, 15),
                "was_home": random.choice([True, False]),
                "kickoff_time": (
                    datetime(2024, 8, 17) + timedelta(weeks=gw - 1)
                ).isoformat(),
                "team_h_score": random.randint(0, 4),
                "team_a_score": random.randint(0, 4),
                "round": gw,
                "minutes": random.randint(0, 90),
                "goals_scored": random.randint(0, 3),
                "assists": random.randint(0, 2),
                "clean_sheets": random.randint(0, 1),
                "goals_conceded": random.randint(0, 3),
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": random.randint(0, 1),
                "red_cards": 0,
                "saves": random.randint(0, 8),
                "bonus": random.randint(0, 3),
                "bps": random.randint(0, 50),
                "influence": round(random.uniform(0.0, 50.0), 1),
                "creativity": round(random.uniform(0.0, 50.0), 1),
                "threat": round(random.uniform(0.0, 50.0), 1),
                "ict_index": round(random.uniform(0.0, 150.0), 1),
                "value": player["now_cost"] + random.randint(-3, 3),
                "transfers_balance": random.randint(-100000, 100000),
                "selected": random.randint(100000, 3000000),
                "transfers_in": random.randint(0, 200000),
                "transfers_out": random.randint(0, 150000),
            }
            history.append(history_entry)

        return {
            "fixtures": fixtures,
            "history": history,
            "history_past": [],  # Previous seasons data
        }

    def get_fpl_team_data(
        self, team_id: int, gameweek: int | None = None
    ) -> dict[str, Any]:
        """Mock FPL team data."""
        if not self.logged_in:
            msg = "Must be logged in to access team data"
            raise ValueError(msg)

        gw = gameweek or self.current_gameweek

        # Generate squad
        picks = []
        for i in range(15):  # 15 players in squad
            picks.append(
                {
                    "element": random.randint(1, 500),
                    "position": i + 1,
                    "multiplier": 2
                    if i == 0
                    else (1 if i < 11 else 0),  # Captain gets 2x, bench gets 0
                    "is_captain": i == 0,
                    "is_vice_captain": i == 1,
                }
            )

        return {
            "active_chip": None,
            "automatic_subs": [],
            "entry_history": {
                "event": gw,
                "points": random.randint(30, 80),
                "total_points": random.randint(gw * 30, gw * 70),
                "rank": random.randint(10000, 5000000),
                "rank_sort": random.randint(10000, 5000000),
                "overall_rank": random.randint(10000, 5000000),
                "bank": random.randint(0, 50),  # 0.0 to 5.0 million in bank
                "value": random.randint(950, 1050),  # 95.0 to 105.0 squad value
                "event_transfers": random.randint(0, 2),
                "event_transfers_cost": random.randint(0, 8),
                "points_on_bench": random.randint(0, 20),
            },
            "picks": picks,
        }

    def _generate_fixture_stats(self) -> list[dict[str, Any]]:
        """Generate realistic fixture statistics."""
        stat_types = [
            "goals_scored",
            "assists",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
        ]

        stats = []
        for stat_type in stat_types:
            # Generate stats for both teams
            home_players = [
                {"element": random.randint(1, 500), "value": random.randint(0, 3)}
                for _ in range(random.randint(0, 4))
            ]
            away_players = [
                {"element": random.randint(1, 500), "value": random.randint(0, 3)}
                for _ in range(random.randint(0, 4))
            ]

            stats.append(
                {
                    "identifier": stat_type,
                    "a": away_players,
                    "h": home_players,
                }
            )

        return stats


class MockRedisCache:
    """
    Mock Redis cache implementation for testing.

    Provides in-memory cache that mimics Redis behavior without requiring
    a Redis server.
    """

    def __init__(self):
        self._cache = {}
        self._ttl = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }

    def get(self, key: str) -> bytes | None:
        """Get value from cache."""
        self._cleanup_expired()

        if key in self._cache:
            self._stats["hits"] += 1
            return (
                self._cache[key].encode()
                if isinstance(self._cache[key], str)
                else self._cache[key]
            )
        self._stats["misses"] += 1
        return None

    def set(self, key: str, value: str | bytes, ex: int | None = None) -> bool:
        """Set value in cache with optional expiration."""
        self._cache[key] = value.decode() if isinstance(value, bytes) else value

        if ex:
            self._ttl[key] = time.time() + ex
        elif key in self._ttl:
            del self._ttl[key]

        self._stats["sets"] += 1
        return True

    def delete(self, key: str) -> int:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._ttl:
                del self._ttl[key]
            self._stats["deletes"] += 1
            return 1
        return 0

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        self._cleanup_expired()
        return key in self._cache

    def ttl(self, key: str) -> int:
        """Get time to live for key."""
        if key not in self._cache:
            return -2  # Key doesn't exist

        if key not in self._ttl:
            return -1  # Key has no expiration

        remaining = self._ttl[key] - time.time()
        return int(remaining) if remaining > 0 else -2

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        if key in self._cache:
            self._ttl[key] = time.time() + seconds
            return True
        return False

    def flushall(self) -> bool:
        """Clear all cache data."""
        self._cache.clear()
        self._ttl.clear()
        return True

    def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching pattern."""
        self._cleanup_expired()

        if pattern == "*":
            return list(self._cache.keys())

        # Simple pattern matching (only supports * wildcard)
        import fnmatch

        return [key for key in self._cache if fnmatch.fnmatch(key, pattern)]

    def info(self, section: str | None = None) -> dict[str, Any]:
        """Get cache information."""
        info_data = {
            "memory": {
                "used_memory": len(str(self._cache)),
                "used_memory_human": f"{len(str(self._cache))}B",
            },
            "stats": self._stats,
            "keyspace": {
                "db0": {
                    "keys": len(self._cache),
                    "expires": len(self._ttl),
                }
            },
        }

        if section:
            return info_data.get(section, {})
        return info_data

    def _cleanup_expired(self):
        """Remove expired keys."""
        current_time = time.time()
        expired_keys = [key for key, ttl in self._ttl.items() if ttl <= current_time]

        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            del self._ttl[key]


class MockPlayerModel:
    """
    Mock player model implementation for testing base model interfaces.
    """

    def __init__(self, position: str = "MID"):
        self.position = position
        self.player_ids = np.array(range(1, 101))  # 100 players
        self.is_fitted = False
        self.prediction_cache = {}

        # Mock model parameters
        self.weights = np.random.random(10)  # 10 features
        self.bias = random.uniform(-1, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockPlayerModel":
        """Mock model fitting."""
        if X.shape[0] != y.shape[0]:
            msg = "X and y must have same number of samples"
            raise ValueError(msg)

        # Simulate training time
        time.sleep(0.1)

        # Update weights based on input (simplified)
        self.weights = np.random.random(X.shape[1])
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction."""
        if not self.is_fitted:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        # Simple linear prediction with noise
        predictions = np.dot(X, self.weights) + self.bias
        predictions += np.random.normal(0, 0.1, predictions.shape)

        # Ensure predictions are reasonable for FPL points
        return np.clip(predictions, 0, 20)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mock probability prediction."""
        if not self.is_fitted:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        # Generate probability distributions for points (0-20)
        n_samples = X.shape[0]
        n_classes = 21  # 0 to 20 points

        # Generate realistic probability distributions
        return np.random.dirichlet(np.ones(n_classes), size=n_samples)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted to get feature importance"
            raise ValueError(msg)

        feature_names = [
            "minutes",
            "form",
            "fixture_difficulty",
            "team_strength",
            "xg",
            "xa",
            "shots",
            "key_passes",
            "opponent_strength",
            "home_advantage",
        ]

        importance = np.abs(self.weights) / np.sum(np.abs(self.weights))

        return dict(zip(feature_names, importance, strict=False))

    def save_model(self, filepath: str) -> None:
        """Mock model saving."""
        model_data = {
            "position": self.position,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "is_fitted": self.is_fitted,
            "player_ids": self.player_ids.tolist(),
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath: str) -> "MockPlayerModel":
        """Mock model loading."""
        with open(filepath) as f:
            model_data = json.load(f)

        model = cls(position=model_data["position"])
        model.weights = np.array(model_data["weights"])
        model.bias = model_data["bias"]
        model.is_fitted = model_data["is_fitted"]
        model.player_ids = np.array(model_data["player_ids"])

        return model


class MockLogHandler:
    """
    Mock log handler for testing structured logging.
    """

    def __init__(self):
        self.logs = []
        self.log_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0}

    def emit(self, record):
        """Capture log record."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add structured fields if present
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "operation"):
            log_entry["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        self.logs.append(log_entry)
        self.log_counts[record.levelname] += 1

    def get_logs(self, level: str | None = None) -> list[dict[str, Any]]:
        """Get captured logs, optionally filtered by level."""
        if level:
            return [log for log in self.logs if log["level"] == level]
        return self.logs.copy()

    def clear_logs(self):
        """Clear all captured logs."""
        self.logs.clear()
        self.log_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0}

    def get_log_counts(self) -> dict[str, int]:
        """Get counts of logs by level."""
        return self.log_counts.copy()


class MockFPLManager:
    """
    Mock FPL manager for testing team management operations.
    """

    def __init__(self, team_id: int = 742663):
        self.team_id = team_id
        self.current_squad = self._generate_initial_squad()
        self.bank = random.randint(0, 50)  # 0.0 to 5.0 million
        self.free_transfers = 1
        self.gameweek = random.randint(1, 38)
        self.total_points = random.randint(500, 2000)
        self.chips_available = ["wildcard", "freehit", "bboost", "3xc"]
        self.chips_used = []

    def _generate_initial_squad(self) -> list[dict[str, Any]]:
        """Generate initial 15-player squad."""
        squad = []

        # Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD
        positions = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]

        for i, position in enumerate(positions):
            player = {
                "element": random.randint(1, 500),
                "position": i + 1,
                "multiplier": 2 if i == 0 else (1 if i < 11 else 0),
                "is_captain": i == 0,
                "is_vice_captain": i == 1,
                "element_type": position,
                "selling_price": random.randint(40, 120),
                "purchase_price": random.randint(40, 120),
            }
            squad.append(player)

        return squad

    def make_transfer(self, player_out: int, player_in: int) -> dict[str, Any]:
        """Mock transfer operation."""
        # Find player to remove
        player_out_data = next(
            (p for p in self.current_squad if p["element"] == player_out), None
        )
        if not player_out_data:
            msg = f"Player {player_out} not in squad"
            raise ValueError(msg)

        # Mock player prices
        selling_price = random.randint(40, 120)
        buying_price = random.randint(40, 120)

        # Check if affordable
        if buying_price > selling_price + self.bank:
            msg = "Insufficient funds for transfer"
            raise ValueError(msg)

        # Update squad
        player_out_data["element"] = player_in
        player_out_data["selling_price"] = buying_price

        # Update bank
        self.bank += selling_price - buying_price

        # Update free transfers
        if self.free_transfers > 0:
            self.free_transfers -= 1
            cost = 0
        else:
            cost = 4  # 4 point hit

        return {
            "player_in": player_in,
            "player_out": player_out,
            "selling_price": selling_price,
            "buying_price": buying_price,
            "cost": cost,
            "remaining_bank": self.bank,
            "free_transfers_remaining": self.free_transfers,
        }

    def set_captain(self, player_id: int) -> bool:
        """Set team captain."""
        # Reset current captain
        for player in self.current_squad:
            if player["is_captain"]:
                player["is_captain"] = False
                player["multiplier"] = 1 if player["multiplier"] > 0 else 0

        # Set new captain
        captain = next(
            (p for p in self.current_squad if p["element"] == player_id), None
        )
        if captain:
            captain["is_captain"] = True
            captain["multiplier"] = 2 if captain["multiplier"] > 0 else 0
            return True

        return False

    def play_chip(self, chip: str) -> dict[str, Any]:
        """Play a chip."""
        if chip not in self.chips_available:
            msg = f"Chip {chip} not available"
            raise ValueError(msg)

        self.chips_available.remove(chip)
        self.chips_used.append(chip)

        chip_effects = {
            "wildcard": "Free transfers for this gameweek",
            "freehit": "One-week temporary squad",
            "bboost": "All 15 players score points",
            "3xc": "Captain gets triple points",
        }

        return {
            "chip": chip,
            "effect": chip_effects.get(chip, "Unknown chip effect"),
            "gameweek": self.gameweek,
        }

    def get_team_summary(self) -> dict[str, Any]:
        """Get team summary."""
        return {
            "team_id": self.team_id,
            "current_gameweek": self.gameweek,
            "total_points": self.total_points,
            "bank": self.bank,
            "team_value": sum(p["selling_price"] for p in self.current_squad)
            + self.bank,
            "free_transfers": self.free_transfers,
            "squad_size": len(self.current_squad),
            "chips_available": self.chips_available.copy(),
            "chips_used": self.chips_used.copy(),
        }


# Factory functions for creating mock services
def create_mock_fpl_fetcher(
    season: str = "2324", current_gameweek: int | None = None
) -> MockFPLDataFetcher:
    """Create a configured mock FPL data fetcher."""
    fetcher = MockFPLDataFetcher(season)
    if current_gameweek:
        fetcher.current_gameweek = current_gameweek
    return fetcher


def create_mock_redis_cache(
    preload_data: dict[str, Any] | None = None,
) -> MockRedisCache:
    """Create a mock Redis cache with optional preloaded data."""
    cache = MockRedisCache()

    if preload_data:
        for key, value in preload_data.items():
            cache.set(key, value)

    return cache


def create_mock_player_models(
    positions: list[str] | None = None,
) -> dict[str, MockPlayerModel]:
    """Create mock player models for different positions."""
    if positions is None:
        positions = ["GK", "DEF", "MID", "FWD"]

    models = {}
    for position in positions:
        model = MockPlayerModel(position)

        # Generate some training data and fit the model
        X = np.random.random((100, 10))  # 100 samples, 10 features
        y = np.random.uniform(0, 20, 100)  # Points between 0-20
        model.fit(X, y)

        models[position] = model

    return models


def create_mock_log_scenario() -> dict[str, Any]:
    """Create a scenario with various log entries for testing."""
    handler = MockLogHandler()

    # Generate sample logs
    log_scenarios = [
        {
            "level": "INFO",
            "message": "Player prediction completed",
            "operation": "predict",
            "duration_ms": 150,
        },
        {
            "level": "WARNING",
            "message": "High fixture difficulty detected",
            "operation": "analyze",
            "duration_ms": 50,
        },
        {
            "level": "ERROR",
            "message": "Failed to fetch player data",
            "operation": "fetch",
            "error_code": "API_ERROR",
        },
        {
            "level": "DEBUG",
            "message": "Cache hit for player stats",
            "operation": "cache",
            "cache_key": "player_123",
        },
        {
            "level": "INFO",
            "message": "Transfer optimization completed",
            "operation": "optimize",
            "duration_ms": 2500,
        },
    ]

    # Mock log records
    for scenario in log_scenarios:
        record = Mock()
        record.levelname = scenario["level"]
        record.getMessage.return_value = scenario["message"]
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 42

        # Add additional attributes
        for key, value in scenario.items():
            if key not in ["level", "message"]:
                setattr(record, key, value)

        handler.emit(record)

    return {
        "handler": handler,
        "logs": handler.get_logs(),
        "counts": handler.get_log_counts(),
    }
