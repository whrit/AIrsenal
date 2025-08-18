"""
AIrsenal Feature Store

A centralized feature engineering and serving system for FPL prediction models.
Supports both online (real-time) and offline (batch) feature serving with
sophisticated caching, versioning, and time-series capabilities.

Key Features:
- Rolling window computations (moving averages, trends, percentiles)
- Multi-level caching (memory + database) for sub-second serving
- Feature versioning and schema evolution
- Time-series storage for historical analysis
- Integration with existing SQLAlchemy models
- Validation and monitoring capabilities

Usage:
    store = FeatureStore()

    # Register a new feature
    store.register_feature(
        name="rolling_goals_5",
        feature_type="player",
        computation_logic={"window": 5, "metric": "goals", "agg": "mean"}
    )

    # Get features for players
    features = store.get_features(
        entity_type="player",
        entity_ids=[123, 456],
        feature_names=["rolling_goals_5", "xg_form"],
        season="2425",
        gameweek=10
    )

    # Batch compute features
    store.compute_features_batch(
        feature_names=["rolling_goals_5"],
        season="2425",
        gameweek_range=(1, 38)
    )
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from airsenal.framework.schema import (
    ComputedFeature,
    FeatureCache,
    FeatureDefinition,
    FeatureTimeSeries,
    Fixture,
    Player,
    PlayerScore,
    session,
)
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK

logger = logging.getLogger(__name__)


class FeatureComputationError(Exception):
    """Raised when feature computation fails."""


class FeatureNotFoundError(Exception):
    """Raised when a requested feature is not found in the registry."""


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""


class FeatureRegistry:
    """Manages feature definitions and versioning."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession

    def register_feature(
        self,
        name: str,
        feature_type: str,
        data_type: str = "float",
        version: str = "1.0.0",
        description: str | None = None,
        computation_logic: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        replace_existing: bool = False,
    ) -> FeatureDefinition:
        """Register a new feature or update existing one."""

        # Check if feature already exists
        existing = (
            self.dbsession.query(FeatureDefinition)
            .filter_by(name=name, version=version)
            .first()
        )

        if existing and not replace_existing:
            msg = f"Feature {name}:{version} already exists. Use replace_existing=True to update."
            raise ValueError(msg)

        if existing and replace_existing:
            # Update existing feature
            existing.feature_type = feature_type
            existing.data_type = data_type
            existing.description = description
            existing.computation_logic = (
                json.dumps(computation_logic) if computation_logic else None
            )
            existing.dependencies = ",".join(dependencies) if dependencies else None
            existing.updated_at = datetime.now().isoformat()
            feature_def = existing
        else:
            # Create new feature
            feature_def = FeatureDefinition(
                name=name,
                version=version,
                feature_type=feature_type,
                data_type=data_type,
                description=description,
                computation_logic=json.dumps(computation_logic)
                if computation_logic
                else None,
                dependencies=",".join(dependencies) if dependencies else None,
                is_active=True,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            self.dbsession.add(feature_def)

        self.dbsession.commit()
        logger.info("Registered feature: %s:%s", name, version)
        return feature_def

    def get_feature_definition(
        self, name: str, version: str | None = None
    ) -> FeatureDefinition:
        """Get feature definition by name and optional version."""
        query = self.dbsession.query(FeatureDefinition).filter_by(
            name=name, is_active=True
        )

        if version:
            query = query.filter_by(version=version)
        else:
            # Get latest version
            query = query.order_by(desc(FeatureDefinition.created_at))

        feature_def = query.first()
        if not feature_def:
            msg = f"Feature {name}:{version or 'latest'} not found"
            raise FeatureNotFoundError(msg)

        return feature_def

    def list_features(self, feature_type: str | None = None) -> list[FeatureDefinition]:
        """List all active features, optionally filtered by type."""
        query = self.dbsession.query(FeatureDefinition).filter_by(is_active=True)

        if feature_type:
            query = query.filter_by(feature_type=feature_type)

        return query.all()


class FeatureCacheManager:
    """Manages multi-level caching for feature values with Redis integration."""

    def __init__(self, dbsession: Session = session, default_ttl: int = 3600):
        self.dbsession = dbsession
        self.default_ttl = default_ttl  # seconds
        self._memory_cache: dict[str, Any] = {}  # In-memory cache
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "redis_hits": 0,
            "db_hits": 0,
        }

        # Initialize Redis cache integration
        try:
            from airsenal.framework.redis_cache import redis_cache

            self.redis_cache = redis_cache
            self.redis_enabled = redis_cache.is_available()
        except ImportError:
            self.redis_cache = None
            self.redis_enabled = False
            logger.warning("Redis cache not available - using memory + database only")

    def _make_cache_key(
        self, feature_name: str, entity_type: str, entity_id: int, context: str = ""
    ) -> str:
        """Create a cache key from components."""
        return f"{feature_name}:{entity_type}:{entity_id}:{context}"

    def get(
        self, feature_name: str, entity_type: str, entity_id: int, context: str = ""
    ) -> float | str | None:
        """Get cached feature value with memory -> Redis -> database cache hierarchy."""
        cache_key = self._make_cache_key(feature_name, entity_type, entity_id, context)

        # 1. Check memory cache first (fastest)
        if cache_key in self._memory_cache:
            value, expires_at = self._memory_cache[cache_key]
            if datetime.now() < expires_at:
                self._cache_stats["hits"] += 1
                self._cache_stats["memory_hits"] += 1
                return value
            # Expired, remove from memory cache
            del self._memory_cache[cache_key]

        # 2. Check Redis cache (fast)
        if self.redis_enabled:
            try:
                redis_value = self.redis_cache.get_feature_cache(
                    feature_name=feature_name,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    context={"context": context} if context else None,
                )

                if redis_value is not None:
                    # Cache hit in Redis - refresh memory cache
                    expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
                    self._memory_cache[cache_key] = (redis_value, expires_at)
                    self._cache_stats["hits"] += 1
                    self._cache_stats["redis_hits"] += 1
                    return redis_value

            except Exception as e:
                logger.warning("Redis cache error: %s", e)

        # 3. Check database cache (slowest)
        cache_entry = (
            self.dbsession.query(FeatureCache).filter_by(cache_key=cache_key).first()
        )

        if cache_entry:
            expires_at = datetime.fromisoformat(cache_entry.expires_at)
            if datetime.now() < expires_at:
                # Update hit count and refresh both memory and Redis caches
                cache_entry.hit_count += 1
                self.dbsession.commit()

                value = (
                    cache_entry.value
                    if cache_entry.value is not None
                    else cache_entry.string_value
                )

                # Refresh memory cache
                self._memory_cache[cache_key] = (value, expires_at)

                # Refresh Redis cache
                if self.redis_enabled:
                    try:
                        remaining_ttl = int(
                            (expires_at - datetime.now()).total_seconds()
                        )
                        if remaining_ttl > 0:
                            self.redis_cache.set_feature_cache(
                                feature_name=feature_name,
                                entity_type=entity_type,
                                entity_id=entity_id,
                                value=value,
                                context={"context": context} if context else None,
                                ttl=remaining_ttl,
                            )
                    except Exception as e:
                        logger.warning("Redis cache set error: %s", e)

                self._cache_stats["hits"] += 1
                self._cache_stats["db_hits"] += 1
                return value
            # Expired, remove from database
            self.dbsession.delete(cache_entry)
            self.dbsession.commit()

        self._cache_stats["misses"] += 1
        return None

    def set(
        self,
        feature_name: str,
        entity_type: str,
        entity_id: int,
        value: float | str,
        ttl: int | None = None,
        context: str = "",
    ) -> None:
        """Cache feature value in memory, Redis, and database."""
        cache_key = self._make_cache_key(feature_name, entity_type, entity_id, context)
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        # 1. Update memory cache (fastest)
        self._memory_cache[cache_key] = (value, expires_at)

        # 2. Update Redis cache (fast)
        if self.redis_enabled:
            try:
                self.redis_cache.set_feature_cache(
                    feature_name=feature_name,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    value=value,
                    context={"context": context} if context else None,
                    ttl=ttl,
                )
            except Exception as e:
                logger.warning("Redis cache set error: %s", e)

        # 3. Update database cache (persistent)
        cache_entry = (
            self.dbsession.query(FeatureCache).filter_by(cache_key=cache_key).first()
        )

        if cache_entry:
            cache_entry.value = value if isinstance(value, int | float) else None
            cache_entry.string_value = (
                str(value) if not isinstance(value, int | float) else None
            )
            cache_entry.cached_at = datetime.now().isoformat()
            cache_entry.expires_at = expires_at.isoformat()
        else:
            cache_entry = FeatureCache(
                cache_key=cache_key,
                feature_name=feature_name,
                entity_type=entity_type,
                entity_id=entity_id,
                value=value if isinstance(value, int | float) else None,
                string_value=str(value) if not isinstance(value, int | float) else None,
                cached_at=datetime.now().isoformat(),
                expires_at=expires_at.isoformat(),
                hit_count=0,
            )
            self.dbsession.add(cache_entry)

        self.dbsession.commit()

    def invalidate(
        self,
        feature_name: str,
        entity_type: str | None = None,
        entity_id: int | None = None,
    ) -> int:
        """Invalidate cached values matching the criteria from all cache levels."""
        total_invalidated = 0

        # 1. Clear from memory cache
        keys_to_remove = []
        for key in self._memory_cache:
            if key.startswith(f"{feature_name}:"):
                if entity_type is None or f":{entity_type}:" in key:
                    if entity_id is None or f":{entity_id}:" in key:
                        keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._memory_cache[key]
        total_invalidated += len(keys_to_remove)

        # 2. Clear from Redis cache
        if self.redis_enabled:
            try:
                if entity_type and entity_id:
                    # Specific entity invalidation
                    redis_count = (
                        self.redis_cache.invalidate_player_cache(entity_id)
                        if entity_type == "player"
                        else 0
                    )
                else:
                    # Pattern-based invalidation for feature name
                    pattern = f"{self.redis_cache.namespace}:{self.redis_cache.version}:feat:*:{feature_name}*"
                    redis_count = self.redis_cache.delete_pattern(pattern)
                total_invalidated += redis_count
            except Exception as e:
                logger.warning("Redis cache invalidation error: %s", e)

        # 3. Clear from database cache
        query = self.dbsession.query(FeatureCache).filter_by(feature_name=feature_name)
        if entity_type:
            query = query.filter_by(entity_type=entity_type)
        if entity_id:
            query = query.filter_by(entity_id=entity_id)

        db_count = query.count()
        query.delete()
        self.dbsession.commit()
        total_invalidated += db_count

        logger.info(
            "Invalidated %s cache entries for %s", total_invalidated, feature_name
        )
        return total_invalidated

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics across all levels."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        # Calculate hit rates by cache level
        memory_hit_rate = (
            self._cache_stats["memory_hits"] / total_requests
            if total_requests > 0
            else 0
        )
        redis_hit_rate = (
            self._cache_stats["redis_hits"] / total_requests
            if total_requests > 0
            else 0
        )
        db_hit_rate = (
            self._cache_stats["db_hits"] / total_requests if total_requests > 0 else 0
        )

        # Get database cache stats
        db_stats = self.dbsession.query(
            func.count(FeatureCache.id).label("total_entries"),
            func.avg(FeatureCache.hit_count).label("avg_hits"),
        ).first()

        stats = {
            "overall": {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "total_hits": self._cache_stats["hits"],
                "total_misses": self._cache_stats["misses"],
            },
            "memory_cache": {
                "entries": len(self._memory_cache),
                "hit_rate": memory_hit_rate,
                "hits": self._cache_stats["memory_hits"],
            },
            "database_cache": {
                "total_entries": db_stats.total_entries or 0,
                "average_hits": float(db_stats.avg_hits or 0),
                "hit_rate": db_hit_rate,
                "hits": self._cache_stats["db_hits"],
            },
        }

        # Add Redis stats if available
        if self.redis_enabled and self.redis_cache:
            try:
                redis_metrics = self.redis_cache.get_metrics()
                stats["redis_cache"] = {
                    "available": redis_metrics["available"],
                    "hit_rate": redis_hit_rate,
                    "hits": self._cache_stats["redis_hits"],
                    "total_size_mb": redis_metrics.get("total_size_mb", 0),
                    "avg_retrieval_time_ms": redis_metrics.get(
                        "avg_retrieval_time_ms", 0
                    ),
                    "connection_status": redis_metrics.get(
                        "connection_status", "unknown"
                    ),
                    "compression": redis_metrics.get("compression", "none"),
                }

                # Add Redis server info if available
                redis_info = self.redis_cache.get_info()
                if not redis_info.get("error"):
                    stats["redis_cache"]["server_info"] = redis_info

            except Exception as e:
                stats["redis_cache"] = {"available": False, "error": str(e)}
        else:
            stats["redis_cache"] = {
                "available": False,
                "reason": "Redis not enabled or not available",
            }

        return stats


class FeatureComputer:
    """Handles feature computation including rolling windows and time-series analysis."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession

    def compute_rolling_statistic(
        self,
        entity_type: str,
        entity_id: int,
        metric_name: str,
        window_size: int,
        aggregation: str = "mean",
        season: str = CURRENT_SEASON,
        gameweek: int = NEXT_GAMEWEEK,
        min_periods: int = 1,
    ) -> float | None:
        """Compute rolling statistics over a sliding window."""

        if entity_type == "player" and metric_name in [
            "goals",
            "assists",
            "minutes",
            "points",
        ]:
            # Query PlayerScore for historical data
            query = (
                self.dbsession.query(PlayerScore)
                .join(Fixture)
                .filter(
                    PlayerScore.player_id == entity_id,
                    Fixture.season == season,
                    Fixture.gameweek < gameweek,
                )
                .order_by(desc(Fixture.gameweek))
                .limit(window_size)
            )

            scores = query.all()

            if len(scores) < min_periods:
                return None

            values = [
                getattr(score, metric_name)
                for score in scores
                if getattr(score, metric_name) is not None
            ]

            if len(values) < min_periods:
                return None

            # Apply aggregation function
            if aggregation == "mean":
                return float(np.mean(values))
            if aggregation == "sum":
                return float(np.sum(values))
            if aggregation == "std":
                return float(np.std(values))
            if aggregation == "median":
                return float(np.median(values))
            if aggregation == "min":
                return float(np.min(values))
            if aggregation == "max":
                return float(np.max(values))
            if aggregation == "q25":
                return float(np.percentile(values, 25))
            if aggregation == "q75":
                return float(np.percentile(values, 75))
            msg = f"Unsupported aggregation: {aggregation}"
            raise ValueError(msg)

        # Use FeatureTimeSeries for other metrics
        query = (
            self.dbsession.query(FeatureTimeSeries)
            .filter(
                FeatureTimeSeries.entity_type == entity_type,
                FeatureTimeSeries.entity_id == entity_id,
                FeatureTimeSeries.metric_name == metric_name,
                FeatureTimeSeries.season == season,
                FeatureTimeSeries.gameweek < gameweek,
            )
            .order_by(desc(FeatureTimeSeries.gameweek))
            .limit(window_size)
        )

        series_data = query.all()

        if len(series_data) < min_periods:
            return None

        values = [data.value for data in series_data]

        # Apply aggregation (same logic as above)
        if aggregation == "mean":
            return float(np.mean(values))
        if aggregation == "sum":
            return float(np.sum(values))
        if aggregation == "std":
            return float(np.std(values))
        if aggregation == "median":
            return float(np.median(values))
        if aggregation == "min":
            return float(np.min(values))
        if aggregation == "max":
            return float(np.max(values))
        if aggregation == "q25":
            return float(np.percentile(values, 25))
        if aggregation == "q75":
            return float(np.percentile(values, 75))
        msg = f"Unsupported aggregation: {aggregation}"
        raise ValueError(msg)

    def compute_form_metric(
        self,
        entity_type: str,
        entity_id: int,
        metric_name: str,
        season: str = CURRENT_SEASON,
        gameweek: int = NEXT_GAMEWEEK,
        decay_factor: float = 0.9,
    ) -> float | None:
        """Compute exponentially weighted form metric."""

        # Get historical data points
        if entity_type == "player" and metric_name in [
            "goals",
            "assists",
            "minutes",
            "points",
        ]:
            query = (
                self.dbsession.query(PlayerScore)
                .join(Fixture)
                .filter(
                    PlayerScore.player_id == entity_id,
                    Fixture.season == season,
                    Fixture.gameweek < gameweek,
                )
                .order_by(desc(Fixture.gameweek))
                .limit(10)
            )  # Last 10 games for form

            scores = query.all()
            values = [
                getattr(score, metric_name)
                for score in scores
                if getattr(score, metric_name) is not None
            ]
        else:
            query = (
                self.dbsession.query(FeatureTimeSeries)
                .filter(
                    FeatureTimeSeries.entity_type == entity_type,
                    FeatureTimeSeries.entity_id == entity_id,
                    FeatureTimeSeries.metric_name == metric_name,
                    FeatureTimeSeries.season == season,
                    FeatureTimeSeries.gameweek < gameweek,
                )
                .order_by(desc(FeatureTimeSeries.gameweek))
                .limit(10)
            )

            series_data = query.all()
            values = [data.value for data in series_data]

        if not values:
            return None

        # Compute exponentially weighted average
        weights = np.array([decay_factor**i for i in range(len(values))])
        weighted_values = np.array(values) * weights

        return float(weighted_values.sum() / weights.sum())

    def compute_feature_by_logic(
        self,
        feature_def: FeatureDefinition,
        entity_type: str,
        entity_id: int,
        season: str = CURRENT_SEASON,
        gameweek: int = NEXT_GAMEWEEK,
    ) -> float | str | None:
        """Compute feature value based on its computation logic."""

        if not feature_def.computation_logic:
            msg = f"No computation logic defined for feature {feature_def.name}"
            raise FeatureComputationError(msg)

        try:
            logic = json.loads(feature_def.computation_logic)
        except json.JSONDecodeError as e:
            msg = f"Invalid computation logic JSON for {feature_def.name}: {e}"
            raise FeatureComputationError(msg)

        computation_type = logic.get("type", "rolling")

        if computation_type == "rolling":
            return self.compute_rolling_statistic(
                entity_type=entity_type,
                entity_id=entity_id,
                metric_name=logic["metric"],
                window_size=logic["window"],
                aggregation=logic.get("agg", "mean"),
                season=season,
                gameweek=gameweek,
                min_periods=logic.get("min_periods", 1),
            )

        if computation_type == "form":
            return self.compute_form_metric(
                entity_type=entity_type,
                entity_id=entity_id,
                metric_name=logic["metric"],
                season=season,
                gameweek=gameweek,
                decay_factor=logic.get("decay_factor", 0.9),
            )

        if computation_type == "static":
            # For static features that don't change over time
            metric = logic["metric"]
            if entity_type == "player" and metric in ["position", "team"]:
                player = (
                    self.dbsession.query(Player).filter_by(player_id=entity_id).first()
                )
                if player:
                    if metric == "position":
                        return player.position(season)
                    if metric == "team":
                        return player.team(season, gameweek)

            return None

        msg = f"Unsupported computation type: {computation_type}"
        raise FeatureComputationError(msg)


class FeatureValidator:
    """Validates feature values and monitors feature quality."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession

    def validate_feature_value(
        self, feature_def: FeatureDefinition, value: float | str | None
    ) -> bool:
        """Validate that a feature value conforms to its definition."""

        # Check for null values
        if value is None:
            return True  # Null values are generally acceptable

        # Check data type compliance
        if feature_def.data_type == "float" and not isinstance(value, int | float):
            logger.warning(
                "Type mismatch for %s: expected float, got %s",
                feature_def.name,
                type(value),
            )
            return False

        if feature_def.data_type == "int" and not isinstance(value, int):
            logger.warning(
                "Type mismatch for %s: expected int, got %s",
                feature_def.name,
                type(value),
            )
            return False

        if feature_def.data_type == "string" and not isinstance(value, str):
            logger.warning(
                "Type mismatch for %s: expected string, got %s",
                feature_def.name,
                type(value),
            )
            return False

        if feature_def.data_type == "boolean" and not isinstance(value, bool):
            logger.warning(
                "Type mismatch for %s: expected boolean, got %s",
                feature_def.name,
                type(value),
            )
            return False

        # Range validation for numeric features
        if isinstance(value, int | float):
            if np.isnan(value) or np.isinf(value):
                logger.warning(
                    "Invalid numeric value for %s: %s", feature_def.name, value
                )
                return False

        return True

    def check_feature_drift(
        self,
        feature_name: str,
        entity_type: str,
        current_season: str,
        reference_season: str,
        drift_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Check for statistical drift in feature values between seasons."""

        # Get current season values
        current_query = (
            self.dbsession.query(ComputedFeature)
            .join(FeatureDefinition)
            .filter(
                FeatureDefinition.name == feature_name,
                ComputedFeature.entity_type == entity_type,
                ComputedFeature.season == current_season,
                ComputedFeature.value.isnot(None),
            )
        )
        current_values = [cf.value for cf in current_query.all()]

        # Get reference season values
        reference_query = (
            self.dbsession.query(ComputedFeature)
            .join(FeatureDefinition)
            .filter(
                FeatureDefinition.name == feature_name,
                ComputedFeature.entity_type == entity_type,
                ComputedFeature.season == reference_season,
                ComputedFeature.value.isnot(None),
            )
        )
        reference_values = [cf.value for cf in reference_query.all()]

        if not current_values or not reference_values:
            return {"drift_detected": False, "reason": "insufficient_data"}

        # Statistical tests for drift
        current_mean = np.mean(current_values)
        reference_mean = np.mean(reference_values)
        current_std = np.std(current_values)
        reference_std = np.std(reference_values)

        # Check for significant changes in mean and standard deviation
        mean_change = (
            abs(current_mean - reference_mean) / abs(reference_mean)
            if reference_mean != 0
            else 0
        )
        std_change = (
            abs(current_std - reference_std) / abs(reference_std)
            if reference_std != 0
            else 0
        )

        drift_detected = mean_change > drift_threshold or std_change > drift_threshold

        return {
            "drift_detected": drift_detected,
            "mean_change": mean_change,
            "std_change": std_change,
            "current_stats": {
                "mean": current_mean,
                "std": current_std,
                "count": len(current_values),
            },
            "reference_stats": {
                "mean": reference_mean,
                "std": reference_std,
                "count": len(reference_values),
            },
        }


class FeatureStore:
    """Main interface for the AIrsenal Feature Store."""

    def __init__(self, dbsession: Session = session):
        self.dbsession = dbsession
        self.registry = FeatureRegistry(dbsession)
        self.cache = FeatureCacheManager(dbsession)
        self.computer = FeatureComputer(dbsession)
        self.validator = FeatureValidator(dbsession)

        # Initialize with common FPL features
        self._register_default_features()

    def _register_default_features(self):
        """Register commonly used FPL features."""
        default_features = [
            {
                "name": "rolling_goals_5",
                "feature_type": "player",
                "description": "5-game rolling average of goals scored",
                "computation_logic": {
                    "type": "rolling",
                    "metric": "goals",
                    "window": 5,
                    "agg": "mean",
                },
            },
            {
                "name": "rolling_assists_5",
                "feature_type": "player",
                "description": "5-game rolling average of assists",
                "computation_logic": {
                    "type": "rolling",
                    "metric": "assists",
                    "window": 5,
                    "agg": "mean",
                },
            },
            {
                "name": "rolling_minutes_5",
                "feature_type": "player",
                "description": "5-game rolling average of minutes played",
                "computation_logic": {
                    "type": "rolling",
                    "metric": "minutes",
                    "window": 5,
                    "agg": "mean",
                },
            },
            {
                "name": "rolling_points_5",
                "feature_type": "player",
                "description": "5-game rolling average of FPL points",
                "computation_logic": {
                    "type": "rolling",
                    "metric": "points",
                    "window": 5,
                    "agg": "mean",
                },
            },
            {
                "name": "goals_form",
                "feature_type": "player",
                "description": "Exponentially weighted goals form",
                "computation_logic": {
                    "type": "form",
                    "metric": "goals",
                    "decay_factor": 0.9,
                },
            },
            {
                "name": "assists_form",
                "feature_type": "player",
                "description": "Exponentially weighted assists form",
                "computation_logic": {
                    "type": "form",
                    "metric": "assists",
                    "decay_factor": 0.9,
                },
            },
            {
                "name": "player_position",
                "feature_type": "player",
                "data_type": "string",
                "description": "Player position (GK, DEF, MID, FWD)",
                "computation_logic": {"type": "static", "metric": "position"},
            },
            {
                "name": "player_team",
                "feature_type": "player",
                "data_type": "string",
                "description": "Player's current team",
                "computation_logic": {"type": "static", "metric": "team"},
            },
        ]

        for feature_config in default_features:
            try:
                existing = (
                    self.dbsession.query(FeatureDefinition)
                    .filter_by(name=feature_config["name"])
                    .first()
                )

                if not existing:
                    self.registry.register_feature(**feature_config)
            except Exception as e:
                logger.warning(
                    "Failed to register default feature %s: %s",
                    feature_config["name"],
                    e,
                )

    def register_feature(self, **kwargs) -> FeatureDefinition:
        """Register a new feature. Proxy to FeatureRegistry.register_feature."""
        return self.registry.register_feature(**kwargs)

    def get_features(
        self,
        entity_type: str,
        entity_ids: list[int],
        feature_names: list[str],
        season: str = CURRENT_SEASON,
        gameweek: int = NEXT_GAMEWEEK,
        use_cache: bool = True,
    ) -> dict[int, dict[str, float | str | None]]:
        """
        Get feature values for multiple entities and features.

        Returns:
            Dict mapping entity_id -> {feature_name: value}
        """
        start_time = time.time()
        results: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        cache_hits = 0
        computations = 0

        for entity_id in entity_ids:
            for feature_name in feature_names:
                # Try cache first
                if use_cache:
                    cached_value = self.cache.get(
                        feature_name=feature_name,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        context=f"{season}:{gameweek}",
                    )

                    if cached_value is not None:
                        results[entity_id][feature_name] = cached_value
                        cache_hits += 1
                        continue

                # Compute feature
                try:
                    feature_def = self.registry.get_feature_definition(feature_name)
                    value = self.computer.compute_feature_by_logic(
                        feature_def=feature_def,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        season=season,
                        gameweek=gameweek,
                    )

                    # Validate value
                    if not self.validator.validate_feature_value(feature_def, value):
                        logger.warning(
                            "Validation failed for %s on %s:%s",
                            feature_name,
                            entity_type,
                            entity_id,
                        )
                        value = None

                    results[entity_id][feature_name] = value
                    computations += 1

                    # Cache the computed value
                    if use_cache and value is not None:
                        self.cache.set(
                            feature_name=feature_name,
                            entity_type=entity_type,
                            entity_id=entity_id,
                            value=value,
                            context=f"{season}:{gameweek}",
                        )

                except Exception as e:
                    logger.error(
                        "Failed to compute %s for %s:%s: %s",
                        feature_name,
                        entity_type,
                        entity_id,
                        e,
                    )
                    results[entity_id][feature_name] = None

        elapsed_time = time.time() - start_time
        logger.info(
            "Retrieved %s x %s features in %.3fs (%s cache hits, %s computations)",
            len(entity_ids),
            len(feature_names),
            elapsed_time,
            cache_hits,
            computations,
        )

        return dict(results)

    def update_features(
        self,
        entity_type: str,
        entity_id: int,
        features: dict[str, float | str],
        season: str = CURRENT_SEASON,
        gameweek: int | None = None,
        invalidate_cache: bool = True,
    ) -> None:
        """Update or create computed feature values."""

        for feature_name, value in features.items():
            try:
                feature_def = self.registry.get_feature_definition(feature_name)

                # Validate value
                if not self.validator.validate_feature_value(feature_def, value):
                    logger.warning(
                        "Skipping invalid value for %s: %s", feature_name, value
                    )
                    continue

                # Find existing computed feature or create new
                computed_feature = (
                    self.dbsession.query(ComputedFeature)
                    .filter_by(
                        feature_definition_id=feature_def.id,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        season=season,
                        gameweek=gameweek,
                    )
                    .first()
                )

                if computed_feature:
                    # Update existing
                    if isinstance(value, int | float):
                        computed_feature.value = float(value)
                        computed_feature.string_value = None
                    else:
                        computed_feature.value = None
                        computed_feature.string_value = str(value)
                    computed_feature.computed_at = datetime.now().isoformat()
                else:
                    # Create new
                    computed_feature = ComputedFeature(
                        feature_definition_id=feature_def.id,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        gameweek=gameweek,
                        season=season,
                        value=float(value) if isinstance(value, int | float) else None,
                        string_value=str(value)
                        if not isinstance(value, int | float)
                        else None,
                        computed_at=datetime.now().isoformat(),
                    )
                    self.dbsession.add(computed_feature)

                # Invalidate cache
                if invalidate_cache:
                    self.cache.invalidate(
                        feature_name=feature_name,
                        entity_type=entity_type,
                        entity_id=entity_id,
                    )

            except Exception as e:
                logger.error("Failed to update feature %s: %s", feature_name, e)

        self.dbsession.commit()

    def compute_features_batch(
        self,
        feature_names: list[str],
        entity_type: str = "player",
        entity_ids: list[int] | None = None,
        season: str = CURRENT_SEASON,
        gameweek_range: tuple[int, int] | None = None,
        chunk_size: int = 100,
    ) -> dict[str, int]:
        """Batch compute features for multiple entities and gameweeks."""

        if entity_ids is None:
            # Get all players if none specified
            if entity_type == "player":
                entity_ids = [p.player_id for p in self.dbsession.query(Player).all()]
            else:
                msg = "entity_ids must be provided for non-player entity types"
                raise ValueError(msg)

        if gameweek_range is None:
            gameweek_range = (NEXT_GAMEWEEK, NEXT_GAMEWEEK)

        start_gw, end_gw = gameweek_range
        total_computations = 0
        successful_computations = 0

        logger.info(
            "Starting batch computation for %s features, %s entities, gameweeks %s-%s",
            len(feature_names),
            len(entity_ids),
            start_gw,
            end_gw,
        )

        # Process in chunks to avoid memory issues
        for i in range(0, len(entity_ids), chunk_size):
            chunk_entity_ids = entity_ids[i : i + chunk_size]

            for gameweek in range(start_gw, end_gw + 1):
                features_dict = self.get_features(
                    entity_type=entity_type,
                    entity_ids=chunk_entity_ids,
                    feature_names=feature_names,
                    season=season,
                    gameweek=gameweek,
                    use_cache=False,  # Force computation
                )

                # Store computed features
                for entity_id, features in features_dict.items():
                    computed_features = {
                        k: v for k, v in features.items() if v is not None
                    }
                    if computed_features:
                        self.update_features(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            features=computed_features,
                            season=season,
                            gameweek=gameweek,
                            invalidate_cache=False,  # Don't invalidate during batch
                        )
                        successful_computations += len(computed_features)

                    total_computations += len(features)

            logger.info(
                "Processed chunk %s/%s",
                i // chunk_size + 1,
                (len(entity_ids) + chunk_size - 1) // chunk_size,
            )

        stats = {
            "total_computations": total_computations,
            "successful_computations": successful_computations,
            "failure_rate": (total_computations - successful_computations)
            / total_computations
            if total_computations > 0
            else 0,
        }

        logger.info("Batch computation completed: %s", stats)
        return stats

    def get_feature_stats(self, feature_name: str) -> dict[str, Any]:
        """Get statistics and metadata for a feature."""
        feature_def = self.registry.get_feature_definition(feature_name)

        # Count computed values
        value_count = (
            self.dbsession.query(ComputedFeature)
            .filter_by(feature_definition_id=feature_def.id)
            .count()
        )

        # Get value distribution for numeric features
        if feature_def.data_type in ["float", "int"]:
            values_query = self.dbsession.query(ComputedFeature.value).filter(
                ComputedFeature.feature_definition_id == feature_def.id,
                ComputedFeature.value.isnot(None),
            )
            values = [v[0] for v in values_query.all()]

            if values:
                stats = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                }
            else:
                stats = {"count": 0}
        else:
            stats = {"count": value_count}

        # Get cache stats
        cache_stats = self.cache.get_stats()

        return {
            "feature_definition": {
                "name": feature_def.name,
                "version": feature_def.version,
                "type": feature_def.feature_type,
                "data_type": feature_def.data_type,
                "description": feature_def.description,
            },
            "value_stats": stats,
            "cache_stats": cache_stats,
        }
