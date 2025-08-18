"""
Historical Data Backfilling for AIrsenal

This module provides comprehensive historical data backfilling capabilities for xG/xA data
and other metrics, with robust error handling, progress tracking, and data validation.

Key features:
- Batch processing with configurable chunk sizes
- Parallel processing for speed optimization
- Circuit breaker pattern for API reliability
- Comprehensive data validation and reconciliation
- Resume capability for interrupted operations
- Progress tracking and reporting
- Data quality assessment and gap identification
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import and_, func

from airsenal.framework.env import AIRSENAL_HOME
from airsenal.framework.logging_config import get_logger
from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerScore,
    session_scope,
)
from airsenal.framework.season import season_str_to_year
from airsenal.framework.xg_data_provider import XGDataManager, create_xg_manager

logger = get_logger(__name__)


class BackfillStatus(Enum):
    """Status of backfill operations"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ValidationSeverity(Enum):
    """Severity levels for data validation issues"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BackfillProgress:
    """Track progress of backfill operations"""

    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    start_time: datetime | None = None
    last_update_time: datetime | None = None
    estimated_completion: datetime | None = None
    current_operation: str = ""

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        processed = self.completed_items + self.failed_items
        if processed == 0:
            return 1.0
        return self.completed_items / processed

    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time"""
        if self.start_time is None:
            return timedelta(0)
        return datetime.now() - self.start_time

    def estimate_completion_time(self) -> datetime | None:
        """Estimate completion time based on current progress"""
        if (
            self.progress_percentage <= 0
            or self.start_time is None
            or self.completed_items == 0
        ):
            return None

        elapsed = self.elapsed_time.total_seconds()
        rate = self.completed_items / elapsed  # items per second
        remaining_items = self.total_items - self.completed_items

        if rate > 0:
            remaining_seconds = remaining_items / rate
            return datetime.now() + timedelta(seconds=remaining_seconds)

        return None


@dataclass
class ValidationIssue:
    """Represent a data validation issue"""

    severity: ValidationSeverity
    category: str
    description: str
    affected_entity: str
    entity_id: int | None = None
    suggested_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackfillResult:
    """Result of a backfill operation"""

    status: BackfillStatus
    progress: BackfillProgress
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    data_gaps: list[dict[str, Any]] = field(default_factory=list)
    reconciliation_report: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)

    def add_validation_issue(
        self,
        severity: ValidationSeverity,
        category: str,
        description: str,
        affected_entity: str,
        entity_id: int | None = None,
        suggested_action: str | None = None,
        **metadata,
    ):
        """Add a validation issue to the result"""
        issue = ValidationIssue(
            severity=severity,
            category=category,
            description=description,
            affected_entity=affected_entity,
            entity_id=entity_id,
            suggested_action=suggested_action,
            metadata=metadata,
        )
        self.validation_issues.append(issue)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical validation issues"""
        return any(
            issue.severity == ValidationSeverity.CRITICAL
            for issue in self.validation_issues
        )

    @property
    def issues_by_severity(self) -> dict[ValidationSeverity, list[ValidationIssue]]:
        """Group validation issues by severity"""
        grouped = {}
        for issue in self.validation_issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)
        return grouped


class BackfillProgressTracker:
    """Monitor and track progress of long-running backfill operations"""

    def __init__(self, session_id: str, total_items: int):
        self.session_id = session_id
        self.progress = BackfillProgress(total_items=total_items)
        self.lock = threading.Lock()
        self._callbacks = []
        self._persistence_path = (
            Path(AIRSENAL_HOME) / "backfill_progress" / f"{session_id}.json"
        )
        self._persistence_path.parent.mkdir(exist_ok=True)

        # Load existing progress if available
        self._load_progress()

    def start(self):
        """Start progress tracking"""
        with self.lock:
            self.progress.start_time = datetime.now()
            self.progress.last_update_time = datetime.now()
            self._save_progress()

    def update(
        self,
        completed_delta: int = 1,
        failed_delta: int = 0,
        skipped_delta: int = 0,
        current_operation: str = "",
    ):
        """Update progress counters"""
        with self.lock:
            self.progress.completed_items += completed_delta
            self.progress.failed_items += failed_delta
            self.progress.skipped_items += skipped_delta
            self.progress.current_operation = current_operation
            self.progress.last_update_time = datetime.now()
            self.progress.estimated_completion = (
                self.progress.estimate_completion_time()
            )

            self._save_progress()
            self._notify_callbacks()

    def add_callback(self, callback):
        """Add a progress callback function"""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks of progress update"""
        for callback in self._callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    def _save_progress(self):
        """Persist progress to disk for resume capability"""
        try:
            progress_data = {
                "session_id": self.session_id,
                "total_items": self.progress.total_items,
                "completed_items": self.progress.completed_items,
                "failed_items": self.progress.failed_items,
                "skipped_items": self.progress.skipped_items,
                "start_time": self.progress.start_time.isoformat()
                if self.progress.start_time
                else None,
                "last_update_time": self.progress.last_update_time.isoformat()
                if self.progress.last_update_time
                else None,
                "current_operation": self.progress.current_operation,
            }

            with open(self._persistence_path, "w") as f:
                json.dump(progress_data, f, indent=2)

        except Exception as e:
            logger.warning("Failed to save progress: %s", e)

    def _load_progress(self):
        """Load progress from disk if available"""
        try:
            if self._persistence_path.exists():
                with open(self._persistence_path) as f:
                    progress_data = json.load(f)

                self.progress.completed_items = progress_data.get("completed_items", 0)
                self.progress.failed_items = progress_data.get("failed_items", 0)
                self.progress.skipped_items = progress_data.get("skipped_items", 0)
                self.progress.current_operation = progress_data.get(
                    "current_operation", ""
                )

                if progress_data.get("start_time"):
                    self.progress.start_time = datetime.fromisoformat(
                        progress_data["start_time"]
                    )
                if progress_data.get("last_update_time"):
                    self.progress.last_update_time = datetime.fromisoformat(
                        progress_data["last_update_time"]
                    )

        except Exception as e:
            logger.warning("Failed to load progress: %s", e)

    def cleanup(self):
        """Clean up progress tracking files"""
        try:
            if self._persistence_path.exists():
                self._persistence_path.unlink()
        except Exception as e:
            logger.warning("Failed to cleanup progress file: %s", e)


class BackfillValidator:
    """Comprehensive data quality validation for backfilled data"""

    def __init__(self):
        self.validation_rules = {
            "xg_data_consistency": self._validate_xg_data_consistency,
            "missing_data_detection": self._validate_missing_data,
            "statistical_outliers": self._validate_statistical_outliers,
            "cross_reference_validation": self._validate_cross_references,
            "temporal_consistency": self._validate_temporal_consistency,
        }

    def validate_backfilled_data(
        self, season: str, gameweeks: list[int] | None = None
    ) -> list[ValidationIssue]:
        """Run comprehensive validation on backfilled data"""
        issues = []

        logger.info(
            f"Starting validation for {season}"
            + (f" gameweeks {gameweeks}" if gameweeks else "")
        )

        for rule_name, rule_func in self.validation_rules.items():
            try:
                logger.debug("Running validation rule: %s", rule_name)
                rule_issues = rule_func(season, gameweeks)
                issues.extend(rule_issues)
                logger.debug("Rule %s found %s issues", rule_name, len(rule_issues))
            except Exception as e:
                logger.error("Validation rule %s failed: %s", rule_name, e)
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="validation_system",
                        description=f"Validation rule {rule_name} failed: {e}",
                        affected_entity="validation_system",
                    )
                )

        logger.info("Validation completed with %s total issues", len(issues))
        return issues

    def _validate_xg_data_consistency(
        self, season: str, gameweeks: list[int] | None
    ) -> list[ValidationIssue]:
        """Validate xG data consistency and calculations"""
        issues = []

        with session_scope() as db_session:
            query = db_session.query(PlayerScore).filter(
                PlayerScore.fixture.has(Fixture.season == season)
            )

            if gameweeks:
                query = query.filter(
                    PlayerScore.fixture.has(Fixture.gameweek.in_(gameweeks))
                )

            # Check for xG data without corresponding match data
            scores_with_xg = query.filter(PlayerScore.expected_goals.isnot(None)).all()

            for score in scores_with_xg:
                # Validate xG + xA = xGI consistency
                if (
                    score.expected_goals is not None
                    and score.expected_assists is not None
                    and score.expected_goal_involvements is not None
                ):
                    calculated_xgi = score.expected_goals + score.expected_assists
                    if abs(calculated_xgi - score.expected_goal_involvements) > 0.01:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="xg_consistency",
                                description=f"xGI mismatch: calculated {calculated_xgi:.3f}, stored {score.expected_goal_involvements:.3f}",
                                affected_entity="player_score",
                                entity_id=score.id,
                                suggested_action="Recalculate xGI from xG + xA",
                            )
                        )

                # Validate xG values are reasonable (0 <= xG <= 5 per match)
                if score.expected_goals is not None:
                    if score.expected_goals < 0 or score.expected_goals > 5:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="xg_outlier",
                                description=f"Unrealistic xG value: {score.expected_goals}",
                                affected_entity="player_score",
                                entity_id=score.id,
                                suggested_action="Review and correct xG value",
                            )
                        )

        return issues

    def _validate_missing_data(
        self, season: str, gameweeks: list[int] | None
    ) -> list[ValidationIssue]:
        """Detect missing xG data gaps"""
        issues = []

        with session_scope() as db_session:
            # Count total player scores vs scores with xG data
            base_query = db_session.query(PlayerScore).filter(
                PlayerScore.fixture.has(Fixture.season == season)
            )

            if gameweeks:
                base_query = base_query.filter(
                    PlayerScore.fixture.has(Fixture.gameweek.in_(gameweeks))
                )

            total_scores = base_query.count()
            scores_with_xg = base_query.filter(
                PlayerScore.expected_goals.isnot(None)
            ).count()

            coverage_percentage = (
                (scores_with_xg / total_scores * 100) if total_scores > 0 else 0
            )

            if coverage_percentage < 80:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="data_coverage",
                        description=f"Low xG data coverage: {coverage_percentage:.1f}% ({scores_with_xg}/{total_scores})",
                        affected_entity="season_data",
                        suggested_action="Investigate missing xG data sources",
                        metadata={"coverage_percentage": coverage_percentage},
                    )
                )

            # Check for missing data by gameweek
            if gameweeks is None:
                gameweeks = [
                    gw
                    for (gw,) in db_session.query(Fixture.gameweek)
                    .filter(Fixture.season == season)
                    .distinct()
                    .all()
                ]

            for gw in gameweeks:
                gw_total = base_query.filter(
                    PlayerScore.fixture.has(Fixture.gameweek == gw)
                ).count()

                gw_with_xg = base_query.filter(
                    and_(
                        PlayerScore.fixture.has(Fixture.gameweek == gw),
                        PlayerScore.expected_goals.isnot(None),
                    )
                ).count()

                gw_coverage = (gw_with_xg / gw_total * 100) if gw_total > 0 else 0

                if gw_coverage < 50:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="gameweek_gap",
                            description=f"Very low xG coverage for GW{gw}: {gw_coverage:.1f}%",
                            affected_entity="gameweek",
                            entity_id=gw,
                            suggested_action="Priority backfill for this gameweek",
                            metadata={"gameweek": gw, "coverage": gw_coverage},
                        )
                    )

        return issues

    def _validate_statistical_outliers(
        self, season: str, gameweeks: list[int] | None
    ) -> list[ValidationIssue]:
        """Identify statistical outliers in xG data"""
        issues = []

        with session_scope() as db_session:
            query = db_session.query(PlayerScore).filter(
                and_(
                    PlayerScore.fixture.has(Fixture.season == season),
                    PlayerScore.expected_goals.isnot(None),
                )
            )

            if gameweeks:
                query = query.filter(
                    PlayerScore.fixture.has(Fixture.gameweek.in_(gameweeks))
                )

            # Statistical analysis of xG values
            xg_stats = (
                db_session.query(
                    func.avg(PlayerScore.expected_goals).label("mean_xg"),
                    func.max(PlayerScore.expected_goals).label("max_xg"),
                    func.min(PlayerScore.expected_goals).label("min_xg"),
                    func.count(PlayerScore.expected_goals).label("count_xg"),
                )
                .filter(
                    and_(
                        PlayerScore.fixture.has(Fixture.season == season),
                        PlayerScore.expected_goals.isnot(None),
                    )
                )
                .first()
            )

            if xg_stats and xg_stats.count_xg > 10:
                # Flag extreme outliers (> 3.0 xG per match is very rare)
                extreme_xg_scores = query.filter(PlayerScore.expected_goals > 3.0).all()

                for score in extreme_xg_scores:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            category="statistical_outlier",
                            description=f"Unusually high xG: {score.expected_goals:.3f} for player {score.player_id}",
                            affected_entity="player_score",
                            entity_id=score.id,
                            suggested_action="Verify against match reports",
                            metadata={
                                "xg_value": score.expected_goals,
                                "player_id": score.player_id,
                                "fixture_id": score.fixture_id,
                            },
                        )
                    )

        return issues

    def _validate_cross_references(
        self, season: str, gameweeks: list[int] | None
    ) -> list[ValidationIssue]:
        """Validate cross-references between tables"""
        issues = []

        with session_scope() as db_session:
            # Check for PlayerScore records without corresponding Player records
            orphaned_scores = db_session.query(PlayerScore).filter(
                and_(
                    PlayerScore.fixture.has(Fixture.season == season),
                    ~PlayerScore.player_id.in_(db_session.query(Player.player_id)),
                )
            )

            if gameweeks:
                orphaned_scores = orphaned_scores.filter(
                    PlayerScore.fixture.has(Fixture.gameweek.in_(gameweeks))
                )

            for score in orphaned_scores.all():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="data_integrity",
                        description=f"PlayerScore {score.id} references non-existent player {score.player_id}",
                        affected_entity="player_score",
                        entity_id=score.id,
                        suggested_action="Remove orphaned record or create missing player",
                    )
                )

        return issues

    def _validate_temporal_consistency(
        self, season: str, gameweeks: list[int] | None
    ) -> list[ValidationIssue]:
        """Validate temporal consistency of data"""
        issues = []

        with session_scope() as db_session:
            # Check for fixture dates that don't align with season
            season_year = season_str_to_year(season)

            fixtures_query = db_session.query(Fixture).filter(Fixture.season == season)
            if gameweeks:
                fixtures_query = fixtures_query.filter(Fixture.gameweek.in_(gameweeks))

            for fixture in fixtures_query.all():
                if fixture.date:
                    try:
                        fixture_date = datetime.fromisoformat(
                            fixture.date.replace("Z", "+00:00")
                        )
                        fixture_year = fixture_date.year

                        # Fixture should be in season year or season year + 1
                        if not (season_year <= fixture_year <= season_year + 1):
                            issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    category="temporal_consistency",
                                    description=f"Fixture date {fixture.date} doesn't align with season {season}",
                                    affected_entity="fixture",
                                    entity_id=fixture.fixture_id,
                                    suggested_action="Verify fixture date accuracy",
                                )
                            )
                    except ValueError:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="data_format",
                                description=f"Invalid date format for fixture {fixture.fixture_id}: {fixture.date}",
                                affected_entity="fixture",
                                entity_id=fixture.fixture_id,
                                suggested_action="Correct date format",
                            )
                        )

        return issues


class HistoricalDataBackfiller:
    """Main class for historical data backfilling operations"""

    def __init__(self, xg_manager: XGDataManager | None = None):
        self.xg_manager = xg_manager or create_xg_manager()
        self.validator = BackfillValidator()
        self.logger = get_logger(__name__)
        self._batch_size = 50  # Default batch size for processing
        self._max_workers = 4  # Default thread pool size
        self._retry_attempts = 3
        self._retry_delay = 1.0  # seconds

    def configure(
        self,
        batch_size: int = 50,
        max_workers: int = 4,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """Configure backfill operation parameters"""
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay

        self.logger.info(
            "Configured backfiller: batch_size=%s, max_workers=%s, retry_attempts=%s",
            batch_size,
            max_workers,
            retry_attempts,
        )

    def backfill_season(
        self, season: str, resume_session_id: str | None = None
    ) -> BackfillResult:
        """Backfill all data for a complete season"""
        session_id = resume_session_id or f"season_{season}_{int(time.time())}"

        self.logger.info(
            "Starting season backfill for %s (session: %s)", season, session_id
        )

        # Get all gameweeks for the season
        with session_scope() as db_session:
            gameweeks = [
                gw
                for (gw,) in db_session.query(Fixture.gameweek)
                .filter(Fixture.season == season)
                .distinct()
                .order_by(Fixture.gameweek)
                .all()
            ]

        if not gameweeks:
            result = BackfillResult(
                status=BackfillStatus.FAILED, progress=BackfillProgress()
            )
            result.add_validation_issue(
                ValidationSeverity.CRITICAL,
                "no_data",
                f"No fixtures found for season {season}",
                "season",
                suggested_action="Verify season exists in database",
            )
            return result

        self.logger.info(
            "Found %s gameweeks for %s: %s", len(gameweeks), season, gameweeks
        )

        # Initialize progress tracker
        tracker = BackfillProgressTracker(session_id, len(gameweeks))
        tracker.start()

        result = BackfillResult(
            status=BackfillStatus.RUNNING, progress=tracker.progress
        )

        # Add progress logging callback
        def log_progress(progress: BackfillProgress):
            if progress.completed_items % 5 == 0:  # Log every 5 gameweeks
                self.logger.info(
                    "Season %s backfill progress: %.1f%% (%s/%s)",
                    season,
                    progress.progress_percentage,
                    progress.completed_items,
                    progress.total_items,
                )

        tracker.add_callback(log_progress)

        try:
            # Process gameweeks in parallel batches
            failed_gameweeks = []

            for i in range(0, len(gameweeks), self._batch_size):
                batch = gameweeks[i : i + self._batch_size]
                tracker.update(
                    current_operation=f"Processing gameweeks {batch[0]}-{batch[-1]}"
                )

                batch_results = self._process_gameweek_batch(season, batch)

                for gw, gw_result in batch_results.items():
                    if gw_result.status == BackfillStatus.COMPLETED:
                        tracker.update(completed_delta=1)
                    else:
                        tracker.update(failed_delta=1)
                        failed_gameweeks.append(gw)
                        result.validation_issues.extend(gw_result.validation_issues)

            # Retry failed gameweeks
            if failed_gameweeks:
                self.logger.warning(
                    "Retrying %s failed gameweeks", len(failed_gameweeks)
                )
                tracker.update(current_operation="Retrying failed gameweeks")

                retry_results = self._retry_failed_gameweeks(season, failed_gameweeks)
                for gw, gw_result in retry_results.items():
                    if gw_result.status == BackfillStatus.COMPLETED:
                        tracker.update(completed_delta=1, failed_delta=-1)
                        failed_gameweeks.remove(gw)

            # Final validation
            tracker.update(current_operation="Running final validation")
            validation_issues = self.validator.validate_backfilled_data(season)
            result.validation_issues.extend(validation_issues)

            # Generate reconciliation report
            result.reconciliation_report = self._generate_reconciliation_report(season)

            # Determine final status
            if not failed_gameweeks and not result.has_critical_issues:
                result.status = BackfillStatus.COMPLETED
                self.logger.info("Season %s backfill completed successfully", season)
            else:
                result.status = BackfillStatus.FAILED
                self.logger.error(
                    "Season %s backfill failed. Failed gameweeks: %s",
                    season,
                    failed_gameweeks,
                )

            result.progress = tracker.progress

        except Exception as e:
            self.logger.error("Season %s backfill failed with exception: %s", season, e)
            result.status = BackfillStatus.FAILED
            result.add_validation_issue(
                ValidationSeverity.CRITICAL,
                "system_error",
                f"Backfill failed with exception: {e}",
                "system",
                suggested_action="Check logs and system resources",
            )

        finally:
            tracker.cleanup()

        return result

    def backfill_gameweeks(
        self, season: str, gameweeks: list[int], session_id: str | None = None
    ) -> BackfillResult:
        """Backfill specific gameweeks"""
        session_id = (
            session_id
            or f"gameweeks_{season}_{'_'.join(map(str, gameweeks))}_{int(time.time())}"
        )

        self.logger.info(
            "Starting gameweeks backfill for %s GW%s (session: %s)",
            season,
            gameweeks,
            session_id,
        )

        tracker = BackfillProgressTracker(session_id, len(gameweeks))
        tracker.start()

        result = BackfillResult(
            status=BackfillStatus.RUNNING, progress=tracker.progress
        )

        try:
            batch_results = self._process_gameweek_batch(season, gameweeks)

            failed_gameweeks = []
            for gw, gw_result in batch_results.items():
                if gw_result.status == BackfillStatus.COMPLETED:
                    tracker.update(completed_delta=1)
                else:
                    tracker.update(failed_delta=1)
                    failed_gameweeks.append(gw)
                    result.validation_issues.extend(gw_result.validation_issues)

            # Validation
            validation_issues = self.validator.validate_backfilled_data(
                season, gameweeks
            )
            result.validation_issues.extend(validation_issues)

            # Reconciliation report
            result.reconciliation_report = self._generate_reconciliation_report(
                season, gameweeks
            )

            # Final status
            if not failed_gameweeks and not result.has_critical_issues:
                result.status = BackfillStatus.COMPLETED
            else:
                result.status = BackfillStatus.FAILED

            result.progress = tracker.progress

        except Exception as e:
            self.logger.error("Gameweeks backfill failed: %s", e)
            result.status = BackfillStatus.FAILED
            result.add_validation_issue(
                ValidationSeverity.CRITICAL,
                "system_error",
                f"Backfill failed: {e}",
                "system",
            )
        finally:
            tracker.cleanup()

        return result

    def _process_gameweek_batch(
        self, season: str, gameweeks: list[int]
    ) -> dict[int, BackfillResult]:
        """Process a batch of gameweeks in parallel"""
        results = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all gameweek tasks
            future_to_gw = {
                executor.submit(self._backfill_single_gameweek, season, gw): gw
                for gw in gameweeks
            }

            # Collect results as they complete
            for future in as_completed(future_to_gw):
                gw = future_to_gw[future]
                try:
                    results[gw] = future.result()
                except Exception as e:
                    self.logger.error("Failed to process gameweek %s: %s", gw, e)
                    result = BackfillResult(
                        status=BackfillStatus.FAILED, progress=BackfillProgress()
                    )
                    result.add_validation_issue(
                        ValidationSeverity.ERROR,
                        "processing_error",
                        f"Failed to process gameweek {gw}: {e}",
                        "gameweek",
                        entity_id=gw,
                    )
                    results[gw] = result

        return results

    def _backfill_single_gameweek(self, season: str, gameweek: int) -> BackfillResult:
        """Backfill data for a single gameweek"""
        result = BackfillResult(
            status=BackfillStatus.RUNNING, progress=BackfillProgress(total_items=1)
        )

        try:
            self.logger.debug("Processing %s GW%s", season, gameweek)

            # Use existing xG manager to sync data
            updated_count = self.xg_manager.sync_xg_data_for_gameweek(season, gameweek)

            if updated_count > 0:
                self.logger.debug(
                    "Updated %s records for %s GW%s", updated_count, season, gameweek
                )
                result.status = BackfillStatus.COMPLETED
                result.progress.completed_items = 1
            else:
                self.logger.warning("No data updated for %s GW%s", season, gameweek)
                result.status = BackfillStatus.COMPLETED  # Not necessarily a failure
                result.progress.skipped_items = 1
                result.add_validation_issue(
                    ValidationSeverity.WARNING,
                    "no_updates",
                    f"No data updated for gameweek {gameweek}",
                    "gameweek",
                    entity_id=gameweek,
                    suggested_action="Check if data already exists or provider has data",
                )

        except Exception as e:
            self.logger.error("Failed to backfill %s GW%s: %s", season, gameweek, e)
            result.status = BackfillStatus.FAILED
            result.progress.failed_items = 1
            result.add_validation_issue(
                ValidationSeverity.ERROR,
                "backfill_error",
                f"Failed to backfill gameweek {gameweek}: {e}",
                "gameweek",
                entity_id=gameweek,
            )

        return result

    def _retry_failed_gameweeks(
        self, season: str, gameweeks: list[int]
    ) -> dict[int, BackfillResult]:
        """Retry failed gameweeks with exponential backoff"""
        results = {}

        for gw in gameweeks:
            for attempt in range(self._retry_attempts):
                try:
                    self.logger.info(
                        "Retry attempt %s/%s for %s GW%s",
                        attempt + 1,
                        self._retry_attempts,
                        season,
                        gw,
                    )

                    # Exponential backoff
                    if attempt > 0:
                        delay = self._retry_delay * (2 ** (attempt - 1))
                        time.sleep(delay)

                    result = self._backfill_single_gameweek(season, gw)

                    if result.status == BackfillStatus.COMPLETED:
                        results[gw] = result
                        break

                except Exception as e:
                    self.logger.warning(
                        "Retry %s failed for %s GW%s: %s", attempt + 1, season, gw, e
                    )
                    if attempt == self._retry_attempts - 1:
                        # Final attempt failed
                        failed_result = BackfillResult(
                            status=BackfillStatus.FAILED, progress=BackfillProgress()
                        )
                        failed_result.add_validation_issue(
                            ValidationSeverity.ERROR,
                            "retry_exhausted",
                            f"All retry attempts exhausted for gameweek {gw}",
                            "gameweek",
                            entity_id=gw,
                        )
                        results[gw] = failed_result

        return results

    def _generate_reconciliation_report(
        self, season: str, gameweeks: list[int] | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive reconciliation report"""
        report = {
            "season": season,
            "gameweeks": gameweeks,
            "timestamp": datetime.now().isoformat(),
            "data_coverage": {},
            "quality_metrics": {},
            "gaps_identified": [],
            "recommendations": [],
        }

        with session_scope() as db_session:
            # Base query
            base_query = db_session.query(PlayerScore).filter(
                PlayerScore.fixture.has(Fixture.season == season)
            )

            if gameweeks:
                base_query = base_query.filter(
                    PlayerScore.fixture.has(Fixture.gameweek.in_(gameweeks))
                )

            # Calculate coverage metrics
            total_scores = base_query.count()
            scores_with_xg = base_query.filter(
                PlayerScore.expected_goals.isnot(None)
            ).count()
            scores_with_xa = base_query.filter(
                PlayerScore.expected_assists.isnot(None)
            ).count()
            scores_with_xgi = base_query.filter(
                PlayerScore.expected_goal_involvements.isnot(None)
            ).count()

            report["data_coverage"] = {
                "total_player_scores": total_scores,
                "scores_with_xg": scores_with_xg,
                "scores_with_xa": scores_with_xa,
                "scores_with_xgi": scores_with_xgi,
                "xg_coverage_percentage": (scores_with_xg / total_scores * 100)
                if total_scores > 0
                else 0,
                "xa_coverage_percentage": (scores_with_xa / total_scores * 100)
                if total_scores > 0
                else 0,
                "xgi_coverage_percentage": (scores_with_xgi / total_scores * 100)
                if total_scores > 0
                else 0,
            }

            # Quality metrics
            if scores_with_xg > 0:
                xg_stats = (
                    db_session.query(
                        func.avg(PlayerScore.expected_goals).label("avg_xg"),
                        func.max(PlayerScore.expected_goals).label("max_xg"),
                        func.min(PlayerScore.expected_goals).label("min_xg"),
                        func.count(PlayerScore.expected_goals).label("count_xg"),
                    )
                    .filter(
                        and_(
                            PlayerScore.fixture.has(Fixture.season == season),
                            PlayerScore.expected_goals.isnot(None),
                        )
                    )
                    .first()
                )

                report["quality_metrics"] = {
                    "average_xg_per_player_per_match": float(xg_stats.avg_xg)
                    if xg_stats.avg_xg
                    else 0,
                    "maximum_xg_in_match": float(xg_stats.max_xg)
                    if xg_stats.max_xg
                    else 0,
                    "minimum_xg_in_match": float(xg_stats.min_xg)
                    if xg_stats.min_xg
                    else 0,
                }

            # Identify gaps by gameweek
            gw_query = gameweeks or [
                gw
                for (gw,) in db_session.query(Fixture.gameweek)
                .filter(Fixture.season == season)
                .distinct()
                .all()
            ]

            for gw in gw_query:
                gw_total = base_query.filter(
                    PlayerScore.fixture.has(Fixture.gameweek == gw)
                ).count()

                gw_with_xg = base_query.filter(
                    and_(
                        PlayerScore.fixture.has(Fixture.gameweek == gw),
                        PlayerScore.expected_goals.isnot(None),
                    )
                ).count()

                gw_coverage = (gw_with_xg / gw_total * 100) if gw_total > 0 else 0

                if gw_coverage < 90:  # Flag gameweeks with < 90% coverage
                    report["gaps_identified"].append(
                        {
                            "gameweek": gw,
                            "total_scores": gw_total,
                            "scores_with_xg": gw_with_xg,
                            "coverage_percentage": gw_coverage,
                            "severity": "high" if gw_coverage < 50 else "medium",
                        }
                    )

            # Generate recommendations
            overall_coverage = report["data_coverage"]["xg_coverage_percentage"]

            if overall_coverage < 80:
                report["recommendations"].append(
                    {
                        "priority": "high",
                        "action": "comprehensive_backfill",
                        "description": f"Overall xG coverage is only {overall_coverage:.1f}%. Consider full season re-backfill.",
                    }
                )

            if len(report["gaps_identified"]) > 0:
                high_priority_gaps = [
                    gap
                    for gap in report["gaps_identified"]
                    if gap["severity"] == "high"
                ]
                if high_priority_gaps:
                    report["recommendations"].append(
                        {
                            "priority": "high",
                            "action": "targeted_backfill",
                            "description": f"Prioritize backfill for {len(high_priority_gaps)} gameweeks with <50% coverage",
                            "gameweeks": [
                                gap["gameweek"] for gap in high_priority_gaps
                            ],
                        }
                    )

        return report

    def incremental_backfill(self, season: str, days_back: int = 7) -> BackfillResult:
        """Perform incremental backfill for recent data"""
        self.logger.info(
            "Starting incremental backfill for %s, last %s days", season, days_back
        )

        cutoff_date = datetime.now() - timedelta(days=days_back)

        with session_scope() as db_session:
            # Find recent fixtures
            recent_fixtures = (
                db_session.query(Fixture)
                .filter(and_(Fixture.season == season, Fixture.date.isnot(None)))
                .all()
            )

            # Filter fixtures by date
            target_gameweeks = set()
            for fixture in recent_fixtures:
                try:
                    fixture_date = datetime.fromisoformat(
                        fixture.date.replace("Z", "+00:00")
                    )
                    if fixture_date >= cutoff_date:
                        target_gameweeks.add(fixture.gameweek)
                except (ValueError, AttributeError):
                    continue

            if not target_gameweeks:
                result = BackfillResult(
                    status=BackfillStatus.COMPLETED, progress=BackfillProgress()
                )
                result.add_validation_issue(
                    ValidationSeverity.INFO,
                    "no_recent_data",
                    f"No fixtures found in last {days_back} days for {season}",
                    "season",
                )
                return result

            self.logger.info(
                "Found %s recent gameweeks: %s",
                len(target_gameweeks),
                sorted(target_gameweeks),
            )

            # Backfill the recent gameweeks
            return self.backfill_gameweeks(season, sorted(target_gameweeks))

    def get_missing_data_report(self, season: str) -> dict[str, Any]:
        """Generate comprehensive missing data report"""
        self.logger.info("Generating missing data report for %s", season)

        validation_issues = self.validator.validate_backfilled_data(season)
        reconciliation_report = self._generate_reconciliation_report(season)

        # Categorize issues
        missing_data_issues = [
            issue
            for issue in validation_issues
            if issue.category in ["data_coverage", "gameweek_gap", "missing_data"]
        ]

        return {
            "season": season,
            "timestamp": datetime.now().isoformat(),
            "missing_data_issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "description": issue.description,
                    "affected_entity": issue.affected_entity,
                    "entity_id": issue.entity_id,
                    "suggested_action": issue.suggested_action,
                    "metadata": issue.metadata,
                }
                for issue in missing_data_issues
            ],
            "reconciliation_report": reconciliation_report,
            "summary": {
                "total_issues": len(missing_data_issues),
                "critical_issues": len(
                    [
                        i
                        for i in missing_data_issues
                        if i.severity == ValidationSeverity.CRITICAL
                    ]
                ),
                "high_priority_gaps": len(
                    reconciliation_report.get("gaps_identified", [])
                ),
                "overall_coverage": reconciliation_report["data_coverage"][
                    "xg_coverage_percentage"
                ],
            },
        }
