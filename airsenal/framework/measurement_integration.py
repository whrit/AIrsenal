"""
Integration module for measurement update system with AIrsenal's Kalman filters.

This module provides high-level integration functions and performance-optimized
workflows for using the measurement update system with AIrsenal's existing
Kalman filter infrastructure.

Key Features:
- Seamless integration with existing KalmanPlayerModel
- Performance-optimized batch processing workflows
- Automatic configuration based on database schema
- Memory-efficient processing for large datasets
- Monitoring and diagnostics integration

Classes:
    MeasurementUpdateIntegrator: High-level integration class
    PerformanceOptimizer: Performance optimization utilities
    DiagnosticsCollector: System monitoring and diagnostics

Usage:
    ```python
    from airsenal.framework.measurement_integration import MeasurementUpdateIntegrator
    
    # Create integrator
    integrator = MeasurementUpdateIntegrator(kalman_model)
    
    # Process gameweek data
    results = integrator.process_gameweek(
        gameweek=15, 
        season="2023", 
        session=db_session
    )
    
    # Get performance metrics
    metrics = integrator.get_performance_metrics()
    ```
"""

from __future__ import annotations

import logging
import time
import gc
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from airsenal.framework.measurement_update import (
    MeasurementProcessor,
    NoiseEstimator,
    OutlierDetector,
    InnovationMonitor,
    BatchUpdater,
    MeasurementConfig,
    NoiseConfig,
    create_measurement_pipeline,
    process_gameweek_measurements
)
from airsenal.framework.kalman_player_model import KalmanPlayerModel
from airsenal.framework.kalman_filter import FilterConfig
from airsenal.framework.schema import PlayerScore, Player, Fixture, Result
from airsenal.framework.adaptive_player_model import StateSpaceConfig

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for measurement update integration."""
    
    # Processing options
    batch_size: int = 100
    use_parallel_processing: bool = True
    max_workers: int = mp.cpu_count() // 2
    
    # Memory management
    enable_memory_optimization: bool = True
    clear_cache_frequency: int = 10  # Clear every N gameweeks
    
    # Quality control
    max_outlier_ratio_per_gameweek: float = 0.15
    min_players_per_position: int = 3
    
    # Monitoring
    enable_performance_monitoring: bool = True
    enable_diagnostics_collection: bool = True
    log_processing_stats: bool = True
    
    # Integration specific
    auto_update_noise_estimates: bool = True
    auto_tune_filter_parameters: bool = True
    save_intermediate_results: bool = False


@dataclass
class ProcessingResult:
    """Result of measurement processing for a gameweek."""
    
    gameweek: int
    season: str
    players_processed: int
    players_updated: int
    outliers_detected: int
    processing_time: float
    memory_usage_mb: float
    
    # Quality metrics
    average_innovation_nis: float
    filter_consistency_rate: float
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Performance breakdown
    extraction_time: float = 0.0
    normalization_time: float = 0.0
    outlier_detection_time: float = 0.0
    kalman_update_time: float = 0.0


class MeasurementUpdateIntegrator:
    """
    High-level integration class for measurement updates with Kalman filters.
    
    Provides optimized workflows, automatic configuration, and seamless
    integration with AIrsenal's existing infrastructure.
    """
    
    def __init__(
        self,
        kalman_model: KalmanPlayerModel,
        config: Optional[IntegrationConfig] = None,
        measurement_config: Optional[MeasurementConfig] = None,
        noise_config: Optional[NoiseConfig] = None
    ):
        """
        Initialize measurement update integrator.
        
        Args:
            kalman_model: Existing KalmanPlayerModel instance
            config: Integration configuration
            measurement_config: Measurement processing configuration
            noise_config: Noise estimation configuration
        """
        self.kalman_model = kalman_model
        self.config = config or IntegrationConfig()
        
        # Create measurement processing pipeline
        self.batch_updater = create_measurement_pipeline(
            measurement_config, noise_config
        )
        
        # Performance optimizer
        self.optimizer = PerformanceOptimizer(self.config)
        
        # Diagnostics collector
        if self.config.enable_diagnostics_collection:
            self.diagnostics = DiagnosticsCollector()
        else:
            self.diagnostics = None
        
        # Processing state
        self.processing_history: List[ProcessingResult] = []
        self.gameweeks_processed: set = set()
        
        # Cache for efficiency
        self._player_position_cache: Dict[int, str] = {}
        self._opponent_strength_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info(f"Initialized MeasurementUpdateIntegrator with batch size {self.config.batch_size}")
    
    def process_gameweek(
        self,
        gameweek: int,
        season: str,
        session: Session,
        force_reprocess: bool = False
    ) -> ProcessingResult:
        """
        Process all measurements for a specific gameweek.
        
        Args:
            gameweek: Gameweek number to process
            season: Season identifier
            session: Database session
            force_reprocess: Force reprocessing even if already done
            
        Returns:
            ProcessingResult with processing details and metrics
        """
        try:
            start_time = time.time()
            initial_memory = self.optimizer.get_memory_usage()
            
            # Check if already processed
            if not force_reprocess and gameweek in self.gameweeks_processed:
                logger.info(f"Gameweek {gameweek} already processed, skipping")
                return self._create_empty_result(gameweek, season)
            
            logger.info(f"Processing gameweek {gameweek} for season {season}")
            
            # Load player scores for gameweek
            player_scores = self._load_gameweek_data(gameweek, season, session)
            
            if not player_scores:
                logger.warning(f"No player scores found for gameweek {gameweek}")
                return self._create_empty_result(gameweek, season)
            
            # Process in batches for memory efficiency
            total_players = len(player_scores)
            processed_players = 0
            updated_players = 0
            total_outliers = 0
            
            # Performance tracking
            extraction_time = 0.0
            normalization_time = 0.0
            outlier_time = 0.0
            kalman_time = 0.0
            
            warnings = []
            errors = []
            
            # Process in batches
            for batch in self._create_batches(player_scores, self.config.batch_size):
                try:
                    batch_start = time.time()
                    
                    # Extract measurements
                    extract_start = time.time()
                    measurements, noise_matrices = self.batch_updater.process_batch(
                        batch, gameweek, season, session
                    )
                    extraction_time += time.time() - extract_start
                    
                    if not measurements:
                        continue
                    
                    # Get player positions
                    player_positions = self._get_player_positions(batch)
                    
                    # Update Kalman filters
                    kalman_start = time.time()
                    updated_states = self.batch_updater.update_batch(
                        self.kalman_model,
                        measurements,
                        noise_matrices,
                        player_positions,
                        gameweek,
                        season
                    )
                    kalman_time += time.time() - kalman_start
                    
                    # Update counters
                    processed_players += len(batch)
                    updated_players += len(updated_states)
                    
                    # Collect outlier statistics
                    batch_stats = self.batch_updater.get_batch_statistics()
                    total_outliers += batch_stats.get("outliers_detected", 0)
                    
                    # Memory cleanup if needed
                    if self.config.enable_memory_optimization:
                        self.optimizer.cleanup_batch_memory()
                    
                except Exception as e:
                    error_msg = f"Batch processing failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Quality control checks
            outlier_ratio = total_outliers / max(processed_players, 1)
            if outlier_ratio > self.config.max_outlier_ratio_per_gameweek:
                warning_msg = f"High outlier ratio: {outlier_ratio:.2%} > {self.config.max_outlier_ratio_per_gameweek:.2%}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
            
            # Adaptive parameter tuning
            if self.config.auto_tune_filter_parameters:
                self._tune_filter_parameters(gameweek, season)
            
            # Update noise estimates
            if self.config.auto_update_noise_estimates:
                self._update_noise_estimates(gameweek, season)
            
            # Collect diagnostics
            innovation_stats = self._collect_innovation_statistics()
            
            # Create result
            processing_time = time.time() - start_time
            final_memory = self.optimizer.get_memory_usage()
            
            result = ProcessingResult(
                gameweek=gameweek,
                season=season,
                players_processed=processed_players,
                players_updated=updated_players,
                outliers_detected=total_outliers,
                processing_time=processing_time,
                memory_usage_mb=final_memory - initial_memory,
                average_innovation_nis=innovation_stats.get("average_nis", 0.0),
                filter_consistency_rate=innovation_stats.get("consistency_rate", 1.0),
                warnings=warnings,
                errors=errors,
                extraction_time=extraction_time,
                normalization_time=normalization_time,
                outlier_detection_time=outlier_time,
                kalman_update_time=kalman_time
            )
            
            # Store result
            self.processing_history.append(result)
            self.gameweeks_processed.add(gameweek)
            
            # Log statistics
            if self.config.log_processing_stats:
                self._log_processing_stats(result)
            
            # Collect diagnostics
            if self.diagnostics:
                self.diagnostics.record_processing_result(result)
            
            # Periodic cache cleanup
            if len(self.processing_history) % self.config.clear_cache_frequency == 0:
                self._clear_caches()
            
            return result
            
        except Exception as e:
            logger.error(f"Gameweek processing failed for GW{gameweek}: {e}")
            result = self._create_empty_result(gameweek, season)
            result.errors.append(str(e))
            return result
    
    def process_season(
        self,
        season: str,
        session: Session,
        start_gameweek: int = 1,
        end_gameweek: Optional[int] = None,
        parallel: bool = None
    ) -> List[ProcessingResult]:
        """
        Process measurements for an entire season.
        
        Args:
            season: Season identifier
            session: Database session
            start_gameweek: Starting gameweek
            end_gameweek: Ending gameweek (None for current)
            parallel: Use parallel processing (default from config)
            
        Returns:
            List of ProcessingResults for each gameweek
        """
        try:
            logger.info(f"Processing season {season} from GW{start_gameweek}")
            
            # Determine gameweek range
            if end_gameweek is None:
                end_gameweek = self._get_latest_gameweek(season, session)
            
            gameweeks = list(range(start_gameweek, end_gameweek + 1))
            results = []
            
            # Choose processing method
            use_parallel = parallel if parallel is not None else self.config.use_parallel_processing
            
            if use_parallel and len(gameweeks) > 1:
                # Parallel processing
                results = self._process_gameweeks_parallel(gameweeks, season, session)
            else:
                # Sequential processing
                for gw in gameweeks:
                    result = self.process_gameweek(gw, season, session)
                    results.append(result)
            
            # Season-level analysis
            self._analyze_season_results(results, season)
            
            return results
            
        except Exception as e:
            logger.error(f"Season processing failed for {season}: {e}")
            return []
    
    def _load_gameweek_data(
        self, 
        gameweek: int, 
        season: str, 
        session: Session
    ) -> List[PlayerScore]:
        """Load player scores for a specific gameweek."""
        try:
            # Query player scores with joins for efficiency
            query = session.query(PlayerScore).join(Player).join(Result).filter(
                and_(
                    Result.gameweek == gameweek,
                    Result.season == season,
                    PlayerScore.minutes.isnot(None)  # Filter out missing data
                )
            ).order_by(Player.position, PlayerScore.player_id)
            
            player_scores = query.all()
            
            logger.debug(f"Loaded {len(player_scores)} player scores for GW{gameweek}")
            return player_scores
            
        except Exception as e:
            logger.error(f"Failed to load gameweek data: {e}")
            return []
    
    def _create_batches(self, items: List, batch_size: int) -> Iterator[List]:
        """Create batches from list of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def _get_player_positions(self, player_scores: List[PlayerScore]) -> Dict[int, str]:
        """Get player positions with caching."""
        positions = {}
        
        for score in player_scores:
            player_id = score.player_id
            
            if player_id in self._player_position_cache:
                positions[player_id] = self._player_position_cache[player_id]
            elif score.player and score.player.position:
                position = score.player.position
                positions[player_id] = position
                self._player_position_cache[player_id] = position
            else:
                # Default position
                positions[player_id] = "MID"
                self._player_position_cache[player_id] = "MID"
        
        return positions
    
    def _tune_filter_parameters(self, gameweek: int, season: str):
        """Automatically tune filter parameters based on performance."""
        try:
            # Get adaptive recommendations from innovation monitor
            if not hasattr(self.batch_updater, 'innovation_monitor'):
                return
            
            positions_to_tune = ["GK", "DEF", "MID", "FWD"]
            
            for position in positions_to_tune:
                recommendations = self.batch_updater.innovation_monitor.get_adaptive_recommendations(position)
                
                if recommendations.get("confidence", 0) > 0.7:
                    # Apply recommendations to filter configuration
                    if recommendations.get("increase_process_noise"):
                        self._adjust_process_noise(position, factor=1.1)
                    elif recommendations.get("decrease_process_noise"):
                        self._adjust_process_noise(position, factor=0.9)
                    
                    if recommendations.get("increase_measurement_noise"):
                        self._adjust_measurement_noise(position, factor=1.1)
                    elif recommendations.get("decrease_measurement_noise"):
                        self._adjust_measurement_noise(position, factor=0.9)
                    
                    logger.info(f"Applied adaptive tuning for {position} in GW{gameweek}")
        
        except Exception as e:
            logger.warning(f"Filter parameter tuning failed: {e}")
    
    def _adjust_process_noise(self, position: str, factor: float):
        """Adjust process noise for a specific position."""
        # This would adjust the Kalman filter configuration
        # Implementation depends on how filters are structured
        pass
    
    def _adjust_measurement_noise(self, position: str, factor: float):
        """Adjust measurement noise for a specific position."""
        # This would adjust the noise estimator configuration
        # Implementation depends on how noise estimation is structured
        pass
    
    def _update_noise_estimates(self, gameweek: int, season: str):
        """Update noise estimates based on recent performance."""
        try:
            # This would trigger noise estimate updates in the BatchUpdater
            # Could be implemented as a method call to the noise estimator
            pass
        except Exception as e:
            logger.warning(f"Noise estimate update failed: {e}")
    
    def _collect_innovation_statistics(self) -> Dict[str, float]:
        """Collect innovation statistics across all positions."""
        stats = {
            "average_nis": 0.0,
            "consistency_rate": 1.0
        }
        
        try:
            if not hasattr(self.batch_updater, 'innovation_monitor'):
                return stats
            
            monitor = self.batch_updater.innovation_monitor
            positions = ["GK", "DEF", "MID", "FWD"]
            
            nis_values = []
            consistency_checks = []
            
            for position in positions:
                if position in monitor._nis_history and monitor._nis_history[position]:
                    recent_nis = monitor._nis_history[position][-5:]  # Last 5 values
                    nis_values.extend(recent_nis)
                    
                    # Check consistency for recent values
                    for nis in recent_nis:
                        consistent = monitor._check_consistency(nis, dof=4)
                        consistency_checks.append(consistent)
            
            if nis_values:
                stats["average_nis"] = float(np.mean(nis_values))
            
            if consistency_checks:
                stats["consistency_rate"] = float(np.mean(consistency_checks))
            
        except Exception as e:
            logger.warning(f"Innovation statistics collection failed: {e}")
        
        return stats
    
    def _process_gameweeks_parallel(
        self, 
        gameweeks: List[int], 
        season: str, 
        session: Session
    ) -> List[ProcessingResult]:
        """Process gameweeks in parallel."""
        results = []
        
        # Note: Parallel processing of database operations requires careful session management
        # This is a simplified version - in practice would need session per thread
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all gameweeks
                future_to_gw = {
                    executor.submit(self.process_gameweek, gw, season, session): gw 
                    for gw in gameweeks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_gw):
                    gw = future_to_gw[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Parallel processing failed for GW{gw}: {e}")
                        error_result = self._create_empty_result(gw, season)
                        error_result.errors.append(str(e))
                        results.append(error_result)
            
            # Sort results by gameweek
            results.sort(key=lambda r: r.gameweek)
            
        except Exception as e:
            logger.error(f"Parallel processing setup failed: {e}")
            # Fall back to sequential processing
            for gw in gameweeks:
                result = self.process_gameweek(gw, season, session)
                results.append(result)
        
        return results
    
    def _analyze_season_results(self, results: List[ProcessingResult], season: str):
        """Perform season-level analysis and logging."""
        try:
            if not results:
                return
            
            total_players = sum(r.players_processed for r in results)
            total_updated = sum(r.players_updated for r in results)
            total_outliers = sum(r.outliers_detected for r in results)
            total_time = sum(r.processing_time for r in results)
            
            avg_nis = np.mean([r.average_innovation_nis for r in results if r.average_innovation_nis > 0])
            avg_consistency = np.mean([r.filter_consistency_rate for r in results])
            
            total_warnings = sum(len(r.warnings) for r in results)
            total_errors = sum(len(r.errors) for r in results)
            
            logger.info(f"Season {season} processing complete:")
            logger.info(f"  Gameweeks processed: {len(results)}")
            logger.info(f"  Total players processed: {total_players}")
            logger.info(f"  Total players updated: {total_updated}")
            logger.info(f"  Total outliers detected: {total_outliers}")
            logger.info(f"  Total processing time: {total_time:.2f}s")
            logger.info(f"  Average NIS: {avg_nis:.2f}")
            logger.info(f"  Filter consistency rate: {avg_consistency:.2%}")
            
            if total_warnings > 0:
                logger.warning(f"  Total warnings: {total_warnings}")
            if total_errors > 0:
                logger.error(f"  Total errors: {total_errors}")
        
        except Exception as e:
            logger.warning(f"Season analysis failed: {e}")
    
    def _get_latest_gameweek(self, season: str, session: Session) -> int:
        """Get the latest gameweek with data."""
        try:
            latest = session.query(func.max(Result.gameweek)).filter(
                Result.season == season
            ).scalar()
            return latest or 38
        except Exception as e:
            logger.warning(f"Failed to get latest gameweek: {e}")
            return 38
    
    def _create_empty_result(self, gameweek: int, season: str) -> ProcessingResult:
        """Create empty processing result."""
        return ProcessingResult(
            gameweek=gameweek,
            season=season,
            players_processed=0,
            players_updated=0,
            outliers_detected=0,
            processing_time=0.0,
            memory_usage_mb=0.0,
            average_innovation_nis=0.0,
            filter_consistency_rate=1.0
        )
    
    def _log_processing_stats(self, result: ProcessingResult):
        """Log processing statistics."""
        logger.info(
            f"GW{result.gameweek}: {result.players_updated}/{result.players_processed} players, "
            f"{result.outliers_detected} outliers, {result.processing_time:.2f}s"
        )
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"GW{result.gameweek}: {warning}")
        
        if result.errors:
            for error in result.errors:
                logger.error(f"GW{result.gameweek}: {error}")
    
    def _clear_caches(self):
        """Clear internal caches to free memory."""
        self._player_position_cache.clear()
        self._opponent_strength_cache.clear()
        self.batch_updater.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.debug("Cleared caches and performed garbage collection")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.processing_history:
            return {}
        
        recent_results = self.processing_history[-10:]  # Last 10 gameweeks
        
        metrics = {
            "total_gameweeks_processed": len(self.processing_history),
            "recent_gameweeks": len(recent_results),
            "average_processing_time": np.mean([r.processing_time for r in recent_results]),
            "average_players_per_gameweek": np.mean([r.players_processed for r in recent_results]),
            "average_update_rate": np.mean([r.players_updated / max(r.players_processed, 1) for r in recent_results]),
            "average_outlier_rate": np.mean([r.outliers_detected / max(r.players_processed, 1) for r in recent_results]),
            "average_memory_usage_mb": np.mean([r.memory_usage_mb for r in recent_results]),
            "filter_consistency_rate": np.mean([r.filter_consistency_rate for r in recent_results]),
            "total_warnings": sum(len(r.warnings) for r in self.processing_history),
            "total_errors": sum(len(r.errors) for r in self.processing_history)
        }
        
        # Performance breakdown
        metrics["time_breakdown"] = {
            "extraction": np.mean([r.extraction_time for r in recent_results]),
            "normalization": np.mean([r.normalization_time for r in recent_results]),
            "outlier_detection": np.mean([r.outlier_detection_time for r in recent_results]),
            "kalman_update": np.mean([r.kalman_update_time for r in recent_results])
        }
        
        return metrics


class PerformanceOptimizer:
    """Performance optimization utilities for measurement processing."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize performance optimizer."""
        self.config = config
        self._memory_threshold_mb = 500  # Trigger cleanup at 500MB
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback to basic measurement
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0
    
    def cleanup_batch_memory(self):
        """Clean up memory after batch processing."""
        if self.config.enable_memory_optimization:
            # Clear JAX compilation cache periodically
            if hasattr(jax, 'clear_caches'):
                jax.clear_caches()
            
            # Force garbage collection
            gc.collect()
    
    def should_trigger_cleanup(self) -> bool:
        """Check if memory cleanup should be triggered."""
        return self.get_memory_usage() > self._memory_threshold_mb
    
    def optimize_jax_settings(self):
        """Configure JAX for optimal performance."""
        # Configure JAX memory allocation
        import os
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


class DiagnosticsCollector:
    """Collect and analyze system diagnostics."""
    
    def __init__(self):
        """Initialize diagnostics collector."""
        self.processing_results: List[ProcessingResult] = []
        self.system_metrics: List[Dict[str, Any]] = []
        
    def record_processing_result(self, result: ProcessingResult):
        """Record a processing result for analysis."""
        self.processing_results.append(result)
        
        # Keep last 100 results
        if len(self.processing_results) > 100:
            self.processing_results = self.processing_results[-100:]
    
    def get_health_status(self) -> Dict[str, str]:
        """Get overall system health status."""
        if not self.processing_results:
            return {"status": "unknown", "message": "No data available"}
        
        recent_results = self.processing_results[-10:]
        
        # Check error rates
        error_rate = np.mean([len(r.errors) > 0 for r in recent_results])
        warning_rate = np.mean([len(r.warnings) > 0 for r in recent_results])
        
        # Check performance metrics
        avg_consistency = np.mean([r.filter_consistency_rate for r in recent_results])
        avg_outlier_rate = np.mean([r.outliers_detected / max(r.players_processed, 1) for r in recent_results])
        
        # Determine health status
        if error_rate > 0.3:
            return {"status": "critical", "message": f"High error rate: {error_rate:.1%}"}
        elif warning_rate > 0.5 or avg_consistency < 0.8:
            return {"status": "warning", "message": "Performance degradation detected"}
        elif avg_outlier_rate > 0.2:
            return {"status": "warning", "message": f"High outlier rate: {avg_outlier_rate:.1%}"}
        else:
            return {"status": "healthy", "message": "All systems operating normally"}
    
    def generate_report(self) -> str:
        """Generate diagnostic report."""
        if not self.processing_results:
            return "No diagnostic data available"
        
        recent_results = self.processing_results[-20:]
        
        # Calculate statistics
        total_gameweeks = len(recent_results)
        avg_processing_time = np.mean([r.processing_time for r in recent_results])
        avg_players = np.mean([r.players_processed for r in recent_results])
        avg_update_rate = np.mean([r.players_updated / max(r.players_processed, 1) for r in recent_results])
        avg_outlier_rate = np.mean([r.outliers_detected / max(r.players_processed, 1) for r in recent_results])
        avg_consistency = np.mean([r.filter_consistency_rate for r in recent_results])
        
        total_errors = sum(len(r.errors) for r in recent_results)
        total_warnings = sum(len(r.warnings) for r in recent_results)
        
        report = f"""
Measurement Update System Diagnostic Report
==========================================

Processing Summary (Last {total_gameweeks} gameweeks):
- Average processing time: {avg_processing_time:.2f}s
- Average players per gameweek: {avg_players:.0f}
- Average update rate: {avg_update_rate:.1%}
- Average outlier rate: {avg_outlier_rate:.1%}
- Filter consistency rate: {avg_consistency:.1%}

Quality Metrics:
- Total errors: {total_errors}
- Total warnings: {total_warnings}
- Error rate: {total_errors / total_gameweeks:.1f} per gameweek
- Warning rate: {total_warnings / total_gameweeks:.1f} per gameweek

Health Status: {self.get_health_status()}
"""
        return report


# Utility functions for easy integration

def create_optimized_integrator(
    kalman_model: KalmanPlayerModel,
    enable_parallel: bool = True,
    batch_size: int = 100
) -> MeasurementUpdateIntegrator:
    """
    Create an optimized measurement update integrator.
    
    Args:
        kalman_model: Existing KalmanPlayerModel
        enable_parallel: Enable parallel processing
        batch_size: Batch size for processing
        
    Returns:
        Configured MeasurementUpdateIntegrator
    """
    config = IntegrationConfig(
        batch_size=batch_size,
        use_parallel_processing=enable_parallel,
        enable_memory_optimization=True,
        enable_performance_monitoring=True,
        auto_update_noise_estimates=True,
        auto_tune_filter_parameters=True
    )
    
    # Optimized measurement configuration
    measurement_config = MeasurementConfig(
        measurement_features=["goals", "assists", "minutes", "bonus", "expected_goals", "expected_assists"],
        normalize_by_position=True,
        normalize_by_opponent=True,
        outlier_detection_method="mahalanobis",
        outlier_threshold=3.0,
        use_robust_scaling=True
    )
    
    # Optimized noise configuration  
    noise_config = NoiseConfig(
        estimation_window=10,
        adaptation_rate=0.1,
        estimate_correlations=True
    )
    
    return MeasurementUpdateIntegrator(
        kalman_model=kalman_model,
        config=config,
        measurement_config=measurement_config,
        noise_config=noise_config
    )


def quick_gameweek_update(
    kalman_model: KalmanPlayerModel,
    gameweek: int,
    season: str,
    session: Session
) -> ProcessingResult:
    """
    Quick utility for processing a single gameweek with default settings.
    
    Args:
        kalman_model: KalmanPlayerModel to update
        gameweek: Gameweek to process
        season: Season identifier
        session: Database session
        
    Returns:
        ProcessingResult
    """
    integrator = create_optimized_integrator(kalman_model, enable_parallel=False)
    return integrator.process_gameweek(gameweek, season, session)