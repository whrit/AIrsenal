"""add_model_persistence_tables

Revision ID: b180bb12988e
Revises: a55a2bfb8b17
Create Date: 2025-08-17 18:34:29.491968

This migration adds comprehensive model persistence infrastructure for AIrsenal's
machine learning models, implementing TASK-203 from Sprint 02.

New tables added:
- model_checkpoint: Storage for complete model checkpoints with compression and integrity
- model_state: Individual player model states for granular persistence
- model_metadata: Extended metadata for model configurations and tracking

Features:
- Compression support (gzip, bz2, lzma) for efficient storage
- Integrity verification using SHA256 checksums
- Support for multiple storage backends (database, file, cloud)
- Comprehensive metadata tracking for experiments and model lineage
- Automatic checkpoint management and recovery capabilities
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b180bb12988e'
down_revision: str | Sequence[str] | None = 'a55a2bfb8b17'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add model persistence tables for checkpoint management and state storage."""
    logger = logging.getLogger(__name__)
    logger.info("Adding model persistence tables...")

    connection = op.get_bind()
    dialect_name = connection.dialect.name
    logger.info(f"Detected database dialect: {dialect_name}")

    # Create ModelCheckpoint table
    logger.info("Creating model_checkpoint table...")
    op.create_table(
        'model_checkpoint',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('version_id', sa.Integer(), nullable=False),
        sa.Column('checkpoint_name', sa.String(length=100), nullable=False),
        sa.Column('checkpoint_type', sa.String(length=100), nullable=False),
        
        # Model state storage
        sa.Column('model_data', sa.LargeBinary(), nullable=True),
        sa.Column('compression_format', sa.String(length=100), nullable=False, server_default='gzip'),
        sa.Column('original_size_bytes', sa.Integer(), nullable=True),
        sa.Column('compressed_size_bytes', sa.Integer(), nullable=True),
        
        # Integrity verification
        sa.Column('checksum', sa.String(length=100), nullable=False),
        sa.Column('checksum_algorithm', sa.String(length=100), nullable=False, server_default='sha256'),
        
        # Storage metadata
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('cloud_path', sa.String(length=500), nullable=True),
        sa.Column('storage_backend', sa.String(length=100), nullable=False, server_default='database'),
        
        # Checkpoint metadata
        sa.Column('created_at', sa.String(length=100), nullable=False),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('gameweek', sa.Integer(), nullable=True),
        sa.Column('season', sa.String(length=100), nullable=True),
        sa.Column('tags', sa.String(length=500), nullable=True),
        sa.Column('description', sa.String(length=1000), nullable=True),
        
        # Performance metrics
        sa.Column('validation_score', sa.Float(), nullable=True),
        sa.Column('training_loss', sa.Float(), nullable=True),
        sa.Column('model_size_mb', sa.Float(), nullable=True),
        
        # Status and lifecycle
        sa.Column('status', sa.String(length=100), nullable=False, server_default='active'),
        sa.Column('is_recoverable', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('recovery_priority', sa.Integer(), nullable=False, server_default='1'),
        
        sa.ForeignKeyConstraint(['version_id'], ['model_version.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create ModelState table
    logger.info("Creating model_state table...")
    op.create_table(
        'model_state',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('checkpoint_id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        
        # State vector and covariance data
        sa.Column('state_mean', sa.LargeBinary(), nullable=True),
        sa.Column('state_covariance', sa.LargeBinary(), nullable=True),
        sa.Column('state_history', sa.LargeBinary(), nullable=True),
        
        # State metadata
        sa.Column('state_dimension', sa.Integer(), nullable=False),
        sa.Column('gameweek', sa.Integer(), nullable=False),
        sa.Column('season', sa.String(length=100), nullable=False),
        sa.Column('last_updated', sa.String(length=100), nullable=False),
        
        # Serialization metadata
        sa.Column('serialization_format', sa.String(length=100), nullable=False, server_default='numpy'),
        sa.Column('compression_used', sa.Boolean(), nullable=False, server_default='1'),
        
        # State validation
        sa.Column('state_checksum', sa.String(length=100), nullable=False),
        sa.Column('is_valid', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('validation_errors', sa.String(length=1000), nullable=True),
        
        # Performance tracking
        sa.Column('prediction_accuracy', sa.Float(), nullable=True),
        sa.Column('uncertainty_score', sa.Float(), nullable=True),
        sa.Column('update_frequency', sa.Integer(), nullable=False, server_default='0'),
        
        sa.ForeignKeyConstraint(['checkpoint_id'], ['model_checkpoint.id'], ),
        sa.ForeignKeyConstraint(['player_id'], ['player.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create ModelMetadata table
    logger.info("Creating model_metadata table...")
    op.create_table(
        'model_metadata',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('version_id', sa.Integer(), nullable=False),
        
        # Model configuration
        sa.Column('state_space_config', sa.String(length=2000), nullable=True),
        sa.Column('hyperparameters', sa.String(length=2000), nullable=True),
        sa.Column('feature_config', sa.String(length=2000), nullable=True),
        
        # Training configuration
        sa.Column('training_algorithm', sa.String(length=100), nullable=True),
        sa.Column('optimizer_config', sa.String(length=1000), nullable=True),
        sa.Column('regularization_config', sa.String(length=1000), nullable=True),
        
        # Data configuration
        sa.Column('training_data_sources', sa.String(length=1000), nullable=True),
        sa.Column('feature_selection_method', sa.String(length=100), nullable=True),
        sa.Column('data_preprocessing_steps', sa.String(length=1000), nullable=True),
        
        # Performance metrics
        sa.Column('convergence_metrics', sa.String(length=1000), nullable=True),
        sa.Column('computational_metrics', sa.String(length=1000), nullable=True),
        sa.Column('memory_usage_metrics', sa.String(length=1000), nullable=True),
        
        # Experiment tracking
        sa.Column('experiment_id', sa.String(length=100), nullable=True),
        sa.Column('run_id', sa.String(length=100), nullable=True),
        sa.Column('parent_run_id', sa.String(length=100), nullable=True),
        
        # Model lineage
        sa.Column('derived_from_version_id', sa.Integer(), nullable=True),
        sa.Column('inheritance_type', sa.String(length=100), nullable=True),
        sa.Column('modification_summary', sa.String(length=1000), nullable=True),
        
        # Quality metrics
        sa.Column('data_quality_score', sa.Float(), nullable=True),
        sa.Column('model_complexity_score', sa.Float(), nullable=True),
        sa.Column('interpretability_score', sa.Float(), nullable=True),
        
        # Deployment metadata
        sa.Column('deployment_requirements', sa.String(length=1000), nullable=True),
        sa.Column('compatibility_info', sa.String(length=1000), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.String(length=100), nullable=False),
        sa.Column('updated_at', sa.String(length=100), nullable=False),
        
        sa.ForeignKeyConstraint(['derived_from_version_id'], ['model_version.id'], ),
        sa.ForeignKeyConstraint(['version_id'], ['model_version.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create performance indexes
    logger.info("Creating performance indexes...")
    
    # ModelCheckpoint indexes
    op.create_index('idx_model_checkpoint_version', 'model_checkpoint', ['version_id'])
    op.create_index('idx_model_checkpoint_type_status', 'model_checkpoint', ['checkpoint_type', 'status'])
    op.create_index('idx_model_checkpoint_created_at', 'model_checkpoint', ['created_at'])
    op.create_index('idx_model_checkpoint_season_gw', 'model_checkpoint', ['season', 'gameweek'])
    
    # ModelState indexes
    op.create_index('idx_model_state_checkpoint', 'model_state', ['checkpoint_id'])
    op.create_index('idx_model_state_player', 'model_state', ['player_id'])
    op.create_index('idx_model_state_player_season', 'model_state', ['player_id', 'season'])
    op.create_index('idx_model_state_gameweek', 'model_state', ['gameweek'])
    op.create_index('idx_model_state_valid', 'model_state', ['is_valid'])
    
    # ModelMetadata indexes
    op.create_index('idx_model_metadata_version', 'model_metadata', ['version_id'])
    op.create_index('idx_model_metadata_algorithm', 'model_metadata', ['training_algorithm'])
    op.create_index('idx_model_metadata_experiment', 'model_metadata', ['experiment_id'])
    op.create_index('idx_model_metadata_lineage', 'model_metadata', ['derived_from_version_id'])

    logger.info("Model persistence tables created successfully!")


def downgrade() -> None:
    """Remove model persistence tables and related indexes."""
    logger = logging.getLogger(__name__)
    logger.info("Removing model persistence tables...")

    # Drop indexes first
    logger.info("Dropping indexes...")
    
    # ModelMetadata indexes
    op.drop_index('idx_model_metadata_lineage', table_name='model_metadata')
    op.drop_index('idx_model_metadata_experiment', table_name='model_metadata')
    op.drop_index('idx_model_metadata_algorithm', table_name='model_metadata')
    op.drop_index('idx_model_metadata_version', table_name='model_metadata')
    
    # ModelState indexes
    op.drop_index('idx_model_state_valid', table_name='model_state')
    op.drop_index('idx_model_state_gameweek', table_name='model_state')
    op.drop_index('idx_model_state_player_season', table_name='model_state')
    op.drop_index('idx_model_state_player', table_name='model_state')
    op.drop_index('idx_model_state_checkpoint', table_name='model_state')
    
    # ModelCheckpoint indexes
    op.drop_index('idx_model_checkpoint_season_gw', table_name='model_checkpoint')
    op.drop_index('idx_model_checkpoint_created_at', table_name='model_checkpoint')
    op.drop_index('idx_model_checkpoint_type_status', table_name='model_checkpoint')
    op.drop_index('idx_model_checkpoint_version', table_name='model_checkpoint')

    # Drop tables in reverse order (respecting foreign key dependencies)
    logger.info("Dropping tables...")
    op.drop_table('model_metadata')
    op.drop_table('model_state')
    op.drop_table('model_checkpoint')

    logger.info("Model persistence tables removed successfully!")
