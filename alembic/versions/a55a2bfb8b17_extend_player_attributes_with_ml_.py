"""extend_player_attributes_with_ml_features

Revision ID: a55a2bfb8b17
Revises:
Create Date: 2025-08-16 13:04:44.195978

This migration extends the PlayerAttributes table with machine learning features
including xG metrics, form indicators, fixture difficulty ratings, player roles,
and advanced performance statistics to support enhanced FPL predictions.

Features added:
- Expected Goals (xG) metrics: xg_per_90, xa_per_90, xgi_per_90
- Form metrics: form_3_games, form_5_games, form_10_games, momentum
- Fixture difficulty: next_3_fixture_difficulty, next_5_fixture_difficulty
- Player roles: is_penalty_taker, is_free_kick_taker, is_corner_taker, role_confidence
- Advanced stats: shots_per_90, key_passes_per_90, tackles_per_90, interceptions_per_90, clearances_per_90

Performance indexes added for optimized query patterns.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a55a2bfb8b17"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Extend PlayerAttributes table with ML features and performance indexes."""
    logger = logging.getLogger(__name__)
    logger.info("Starting PlayerAttributes ML features migration...")

    connection = op.get_bind()
    dialect_name = connection.dialect.name
    logger.info(f"Detected database dialect: {dialect_name}")

    # Check if table exists before proceeding
    inspector = sa.inspect(connection)
    if "player_attributes" not in inspector.get_table_names():
        logger.error(
            "player_attributes table not found! Cannot proceed with migration."
        )
        msg = "PlayerAttributes table does not exist"
        raise Exception(msg)

    # Get existing columns to avoid duplicate additions
    existing_columns = {
        col["name"] for col in inspector.get_columns("player_attributes")
    }
    logger.info(f"Found {len(existing_columns)} existing columns in player_attributes")

    # Use batch operations for SQLite compatibility
    with op.batch_alter_table("player_attributes", schema=None) as batch_op:
        # Add Expected Goals (xG) metrics
        if "xg_per_90" not in existing_columns:
            logger.info("Adding xG metrics columns...")
            batch_op.add_column(
                sa.Column(
                    "xg_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Expected goals per 90 minutes",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "xa_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Expected assists per 90 minutes",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "xgi_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Expected goal involvements (xG + xA) per 90 minutes",
                )
            )
        else:
            logger.info("xG metrics columns already exist, skipping...")

        # Add form metrics - rolling averages of FPL points
        if "form_3_games" not in existing_columns:
            logger.info("Adding form metrics columns...")
            batch_op.add_column(
                sa.Column(
                    "form_3_games",
                    sa.Float(),
                    nullable=True,
                    comment="Average FPL points over last 3 games",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "form_5_games",
                    sa.Float(),
                    nullable=True,
                    comment="Average FPL points over last 5 games",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "form_10_games",
                    sa.Float(),
                    nullable=True,
                    comment="Average FPL points over last 10 games",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "momentum",
                    sa.Float(),
                    nullable=True,
                    comment="Trend indicator for recent performance (-1 to 1)",
                )
            )
        else:
            logger.info("Form metrics columns already exist, skipping...")

        # Add fixture difficulty ratings
        if "next_3_fixture_difficulty" not in existing_columns:
            logger.info("Adding fixture difficulty columns...")
            batch_op.add_column(
                sa.Column(
                    "next_3_fixture_difficulty",
                    sa.Float(),
                    nullable=True,
                    comment="Average difficulty rating for next 3 fixtures (1-5 scale)",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "next_5_fixture_difficulty",
                    sa.Float(),
                    nullable=True,
                    comment="Average difficulty rating for next 5 fixtures (1-5 scale)",
                )
            )
        else:
            logger.info("Fixture difficulty columns already exist, skipping...")

        # Add player role indicators with proper SQLite handling
        if "is_penalty_taker" not in existing_columns:
            logger.info("Adding player role columns...")
            batch_op.add_column(
                sa.Column(
                    "is_penalty_taker",
                    sa.Boolean(),
                    nullable=True,
                    comment="Whether player is primary penalty taker",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "is_free_kick_taker",
                    sa.Boolean(),
                    nullable=True,
                    comment="Whether player is primary free kick taker",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "is_corner_taker",
                    sa.Boolean(),
                    nullable=True,
                    comment="Whether player is primary corner taker",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "role_confidence",
                    sa.Float(),
                    nullable=True,
                    comment="Confidence score for set piece roles (0-1 scale)",
                )
            )
        else:
            logger.info("Player role columns already exist, skipping...")

        # Add advanced performance statistics per 90 minutes
        if "shots_per_90" not in existing_columns:
            logger.info("Adding advanced performance statistics columns...")
            batch_op.add_column(
                sa.Column(
                    "shots_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Total shots per 90 minutes",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "key_passes_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Key passes (passes leading to shots) per 90 minutes",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "tackles_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Successful tackles per 90 minutes",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "interceptions_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Interceptions per 90 minutes",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "clearances_per_90",
                    sa.Float(),
                    nullable=True,
                    comment="Clearances per 90 minutes",
                )
            )
        else:
            logger.info(
                "Advanced performance statistics columns already exist, skipping..."
            )

    # Initialize boolean role columns with default values if they were just added
    if "is_penalty_taker" not in existing_columns:
        logger.info("Initializing boolean role columns with default values...")
        connection.execute(
            sa.text("""
            UPDATE player_attributes
            SET is_penalty_taker = 0,
                is_free_kick_taker = 0,
                is_corner_taker = 0
            WHERE is_penalty_taker IS NULL
               OR is_free_kick_taker IS NULL
               OR is_corner_taker IS NULL
        """)
        )
        logger.info("Boolean role columns initialized successfully")

    # Create performance-optimized indexes with requested naming convention
    logger.info("Creating performance indexes...")
    existing_indexes = {
        idx["name"] for idx in inspector.get_indexes("player_attributes")
    }

    indexes_to_create = [
        # Season/gameweek composite indexes
        ("idx_player_attributes_season_gw", ["season", "gameweek"]),
        ("idx_player_attributes_player_season", ["player_id", "season"]),
        # Form metrics indexes
        ("idx_player_attributes_form_3", ["form_3_games"]),
        ("idx_player_attributes_form_5", ["form_5_games"]),
        # xG metrics indexes
        ("idx_player_attributes_xg", ["xg_per_90"]),
        ("idx_player_attributes_xgi", ["xgi_per_90"]),
        # Fixture difficulty index
        ("idx_player_attributes_fixture_difficulty", ["next_3_fixture_difficulty"]),
        # Player roles composite index
        ("idx_player_attributes_roles", ["position", "is_penalty_taker"]),
        # Momentum index
        ("idx_player_attributes_momentum", ["momentum"]),
    ]

    created_count = 0
    for index_name, columns in indexes_to_create:
        if index_name not in existing_indexes:
            try:
                op.create_index(index_name, "player_attributes", columns, unique=False)
                logger.info(f"Created index: {index_name}")
                created_count += 1
            except Exception as e:
                logger.warning(f"Failed to create index {index_name}: {e}")
        else:
            logger.info(f"Index {index_name} already exists, skipping...")

    logger.info(
        f"Migration completed successfully! Created {created_count} new indexes."
    )


def downgrade() -> None:
    """Remove ML features and indexes from PlayerAttributes table."""
    logger = logging.getLogger(__name__)
    logger.info("Starting PlayerAttributes ML features rollback...")

    connection = op.get_bind()
    dialect_name = connection.dialect.name
    logger.info(f"Detected database dialect: {dialect_name}")

    # Check if table exists before proceeding
    inspector = sa.inspect(connection)
    if "player_attributes" not in inspector.get_table_names():
        logger.warning(
            "player_attributes table not found! Migration already rolled back or table doesn't exist."
        )
        return

    # Get existing columns and indexes to avoid errors when dropping non-existent items
    existing_columns = {
        col["name"] for col in inspector.get_columns("player_attributes")
    }
    existing_indexes = {
        idx["name"] for idx in inspector.get_indexes("player_attributes")
    }

    logger.info(
        f"Found {len(existing_columns)} existing columns and {len(existing_indexes)} existing indexes"
    )

    # Drop indexes first (with updated naming convention)
    logger.info("Dropping performance indexes...")
    indexes_to_drop = [
        "idx_player_attributes_momentum",
        "idx_player_attributes_roles",
        "idx_player_attributes_fixture_difficulty",
        "idx_player_attributes_xgi",
        "idx_player_attributes_xg",
        "idx_player_attributes_form_5",
        "idx_player_attributes_form_3",
        "idx_player_attributes_player_season",
        "idx_player_attributes_season_gw",
    ]

    dropped_indexes = 0
    for index_name in indexes_to_drop:
        if index_name in existing_indexes:
            try:
                op.drop_index(index_name, table_name="player_attributes")
                logger.info(f"Dropped index: {index_name}")
                dropped_indexes += 1
            except Exception as e:
                logger.warning(f"Failed to drop index {index_name}: {e}")
        else:
            logger.info(f"Index {index_name} doesn't exist, skipping...")

    logger.info(f"Dropped {dropped_indexes} indexes")

    # Use batch operations for SQLite compatibility when dropping columns
    logger.info("Dropping ML feature columns...")
    with op.batch_alter_table("player_attributes", schema=None) as batch_op:
        # Drop columns (in reverse order of creation) with existence checks
        columns_to_drop = [
            # Advanced performance statistics
            "clearances_per_90",
            "interceptions_per_90",
            "tackles_per_90",
            "key_passes_per_90",
            "shots_per_90",
            # Player role indicators
            "role_confidence",
            "is_corner_taker",
            "is_free_kick_taker",
            "is_penalty_taker",
            # Fixture difficulty ratings
            "next_5_fixture_difficulty",
            "next_3_fixture_difficulty",
            # Form metrics
            "momentum",
            "form_10_games",
            "form_5_games",
            "form_3_games",
            # Expected Goals (xG) metrics
            "xgi_per_90",
            "xa_per_90",
            "xg_per_90",
        ]

        dropped_columns = 0
        for column_name in columns_to_drop:
            if column_name in existing_columns:
                try:
                    batch_op.drop_column(column_name)
                    logger.info(f"Dropped column: {column_name}")
                    dropped_columns += 1
                except Exception as e:
                    logger.warning(f"Failed to drop column {column_name}: {e}")
            else:
                logger.info(f"Column {column_name} doesn't exist, skipping...")

    logger.info(
        f"Rollback completed successfully! Dropped {dropped_columns} columns and {dropped_indexes} indexes."
    )
