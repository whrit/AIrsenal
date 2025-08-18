"""
Interface to the SQL database.
Use SQLAlchemy to convert between DB tables and python objects.
"""

from contextlib import contextmanager
from typing import Annotated

from sqlalchemy import ForeignKey, Index, String, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from airsenal.framework.env import (
    AIRSENAL_DB_FILE,
    AIRSENAL_DB_PASSWORD,
    AIRSENAL_DB_URI,
    AIRSENAL_DB_USER,
    AIRSENAL_HOME,
    save_env,
)

# Common type annotations using PEP 593 Annotated
intpk = Annotated[int, mapped_column(primary_key=True)]
str100 = Annotated[str, mapped_column(String(100))]
str4 = Annotated[str, mapped_column(String(4))]
str3 = Annotated[str, mapped_column(String(3))]
str100_optional = Annotated[str | None, mapped_column(String(100))]


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "player"
    player_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fpl_api_id: Mapped[int | None]
    name: Mapped[str100]
    attributes: Mapped[list["PlayerAttributes"]] = relationship(back_populates="player")
    absences: Mapped[list["Absence"]] = relationship(back_populates="player")
    results: Mapped[list["Result"]] = relationship(back_populates="player")
    predictions: Mapped[list["PlayerPrediction"]] = relationship(
        back_populates="player"
    )
    scores: Mapped[list["PlayerScore"]] = relationship(back_populates="player")

    def team(self, season: str, gameweek: int) -> str | None:
        """
        Get player's team for given season and gameweek.
        If data not available for specified gameweek but data is available for
        at least one gameweek in specified season, return a best guess value
        based on data nearest to specified gameweek.
        """
        attr = self.get_gameweek_attributes(season, gameweek)
        if attr is not None and not isinstance(attr, tuple):
            return attr.team
        print("No team found for", self.name, "in", season, "season.")
        return None

    def price(self, season: str, gameweek: int) -> int | None:
        """
        get player's price for given season and gameweek
        If data not available for specified gameweek but data is available for
        at least one gameweek in specified season, return a best guess value
        based on data nearest to specified gameweek.
        """
        attr = self.get_gameweek_attributes(season, gameweek, before_and_after=True)
        if attr is not None:
            return self._calculate_price(attr, gameweek)
        print("No price found for", self.name, "in", season, "season.")
        return None

    def _calculate_price(
        self,
        attr: "PlayerAttributes | tuple[PlayerAttributes, PlayerAttributes]",
        gameweek: int,
    ) -> int:
        """
        Either return price available for specified gameweek or interpolate based
        on nearest available price.
        """
        if not isinstance(attr, tuple):
            return attr.price
        # interpolate price between nearest available gameweeks
        gw_before = attr[0].gameweek
        price_before = attr[0].price
        gw_after = attr[1].gameweek
        price_after = attr[1].price

        gradient = (price_after - price_before) / (gw_after - gw_before)
        intercept = price_before - gradient * gw_before
        price = gradient * gameweek + intercept
        return round(price)

    def position(self, season: str) -> str | None:
        """
        get player's position for given season
        """
        attr = self.get_gameweek_attributes(season, None)
        if attr is not None and not isinstance(attr, tuple):
            return attr.position
        print("No position found for", self.name, "in", season, "season.")
        return None

    def is_injured_or_suspended(
        self, season: str, current_gw: int, fixture_gw: int
    ) -> bool:
        """Check whether a player is injured or suspended (<=50% chance of playing).
        current_gw - The current gameweek, i.e. the gameweek when we are querying the
        player's status.
        fixture_gw - The gameweek of the fixture we want to check whether the player
        is available for, i.e. we are checking whether the player is availiable in the
        future week "fixture_gw" at the previous point in time "current_gw".
        """
        attr = self.get_gameweek_attributes(season, current_gw)
        if attr is not None and not isinstance(attr, tuple):
            return (
                attr.chance_of_playing_next_round is not None
                and attr.chance_of_playing_next_round <= 50
            ) and (attr.return_gameweek is None or attr.return_gameweek > fixture_gw)
        return False

    def get_gameweek_attributes(
        self, season: str, gameweek: int | None, before_and_after: bool = False
    ) -> "PlayerAttributes | tuple[PlayerAttributes, PlayerAttributes] | None":
        """Get the PlayerAttributes object for this player in the given gameweek and
        season, or the nearest available gameweek(s) if the exact gameweek is not
        available.
        If no attributes available in the specified season, return None in all cases.
        If before_and_after is True and an exact gameweek & season match is not found,
        return both the nearest gameweek before and after the specified gameweek.
        """
        gw_before = 0
        gw_after = 100
        attr_before = None
        attr_after = None

        for attr in self.attributes:
            if attr.season != season:
                continue

            if gameweek is None:
                # trying to match season only
                return attr
            if attr.gameweek == gameweek:
                return attr
            if (attr.gameweek < gameweek) and (attr.gameweek > gw_before):
                # update last available attr before specified gameweek
                gw_before = attr.gameweek
                attr_before = attr
            elif (attr.gameweek > gameweek) and (attr.gameweek < gw_after):
                # update next available attr after specified gameweek
                gw_after = attr.gameweek
                attr_after = attr

        # ran through all attributes without finding exact gameweek and season match
        if attr_before is None and attr_after is None:
            # no attributes for this player in this season
            return None
        if not attr_after:
            return attr_before
        if not attr_before:
            return attr_after
        if before_and_after:
            return (attr_before, attr_after)
        # return attributes at gameweeek nearest to input gameweek
        if gameweek is not None and (gw_after - gameweek) >= (gameweek - gw_before):
            return attr_before
        return attr_after

    def __str__(self):
        return self.name


class PlayerMapping(Base):
    # alternative names for players
    __tablename__ = "player_mapping"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    alt_name: Mapped[str100]


class PlayerAttributes(Base):
    __tablename__ = "player_attributes"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player: Mapped["Player"] = relationship(back_populates="attributes")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))
    season: Mapped[str100]
    gameweek: Mapped[int]
    price: Mapped[int]
    team: Mapped[str100]
    position: Mapped[str100]

    chance_of_playing_next_round: Mapped[int | None]
    news: Mapped[str100_optional]
    return_gameweek: Mapped[int | None]
    transfers_balance: Mapped[int | None]
    selected: Mapped[int | None]
    transfers_in: Mapped[int | None]
    transfers_out: Mapped[int | None]

    # Expected Goals (xG) metrics
    xg_per_90: Mapped[float | None] = mapped_column(
        comment="Expected goals per 90 minutes"
    )
    xa_per_90: Mapped[float | None] = mapped_column(
        comment="Expected assists per 90 minutes"
    )
    xgi_per_90: Mapped[float | None] = mapped_column(
        comment="Expected goal involvements (xG + xA) per 90 minutes"
    )

    # Form metrics - rolling averages of FPL points
    form_3_games: Mapped[float | None] = mapped_column(
        comment="Average FPL points over last 3 games"
    )
    form_5_games: Mapped[float | None] = mapped_column(
        comment="Average FPL points over last 5 games"
    )
    form_10_games: Mapped[float | None] = mapped_column(
        comment="Average FPL points over last 10 games"
    )
    momentum: Mapped[float | None] = mapped_column(
        comment="Trend indicator for recent performance (-1 to 1)"
    )

    # Fixture difficulty ratings for upcoming fixtures
    next_3_fixture_difficulty: Mapped[float | None] = mapped_column(
        comment="Average difficulty rating for next 3 fixtures (1-5 scale)"
    )
    next_5_fixture_difficulty: Mapped[float | None] = mapped_column(
        comment="Average difficulty rating for next 5 fixtures (1-5 scale)"
    )

    # Player role indicators
    is_penalty_taker: Mapped[bool] = mapped_column(
        default=False, comment="Whether player is primary penalty taker"
    )
    is_free_kick_taker: Mapped[bool] = mapped_column(
        default=False, comment="Whether player is primary free kick taker"
    )
    is_corner_taker: Mapped[bool] = mapped_column(
        default=False, comment="Whether player is primary corner taker"
    )
    role_confidence: Mapped[float | None] = mapped_column(
        comment="Confidence score for set piece roles (0-1 scale)"
    )

    # Advanced performance statistics per 90 minutes
    shots_per_90: Mapped[float | None] = mapped_column(
        comment="Total shots per 90 minutes"
    )
    key_passes_per_90: Mapped[float | None] = mapped_column(
        comment="Key passes (passes leading to shots) per 90 minutes"
    )
    tackles_per_90: Mapped[float | None] = mapped_column(
        comment="Successful tackles per 90 minutes"
    )
    interceptions_per_90: Mapped[float | None] = mapped_column(
        comment="Interceptions per 90 minutes"
    )
    clearances_per_90: Mapped[float | None] = mapped_column(
        comment="Clearances per 90 minutes"
    )

    # Define indexes for frequently queried fields to optimize performance
    __table_args__ = (
        # Composite index for player lookups across seasons/gameweeks
        Index("ix_player_season_gameweek", "player_id", "season", "gameweek"),
        # Indexes for form metrics - frequently used for player selection
        Index("ix_form_3_games", "form_3_games"),
        Index("ix_form_5_games", "form_5_games"),
        # Indexes for xG metrics - key performance indicators
        Index("ix_xg_per_90", "xg_per_90"),
        Index("ix_xgi_per_90", "xgi_per_90"),
        # Index for fixture difficulty - used in transfer optimization
        Index("ix_next_3_fixture_difficulty", "next_3_fixture_difficulty"),
        # Composite index for role-based queries (position + roles)
        Index("ix_position_penalty_taker", "position", "is_penalty_taker"),
        # Index for momentum-based filtering
        Index("ix_momentum", "momentum"),
    )

    def __str__(self):
        return (
            f"{self.player} ({self.season} GW{self.gameweek}): "
            f"£{self.price / 10}, {self.team}, {self.position}"
        )


class Absence(Base):
    __tablename__ = "absence"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player: Mapped["Player"] = relationship(back_populates="absences")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))
    season: Mapped[str100]
    reason: Mapped[str100]  # high-level, e.g. injury/suspension
    details: Mapped[str100_optional]
    date_from: Mapped[str100]
    date_until: Mapped[str100_optional]
    gw_from: Mapped[int]
    gw_until: Mapped[int | None]
    url: Mapped[str100_optional]
    timestamp: Mapped[str100]

    def __str__(self):
        return (
            f"Absence(\n"
            f"  player='{self.player}',\n"
            f"  player_id='{self.player_id}',\n"
            f"  season='{self.season}',\n"
            f"  reason='{self.reason}',\n"
            f"  details='{self.details}',\n"
            f"  date_from='{self.date_from}',\n"
            f"  date_until='{self.date_until}',\n"
            f"  gw_from='{self.gw_from}',\n"
            f"  gw_until='{self.gw_until}',\n"
            f"  url='{self.url}',\n"
            f"  timestamp='{self.timestamp}'\n"
            ")"
        )


class Result(Base):
    __tablename__ = "result"
    result_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship(back_populates="result")
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))
    home_score: Mapped[int]
    away_score: Mapped[int]
    player: Mapped["Player"] = relationship(back_populates="results")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))

    def __str__(self):
        return (
            f"{self.fixture.season} GW{self.fixture.gameweek} "
            f"{self.fixture.home_team} {self.home_score} - "
            f"{self.away_score} {self.fixture.away_team}"
        )


class Fixture(Base):
    __tablename__ = "fixture"
    fixture_id: Mapped[intpk] = mapped_column(autoincrement=True)
    date: Mapped[str | None] = mapped_column(
        String(100)
    )  # In case fixture not yet scheduled!
    gameweek: Mapped[int | None]  # In case fixture not yet scheduled!
    home_team: Mapped[str100]
    away_team: Mapped[str100]
    season: Mapped[str100]
    tag: Mapped[str100]
    result: Mapped["Result | None"] = relationship(back_populates="fixture")

    def __str__(self):
        return f"{self.season} GW{self.gameweek} {self.home_team} vs. {self.away_team}"


class PlayerScore(Base):
    __tablename__ = "player_score"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_team: Mapped[str100]
    opponent: Mapped[str100]
    points: Mapped[int]
    goals: Mapped[int]
    assists: Mapped[int]
    bonus: Mapped[int]
    conceded: Mapped[int]
    minutes: Mapped[int]
    player: Mapped["Player"] = relationship(back_populates="scores")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))
    result: Mapped["Result"] = relationship()
    result_id: Mapped[int | None] = mapped_column(ForeignKey("result.result_id"))
    fixture: Mapped["Fixture"] = relationship()
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))

    # extended features
    clean_sheets: Mapped[int | None]
    own_goals: Mapped[int | None]
    penalties_saved: Mapped[int | None]
    penalties_missed: Mapped[int | None]
    penalties_taken: Mapped[int | None] = mapped_column(
        comment="Number of penalties taken by player"
    )
    penalties_scored: Mapped[int | None] = mapped_column(
        comment="Number of penalties scored by player"
    )
    yellow_cards: Mapped[int | None]
    red_cards: Mapped[int | None]
    saves: Mapped[int | None]
    bps: Mapped[int | None]
    influence: Mapped[float | None]
    creativity: Mapped[float | None]
    threat: Mapped[float | None]
    ict_index: Mapped[float | None]
    expected_goals: Mapped[float | None]
    expected_assists: Mapped[float | None]
    expected_goal_involvements: Mapped[float | None]
    expected_goals_conceded: Mapped[float | None]

    def __str__(self):
        return f"{self.player} ({self.result}): {self.points} pts, {self.minutes} mins"


class PlayerPrediction(Base):
    __tablename__ = "player_prediction"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship()
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))
    predicted_points: Mapped[float]
    tag: Mapped[str100]
    player: Mapped["Player"] = relationship(back_populates="predictions")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))

    def __str__(self):
        return f"{self.player}: Predict {self.predicted_points} pts in {self.fixture}"


class Transaction(Base):
    __tablename__ = "transaction"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int]
    gameweek: Mapped[int]
    bought_or_sold: Mapped[int]  # +1 for bought, -1 for sold
    season: Mapped[str100]
    time: Mapped[str100]
    tag: Mapped[str100]
    price: Mapped[int]
    free_hit: Mapped[int]  # 1 if transfer on Free Hit, 0 otherwise
    fpl_team_id: Mapped[int]

    def __str__(self):
        trans_str = f"{self.season} GW{self.gameweek}: Team {self.fpl_team_id} "
        if self.bought_or_sold == 1:
            trans_str += f"bought player {self.player_id}"
        else:
            trans_str += f"sold player {self.player_id}"
        if self.free_hit:
            trans_str += " (FREE HIT)"
        return trans_str


class TransferSuggestion(Base):
    __tablename__ = "transfer_suggestion"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int]
    in_or_out: Mapped[int]  # +1 for buy, -1 for sell
    gameweek: Mapped[int]
    points_gain: Mapped[float]
    timestamp: Mapped[str100]  # use this to group suggestions
    season: Mapped[str100]
    fpl_team_id: Mapped[int]  # to identify team to apply transfers.
    chip_played: Mapped[str100_optional]

    def __str__(self):
        sugg_str = f"{self.season} GW{self.gameweek}: Suggest "
        if self.in_or_out == 1:
            sugg_str += f"buying {self.player_id} to gain {self.points_gain:.2f} pts"
        else:
            sugg_str += f"selling {self.player_id} to gain {self.points_gain:.2f} pts"
        return sugg_str


class FifaTeamRating(Base):
    __tablename__ = "fifa_rating"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    season: Mapped[str4]
    team: Mapped[str100]
    att: Mapped[int]
    defn: Mapped[int]
    mid: Mapped[int]
    ovr: Mapped[int]

    def __str__(self):
        return (
            f"{self.team} {self.season} FIFA rating: "
            f"ovr {self.ovr}, def {self.defn}, mid {self.mid}, att {self.att}"
        )


class Team(Base):
    __tablename__ = "team"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    name: Mapped[str3]
    full_name: Mapped[str100]
    season: Mapped[str4]
    team_id: Mapped[int]  # the season-dependent team ID (from alphabetical order)

    def __str__(self):
        return f"{self.full_name} ({self.name})"


class SessionSquad(Base):
    __tablename__ = "sessionteam"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    session_id: Mapped[str100]
    player_id: Mapped[int]


class SessionBudget(Base):
    __tablename__ = "sessionbudget"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    session_id: Mapped[str100]
    budget: Mapped[int]


class ModelRegistry(Base):
    """Registry for model types and configurations"""

    __tablename__ = "model_registry"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    model_name: Mapped[str100]  # e.g., "player_model", "team_model"
    model_type: Mapped[
        str100
    ]  # e.g., "NumpyroPlayerModel", "ExtendedDixonColesMatchPredictor"
    description: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[str100]  # ISO datetime string
    created_by: Mapped[str100]  # Author/creator
    is_active: Mapped[bool] = mapped_column(default=True)

    # Relationships
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model_registry"
    )

    def __str__(self):
        return f"ModelRegistry({self.model_name}: {self.model_type})"


class ModelVersion(Base):
    """Specific versions of models with metadata and performance tracking"""

    __tablename__ = "model_version"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    registry_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"))
    version: Mapped[str100]  # Semantic version (e.g., "1.0.0", "1.1.0-beta")
    config_hash: Mapped[str100]  # Hash of training configuration
    training_data_version: Mapped[str100]  # Version/hash of training data used

    # Training metadata
    training_params: Mapped[str | None] = mapped_column(
        String(1000)
    )  # JSON string of parameters
    feature_set: Mapped[str | None] = mapped_column(
        String(500)
    )  # Description of features used
    training_date: Mapped[str100]  # ISO datetime string
    training_duration_seconds: Mapped[int | None]

    # Performance metrics
    validation_mae: Mapped[float | None]
    validation_rmse: Mapped[float | None]
    validation_accuracy: Mapped[float | None]
    cross_validation_score: Mapped[float | None]

    # Status and deployment
    status: Mapped[str100]  # "training", "ready", "deployed", "deprecated", "failed"
    is_production: Mapped[bool] = mapped_column(default=False)
    deployment_date: Mapped[str100 | None]

    # Metadata
    notes: Mapped[str | None] = mapped_column(String(1000))
    tags: Mapped[str | None] = mapped_column(String(500))  # Comma-separated tags

    # Relationships
    model_registry: Mapped["ModelRegistry"] = relationship(back_populates="versions")
    artifacts: Mapped[list["ModelArtifact"]] = relationship(back_populates="version")
    performance_records: Mapped[list["ModelPerformance"]] = relationship(
        back_populates="version"
    )
    experiments: Mapped[list["ModelExperiment"]] = relationship(
        back_populates="model_version", foreign_keys="ModelExperiment.model_version_id"
    )
    checkpoints: Mapped[list["ModelCheckpoint"]] = relationship(back_populates="model_version")
    metadata: Mapped[list["ModelMetadata"]] = relationship(
        back_populates="model_version", foreign_keys="ModelMetadata.version_id"
    )

    def __str__(self):
        return f"ModelVersion({self.model_registry.model_name} v{self.version})"


class ModelArtifact(Base):
    """Storage metadata for model artifacts (serialized models, weights, etc.)"""

    __tablename__ = "model_artifact"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    artifact_type: Mapped[
        str100
    ]  # "model_weights", "full_model", "metadata", "training_state"
    file_path: Mapped[str | None] = mapped_column(String(500))  # Local file path
    s3_path: Mapped[str | None] = mapped_column(
        String(500)
    )  # S3 URL if using cloud storage
    file_size_bytes: Mapped[int | None]
    checksum: Mapped[str100 | None]  # MD5 or SHA256 hash for integrity
    compression: Mapped[str100 | None]  # "gzip", "bzip2", "none"
    serialization_format: Mapped[str100]  # "pickle", "joblib", "jax", "numpy"
    created_at: Mapped[str100]  # ISO datetime string

    # Relationships
    version: Mapped["ModelVersion"] = relationship(back_populates="artifacts")

    def __str__(self):
        return f"ModelArtifact({self.artifact_type} for {self.version})"


class ModelPerformance(Base):
    """Performance metrics for models on specific datasets/time periods"""

    __tablename__ = "model_performance"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    evaluation_date: Mapped[str100]  # ISO datetime string
    dataset_type: Mapped[str100]  # "validation", "test", "production", "backtest"
    time_period_start: Mapped[str100 | None]  # ISO datetime string
    time_period_end: Mapped[str100 | None]  # ISO datetime string

    # Core metrics
    mae: Mapped[float | None]  # Mean Absolute Error
    rmse: Mapped[float | None]  # Root Mean Square Error
    accuracy: Mapped[float | None]  # Classification accuracy if applicable
    precision: Mapped[float | None]
    recall: Mapped[float | None]
    f1_score: Mapped[float | None]

    # Domain-specific metrics
    prediction_correlation: Mapped[float | None]  # Correlation with actual points
    top_transfer_accuracy: Mapped[float | None]  # Accuracy of top transfer suggestions
    points_captured: Mapped[float | None]  # Percentage of possible points captured

    # Additional metrics as JSON
    additional_metrics: Mapped[str | None] = mapped_column(String(1000))  # JSON string

    # Relationships
    version: Mapped["ModelVersion"] = relationship(back_populates="performance_records")

    def __str__(self):
        return (
            f"ModelPerformance({self.version} on {self.dataset_type}: MAE={self.mae})"
        )


class ModelExperiment(Base):
    """A/B testing experiments comparing model versions"""

    __tablename__ = "model_experiment"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    experiment_name: Mapped[str100]
    description: Mapped[str | None] = mapped_column(String(500))

    # Experiment configuration
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    control_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_version.id")
    )
    traffic_split: Mapped[float] = mapped_column(
        default=0.5
    )  # Percentage of traffic for this version

    # Experiment timeline
    start_date: Mapped[str100]  # ISO datetime string
    end_date: Mapped[str100 | None]  # ISO datetime string
    status: Mapped[str100]  # "planned", "running", "completed", "stopped"

    # Results
    winner_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_version.id")
    )
    confidence_level: Mapped[float | None]  # Statistical confidence in results

    # Metadata
    created_by: Mapped[str100]
    notes: Mapped[str | None] = mapped_column(String(1000))

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="experiments", foreign_keys=[model_version_id]
    )

    def __str__(self):
        return f"ModelExperiment({self.experiment_name}: {self.status})"


class ModelCheckpoint(Base):
    """Storage for complete model checkpoints with persistence layer integration"""

    __tablename__ = "model_checkpoint"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    checkpoint_name: Mapped[str100]  # "auto_checkpoint_001", "pre_training", "best_validation"
    checkpoint_type: Mapped[str100]  # "automatic", "manual", "milestone", "recovery"
    
    # Model state storage
    model_data: Mapped[bytes | None] = mapped_column(LargeBinary)  # Compressed model state
    compression_format: Mapped[str100] = mapped_column(default="gzip")  # "gzip", "bz2", "lzma", "none"
    original_size_bytes: Mapped[int | None]
    compressed_size_bytes: Mapped[int | None]
    
    # Integrity verification
    checksum: Mapped[str100]  # SHA256 hash for integrity verification
    checksum_algorithm: Mapped[str100] = mapped_column(default="sha256")
    
    # Storage metadata
    file_path: Mapped[str | None] = mapped_column(String(500))  # Local file path
    cloud_path: Mapped[str | None] = mapped_column(String(500))  # S3/GCS/Azure path
    storage_backend: Mapped[str100] = mapped_column(default="database")  # "database", "file", "s3", "gcs", "azure"
    
    # Checkpoint metadata
    created_at: Mapped[str100]  # ISO datetime string
    created_by: Mapped[str100 | None]  # User or system that created checkpoint
    gameweek: Mapped[int | None]  # Gameweek when checkpoint was created
    season: Mapped[str100 | None]  # Season when checkpoint was created
    tags: Mapped[str | None] = mapped_column(String(500))  # Comma-separated tags
    description: Mapped[str | None] = mapped_column(String(1000))
    
    # Performance metrics at checkpoint time
    validation_score: Mapped[float | None]
    training_loss: Mapped[float | None]
    model_size_mb: Mapped[float | None]
    
    # Status and lifecycle
    status: Mapped[str100] = mapped_column(default="active")  # "active", "archived", "corrupted", "deleted"
    is_recoverable: Mapped[bool] = mapped_column(default=True)
    recovery_priority: Mapped[int] = mapped_column(default=1)  # 1-10 scale for recovery importance
    
    # Relationships
    model_version: Mapped["ModelVersion"] = relationship()
    model_states: Mapped[list["ModelState"]] = relationship(back_populates="checkpoint")

    def __str__(self):
        return f"ModelCheckpoint({self.checkpoint_name} for {self.model_version})"


class ModelState(Base):
    """Individual player model states for granular persistence and recovery"""

    __tablename__ = "model_state"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    checkpoint_id: Mapped[int] = mapped_column(ForeignKey("model_checkpoint.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"))
    
    # State vector and covariance data
    state_mean: Mapped[bytes | None] = mapped_column(LargeBinary)  # Serialized state mean vector
    state_covariance: Mapped[bytes | None] = mapped_column(LargeBinary)  # Serialized covariance matrix
    state_history: Mapped[bytes | None] = mapped_column(LargeBinary)  # Compressed state history
    
    # State metadata
    state_dimension: Mapped[int]  # Dimensionality of state vector
    gameweek: Mapped[int]
    season: Mapped[str100]
    last_updated: Mapped[str100]  # ISO datetime string
    
    # Serialization metadata
    serialization_format: Mapped[str100] = mapped_column(default="numpy")  # "numpy", "pickle", "json"
    compression_used: Mapped[bool] = mapped_column(default=True)
    
    # State validation
    state_checksum: Mapped[str100]  # Checksum for individual state integrity
    is_valid: Mapped[bool] = mapped_column(default=True)
    validation_errors: Mapped[str | None] = mapped_column(String(1000))
    
    # Performance tracking
    prediction_accuracy: Mapped[float | None]  # Recent prediction accuracy for this player
    uncertainty_score: Mapped[float | None]  # Measure of state uncertainty
    update_frequency: Mapped[int] = mapped_column(default=0)  # Number of times state has been updated
    
    # Relationships
    checkpoint: Mapped["ModelCheckpoint"] = relationship(back_populates="model_states")
    player: Mapped["Player"] = relationship()

    def __str__(self):
        return f"ModelState(Player {self.player_id} @ GW{self.gameweek})"


class ModelMetadata(Base):
    """Extended metadata for model configurations and performance tracking"""

    __tablename__ = "model_metadata"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    
    # Model configuration
    state_space_config: Mapped[str | None] = mapped_column(String(2000))  # JSON config
    hyperparameters: Mapped[str | None] = mapped_column(String(2000))  # JSON hyperparameters
    feature_config: Mapped[str | None] = mapped_column(String(2000))  # JSON feature configuration
    
    # Training configuration
    training_algorithm: Mapped[str100 | None]  # "kalman", "particle_filter", "gradient_descent"
    optimizer_config: Mapped[str | None] = mapped_column(String(1000))  # JSON optimizer settings
    regularization_config: Mapped[str | None] = mapped_column(String(1000))  # JSON regularization
    
    # Data configuration
    training_data_sources: Mapped[str | None] = mapped_column(String(1000))  # JSON data sources
    feature_selection_method: Mapped[str100 | None]
    data_preprocessing_steps: Mapped[str | None] = mapped_column(String(1000))  # JSON preprocessing
    
    # Performance metrics
    convergence_metrics: Mapped[str | None] = mapped_column(String(1000))  # JSON convergence data
    computational_metrics: Mapped[str | None] = mapped_column(String(1000))  # JSON compute stats
    memory_usage_metrics: Mapped[str | None] = mapped_column(String(1000))  # JSON memory stats
    
    # Experiment tracking
    experiment_id: Mapped[str100 | None]  # MLflow/Weights&Biases experiment ID
    run_id: Mapped[str100 | None]  # Specific run ID within experiment
    parent_run_id: Mapped[str100 | None]  # Parent run for nested experiments
    
    # Model lineage
    derived_from_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"))
    inheritance_type: Mapped[str100 | None]  # "fine_tuned", "transferred", "ensemble_member"
    modification_summary: Mapped[str | None] = mapped_column(String(1000))
    
    # Quality metrics
    data_quality_score: Mapped[float | None]  # 0-1 score for training data quality
    model_complexity_score: Mapped[float | None]  # Complexity measure
    interpretability_score: Mapped[float | None]  # How interpretable the model is
    
    # Deployment metadata
    deployment_requirements: Mapped[str | None] = mapped_column(String(1000))  # JSON requirements
    compatibility_info: Mapped[str | None] = mapped_column(String(1000))  # JSON compatibility
    
    # Timestamps
    created_at: Mapped[str100]
    updated_at: Mapped[str100]
    
    # Relationships
    model_version: Mapped["ModelVersion"] = relationship()
    derived_from: Mapped["ModelVersion"] = relationship(foreign_keys=[derived_from_version_id])

    def __str__(self):
        return f"ModelMetadata(Version {self.version_id})"


class FeatureDefinition(Base):
    """Registry of available features with their computation logic and metadata."""

    __tablename__ = "feature_definition"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    name: Mapped[str100]  # e.g., "rolling_goals_5", "xg_form", "fixture_difficulty"
    version: Mapped[str100]  # semantic versioning for computation logic
    feature_type: Mapped[str100]  # "player", "team", "fixture", "external"
    data_type: Mapped[str100]  # "float", "int", "boolean", "string"
    description: Mapped[str100_optional]
    computation_logic: Mapped[str | None] = mapped_column(String(1000))  # JSON config
    dependencies: Mapped[str100_optional]  # comma-separated feature names
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[str100]
    updated_at: Mapped[str100]
    # Relationship to computed values
    computed_features: Mapped[list["ComputedFeature"]] = relationship(
        back_populates="feature_definition"
    )

    def __str__(self):
        return f"{self.name}:{self.version} ({self.feature_type})"


class ComputedFeature(Base):
    """Computed feature values for entities (players, teams, fixtures) at specific times."""

    __tablename__ = "computed_feature"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    feature_definition_id: Mapped[int] = mapped_column(
        ForeignKey("feature_definition.id")
    )
    feature_definition: Mapped["FeatureDefinition"] = relationship(
        back_populates="computed_features"
    )
    entity_type: Mapped[str100]  # "player", "team", "fixture"
    entity_id: Mapped[int]  # player_id, team_id, fixture_id
    gameweek: Mapped[int | None]  # for time-aware features
    season: Mapped[str100]
    value: Mapped[float | None]  # stored as float, cast as needed
    string_value: Mapped[str100_optional]  # for non-numeric features
    computed_at: Mapped[str100]  # ISO timestamp when computed
    ttl_expires_at: Mapped[str100_optional]  # cache expiration

    def __str__(self):
        return f"{self.feature_definition.name} for {self.entity_type}:{self.entity_id} = {self.value}"


class FeatureCache(Base):
    """Hot cache for frequently accessed features to enable sub-second serving."""

    __tablename__ = "feature_cache"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    cache_key: Mapped[
        str100
    ]  # composite key: feature_name:entity_type:entity_id:context
    feature_name: Mapped[str100]
    entity_type: Mapped[str100]
    entity_id: Mapped[int]
    value: Mapped[float | None]
    string_value: Mapped[str100_optional]
    cached_at: Mapped[str100]
    expires_at: Mapped[str100]
    hit_count: Mapped[int] = mapped_column(default=0)  # for cache analytics

    def __str__(self):
        return f"Cache[{self.cache_key}] = {self.value or self.string_value}"


class FeatureTimeSeries(Base):
    """Time-series storage for rolling window computations and historical analysis."""

    __tablename__ = "feature_timeseries"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    entity_type: Mapped[str100]
    entity_id: Mapped[int]
    metric_name: Mapped[str100]  # "goals", "assists", "minutes", "xg", etc.
    season: Mapped[str100]
    gameweek: Mapped[int]
    timestamp: Mapped[str100]  # ISO timestamp for fine-grained ordering
    value: Mapped[float]
    context: Mapped[str100_optional]  # additional metadata as JSON

    def __str__(self):
        return f"{self.metric_name} for {self.entity_type}:{self.entity_id} @ GW{self.gameweek} = {self.value}"


class DatabaseVersion(Base):
    """Database schema version tracking for AIrsenal compatibility management"""

    __tablename__ = "database_version"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    version: Mapped[str100]  # Semantic version (e.g., "1.11.0", "1.12.0-beta")
    schema_version: Mapped[
        str100
    ]  # Schema-specific version for tracking DB structure changes
    app_version: Mapped[str100]  # Application version that created/updated this schema
    migration_id: Mapped[str100 | None]  # Alembic migration ID if applicable

    # Migration metadata
    applied_at: Mapped[str100]  # ISO datetime string when version was applied
    applied_by: Mapped[str100]  # User or system that applied the version
    migration_type: Mapped[str100]  # "initial", "upgrade", "rollback", "manual"
    migration_source: Mapped[str100]  # "alembic", "manual", "automated"

    # Migration details
    migration_description: Mapped[str | None] = mapped_column(String(500))
    migration_checksum: Mapped[str100 | None]  # MD5 hash of migration script content
    rollback_info: Mapped[str | None] = mapped_column(
        String(1000)
    )  # JSON string with rollback details

    # Compatibility information
    min_app_version: Mapped[
        str100 | None
    ]  # Minimum app version compatible with this schema
    max_app_version: Mapped[
        str100 | None
    ]  # Maximum app version compatible with this schema
    compatibility_notes: Mapped[str | None] = mapped_column(String(500))

    # Status and validation
    is_active: Mapped[bool] = mapped_column(
        default=True
    )  # Whether this version is currently active
    validation_status: Mapped[str100] = mapped_column(
        default="pending"
    )  # "pending", "validated", "failed"
    validation_errors: Mapped[str | None] = mapped_column(
        String(1000)
    )  # JSON string of validation errors

    # Performance impact tracking
    migration_duration_seconds: Mapped[float | None]  # How long the migration took
    affected_tables: Mapped[str | None] = mapped_column(
        String(500)
    )  # Comma-separated list of affected table names
    records_migrated: Mapped[int | None]  # Number of records affected by migration

    # Relationships with migration history
    migration_history: Mapped[list["MigrationHistory"]] = relationship(
        back_populates="database_version"
    )

    def __str__(self):
        return f"DatabaseVersion(v{self.version}, schema:{self.schema_version}, applied:{self.applied_at})"


class MigrationHistory(Base):
    """Detailed history of all database migrations for audit and rollback purposes"""

    __tablename__ = "migration_history"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    database_version_id: Mapped[int] = mapped_column(ForeignKey("database_version.id"))
    database_version: Mapped["DatabaseVersion"] = relationship(
        back_populates="migration_history"
    )

    # Migration identification
    migration_name: Mapped[str100]  # Human-readable migration name
    migration_hash: Mapped[str100]  # Unique hash identifying this specific migration
    sequence_number: Mapped[int]  # Order of migration execution within a version

    # Execution details
    executed_at: Mapped[str100]  # ISO datetime string
    execution_duration_seconds: Mapped[float]
    executed_by: Mapped[str100]  # User or automated system
    execution_context: Mapped[
        str100
    ]  # "deployment", "development", "testing", "rollback"

    # Migration content and impact
    migration_sql: Mapped[str | None] = mapped_column(
        String(10000)
    )  # SQL executed (if applicable)
    tables_affected: Mapped[str | None] = mapped_column(
        String(500)
    )  # Comma-separated table names
    records_before: Mapped[int | None]  # Total records before migration
    records_after: Mapped[int | None]  # Total records after migration
    records_changed: Mapped[int | None]  # Number of records modified

    # Success and error tracking
    status: Mapped[str100]  # "success", "failed", "partial", "rolled_back"
    error_message: Mapped[str | None] = mapped_column(String(1000))
    warning_count: Mapped[int] = mapped_column(default=0)

    # Rollback information
    rollback_sql: Mapped[str | None] = mapped_column(
        String(10000)
    )  # SQL to rollback this migration
    rollback_tested: Mapped[bool] = mapped_column(
        default=False
    )  # Whether rollback has been tested
    rollback_notes: Mapped[str | None] = mapped_column(String(500))

    # Performance and monitoring
    memory_usage_mb: Mapped[float | None]  # Peak memory usage during migration
    cpu_usage_percent: Mapped[float | None]  # Average CPU usage during migration
    lock_conflicts: Mapped[int] = mapped_column(
        default=0
    )  # Number of lock conflicts encountered

    # Dependencies and prerequisites
    depends_on: Mapped[str | None] = mapped_column(
        String(500)
    )  # Comma-separated list of prerequisite migrations
    blocks: Mapped[str | None] = mapped_column(
        String(500)
    )  # Comma-separated list of migrations blocked by this one

    def __str__(self):
        return f"MigrationHistory({self.migration_name}, status:{self.status}, executed:{self.executed_at})"


class SchemaCompatibility(Base):
    """Matrix defining compatibility between application versions and database schema versions"""

    __tablename__ = "schema_compatibility"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    app_version_min: Mapped[str100]  # Minimum application version
    app_version_max: Mapped[str100]  # Maximum application version
    schema_version_min: Mapped[str100]  # Minimum compatible schema version
    schema_version_max: Mapped[str100]  # Maximum compatible schema version

    # Compatibility metadata
    compatibility_level: Mapped[
        str100
    ]  # "full", "limited", "deprecated", "incompatible"
    compatibility_notes: Mapped[str | None] = mapped_column(String(500))
    migration_required: Mapped[bool] = mapped_column(default=False)
    migration_priority: Mapped[str100] = mapped_column(
        default="normal"
    )  # "critical", "high", "normal", "low"

    # Validation and testing
    tested_combinations: Mapped[str | None] = mapped_column(
        String(1000)
    )  # JSON list of tested version combinations
    known_issues: Mapped[str | None] = mapped_column(
        String(1000)
    )  # JSON list of known compatibility issues
    workarounds: Mapped[str | None] = mapped_column(
        String(1000)
    )  # JSON list of available workarounds

    # Lifecycle management
    created_at: Mapped[str100]
    updated_at: Mapped[str100]
    deprecated_at: Mapped[str100 | None]
    removed_at: Mapped[str100 | None]

    # Performance impact
    performance_impact: Mapped[str100] = mapped_column(
        default="none"
    )  # "none", "low", "medium", "high"
    performance_notes: Mapped[str | None] = mapped_column(String(500))

    def __str__(self):
        return f"SchemaCompatibility(app:{self.app_version_min}-{self.app_version_max}, schema:{self.schema_version_min}-{self.schema_version_max}, level:{self.compatibility_level})"


class TeamStrength(Base):
    """Store current team strength ratings using Bayesian modeling"""

    __tablename__ = "team_strength"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Core strength ratings with uncertainty
    attacking_strength_home: Mapped[float] = mapped_column(
        comment="Home attacking strength (posterior mean)"
    )
    attacking_strength_away: Mapped[float] = mapped_column(
        comment="Away attacking strength (posterior mean)"
    )
    defensive_strength_home: Mapped[float] = mapped_column(
        comment="Home defensive strength (posterior mean)"
    )
    defensive_strength_away: Mapped[float] = mapped_column(
        comment="Away defensive strength (posterior mean)"
    )

    # Uncertainty quantification (standard deviations of posterior)
    attacking_strength_home_std: Mapped[float] = mapped_column(
        comment="Standard deviation of home attacking strength"
    )
    attacking_strength_away_std: Mapped[float] = mapped_column(
        comment="Standard deviation of away attacking strength"
    )
    defensive_strength_home_std: Mapped[float] = mapped_column(
        comment="Standard deviation of home defensive strength"
    )
    defensive_strength_away_std: Mapped[float] = mapped_column(
        comment="Standard deviation of away defensive strength"
    )

    # Overall strength composites
    overall_strength_home: Mapped[float] = mapped_column(
        comment="Combined home strength rating"
    )
    overall_strength_away: Mapped[float] = mapped_column(
        comment="Combined away strength rating"
    )

    # Component metrics contributing to strength
    expected_goals_for_per_game: Mapped[float] = mapped_column(
        comment="Expected goals scored per game"
    )
    expected_goals_against_per_game: Mapped[float] = mapped_column(
        comment="Expected goals conceded per game"
    )
    actual_goals_for_per_game: Mapped[float] = mapped_column(
        comment="Actual goals scored per game"
    )
    actual_goals_against_per_game: Mapped[float] = mapped_column(
        comment="Actual goals conceded per game"
    )
    shots_for_per_game: Mapped[float] = mapped_column(comment="Shots taken per game")
    shots_against_per_game: Mapped[float] = mapped_column(
        comment="Shots conceded per game"
    )
    clean_sheet_probability: Mapped[float] = mapped_column(
        comment="Probability of keeping clean sheet"
    )

    # Form indicators
    form_weighted_strength: Mapped[float] = mapped_column(
        comment="Form-adjusted strength rating"
    )
    recent_performance_trend: Mapped[float] = mapped_column(
        comment="Recent performance trend (-1 to 1)"
    )
    momentum_factor: Mapped[float] = mapped_column(comment="Team momentum factor")

    # Bayesian model metadata
    model_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of Bayesian model used"
    )
    sample_count: Mapped[int] = mapped_column(comment="Number of MCMC samples used")
    convergence_diagnostic: Mapped[float] = mapped_column(
        comment="R-hat convergence diagnostic"
    )
    effective_sample_size: Mapped[int] = mapped_column(comment="Effective sample size")

    # Data quality and context
    matches_played: Mapped[int] = mapped_column(
        comment="Number of matches in calculation"
    )
    home_matches_played: Mapped[int] = mapped_column(comment="Home matches played")
    away_matches_played: Mapped[int] = mapped_column(comment="Away matches played")
    data_quality_score: Mapped[float] = mapped_column(
        comment="Quality of underlying data (0-1)"
    )

    # External factors
    manager_change_adjustment: Mapped[float | None] = mapped_column(
        comment="Adjustment for recent manager changes"
    )
    key_player_injury_impact: Mapped[float | None] = mapped_column(
        comment="Impact of key player injuries"
    )
    transfer_window_impact: Mapped[float | None] = mapped_column(
        comment="Impact of recent transfers"
    )

    # Timestamp
    calculated_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when strength was calculated"
    )
    expires_at: Mapped[str100 | None] = mapped_column(
        comment="When this strength rating expires"
    )

    # Define indexes for efficient querying
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_team_strength_team_season", "team", "season", "gameweek"),
        Index("ix_team_strength_current", "team", "season", "gameweek"),
        # Strength-based queries
        Index("ix_team_strength_attacking_home", "attacking_strength_home"),
        Index("ix_team_strength_defensive_home", "defensive_strength_home"),
        Index(
            "ix_team_strength_overall", "overall_strength_home", "overall_strength_away"
        ),
        # Form and trend queries
        Index("ix_team_strength_form", "form_weighted_strength"),
        Index("ix_team_strength_momentum", "momentum_factor"),
        # Data quality filtering
        Index("ix_team_strength_quality", "data_quality_score"),
        # Time-based queries
        Index("ix_team_strength_calculated_at", "calculated_at"),
    )

    def __str__(self):
        return f"TeamStrength({self.team} {self.season} GW{self.gameweek}: H{self.overall_strength_home:.2f}/A{self.overall_strength_away:.2f})"


class TeamStrengthHistory(Base):
    """Historical team strength ratings for trend analysis and validation"""

    __tablename__ = "team_strength_history"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Historical strength values
    attacking_strength_home: Mapped[float]
    attacking_strength_away: Mapped[float]
    defensive_strength_home: Mapped[float]
    defensive_strength_away: Mapped[float]
    overall_strength_home: Mapped[float]
    overall_strength_away: Mapped[float]

    # Change from previous measurement
    attacking_strength_home_change: Mapped[float | None] = mapped_column(
        comment="Change from previous gameweek"
    )
    attacking_strength_away_change: Mapped[float | None] = mapped_column(
        comment="Change from previous gameweek"
    )
    defensive_strength_home_change: Mapped[float | None] = mapped_column(
        comment="Change from previous gameweek"
    )
    defensive_strength_away_change: Mapped[float | None] = mapped_column(
        comment="Change from previous gameweek"
    )

    # Performance validation
    actual_result: Mapped[str100 | None] = mapped_column(
        comment="Actual match result if applicable"
    )
    predicted_result: Mapped[str100 | None] = mapped_column(
        comment="Predicted result based on strength"
    )
    prediction_accuracy: Mapped[float | None] = mapped_column(
        comment="Accuracy of strength-based prediction"
    )

    # Context information
    matches_used_in_calculation: Mapped[int] = mapped_column(
        comment="Number of matches used for calculation"
    )
    exponential_smoothing_alpha: Mapped[float] = mapped_column(
        comment="Alpha parameter used in exponential smoothing"
    )

    # Calculation metadata
    calculation_trigger: Mapped[str100] = mapped_column(
        comment="What triggered this calculation"
    )
    calculation_duration_ms: Mapped[int | None] = mapped_column(
        comment="Time taken to calculate"
    )

    # Timestamp
    calculated_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when calculated"
    )

    # Define indexes for historical analysis
    __table_args__ = (
        # Time series queries
        Index("ix_team_strength_history_team_time", "team", "season", "gameweek"),
        Index("ix_team_strength_history_calculated_at", "calculated_at"),
        # Validation queries
        Index("ix_team_strength_history_validation", "prediction_accuracy"),
        Index("ix_team_strength_history_results", "actual_result", "predicted_result"),
        # Change analysis
        Index(
            "ix_team_strength_history_changes",
            "attacking_strength_home_change",
            "defensive_strength_home_change",
        ),
    )

    def __str__(self):
        return f"TeamStrengthHistory({self.team} {self.season} GW{self.gameweek} calculated at {self.calculated_at})"


class FixtureDifficulty(Base):
    """Store comprehensive fixture difficulty ratings and component breakdowns."""

    __tablename__ = "fixture_difficulty"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixture.fixture_id"))
    fixture: Mapped["Fixture"] = relationship()
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Core difficulty ratings (1-5 scale)
    home_difficulty: Mapped[float] = mapped_column(
        comment="Home team difficulty rating (1=easiest, 5=hardest)"
    )
    away_difficulty: Mapped[float] = mapped_column(
        comment="Away team difficulty rating (1=easiest, 5=hardest)"
    )

    # Expected win probabilities (0-1 scale)
    home_expected_score: Mapped[float] = mapped_column(
        comment="Home team expected win probability"
    )
    away_expected_score: Mapped[float] = mapped_column(
        comment="Away team expected win probability"
    )

    # Component ratings - Elo system
    home_elo_rating: Mapped[float] = mapped_column(
        comment="Home team Elo rating at time of calculation"
    )
    away_elo_rating: Mapped[float] = mapped_column(
        comment="Away team Elo rating at time of calculation"
    )
    elo_home_advantage: Mapped[float] = mapped_column(
        comment="Home advantage in Elo points"
    )

    # Component ratings - Form analysis
    home_form_score: Mapped[float] = mapped_column(
        comment="Home team recent form score (0-1 scale)"
    )
    away_form_score: Mapped[float] = mapped_column(
        comment="Away team recent form score (0-1 scale)"
    )
    form_adjustment: Mapped[float] = mapped_column(
        comment="Form-based adjustment to expected score"
    )

    # Component ratings - Fixture congestion
    home_congestion_factor: Mapped[float] = mapped_column(
        comment="Home team fixture congestion factor (-1 to 1)"
    )
    away_congestion_factor: Mapped[float] = mapped_column(
        comment="Away team fixture congestion factor (-1 to 1)"
    )
    congestion_adjustment: Mapped[float] = mapped_column(
        comment="Congestion-based adjustment to expected score"
    )

    # Component ratings - Head-to-head
    h2h_home_advantage: Mapped[float] = mapped_column(
        comment="Historical home advantage for this matchup"
    )
    h2h_recent_performance: Mapped[float] = mapped_column(
        comment="Recent H2H performance of home team"
    )
    h2h_matches_analyzed: Mapped[int] = mapped_column(
        comment="Number of H2H matches analyzed"
    )
    h2h_adjustment: Mapped[float] = mapped_column(
        comment="H2H-based adjustment to expected score"
    )

    # Calculation metadata
    calculation_method: Mapped[str100] = mapped_column(
        default="elo_comprehensive", comment="Method used for calculation"
    )
    calculation_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of calculation algorithm"
    )
    calculated_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when difficulty was calculated"
    )
    data_quality_score: Mapped[float | None] = mapped_column(
        comment="Quality score of underlying data (0-1 scale)"
    )

    # Validation tracking
    actual_result: Mapped[str100 | None] = mapped_column(
        comment="Actual match result (H/A/D) for validation"
    )
    prediction_accuracy: Mapped[float | None] = mapped_column(
        comment="Accuracy of prediction vs actual result"
    )
    validation_date: Mapped[str100 | None] = mapped_column(
        comment="Date when validation was performed"
    )

    # Define indexes for efficient querying
    __table_args__ = (
        # Primary query patterns
        Index("ix_fixture_difficulty_fixture_season", "fixture_id", "season"),
        Index("ix_fixture_difficulty_gameweek", "gameweek", "season"),
        # Team-based queries
        Index("ix_fixture_difficulty_home_team", "season", "gameweek"),
        Index("ix_fixture_difficulty_away_team", "season", "gameweek"),
        # Difficulty-based filtering
        Index("ix_fixture_difficulty_home_rating", "home_difficulty"),
        Index("ix_fixture_difficulty_away_rating", "away_difficulty"),
        # Validation queries
        Index(
            "ix_fixture_difficulty_validation", "actual_result", "prediction_accuracy"
        ),
        # Time-based queries
        Index("ix_fixture_difficulty_calculated_at", "calculated_at"),
    )

    def __str__(self):
        return f"FixtureDifficulty(GW{self.gameweek} {self.season}: Home={self.home_difficulty:.1f}, Away={self.away_difficulty:.1f})"


class PenaltyTakerHistory(Base):
    """Track penalty taker assignments and changes over time for audit and analysis."""

    __tablename__ = "penalty_taker_history"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    player: Mapped["Player"] = relationship(foreign_keys=[player_id])
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int | None]  # Can be null for season-level assignments

    # Assignment details
    is_primary_taker: Mapped[bool] = mapped_column(
        default=False, comment="Whether player is primary penalty taker"
    )
    confidence_score: Mapped[float] = mapped_column(
        comment="Confidence score for this assignment (0-1 scale)"
    )
    assignment_source: Mapped[str100]  # "analysis", "attributes", "manual", "external"

    # Historical penalty data at time of assignment
    penalties_taken_total: Mapped[int] = mapped_column(
        default=0, comment="Total penalties taken by player at assignment time"
    )
    penalties_scored_total: Mapped[int] = mapped_column(
        default=0, comment="Total penalties scored by player at assignment time"
    )
    success_rate: Mapped[float | None] = mapped_column(
        comment="Penalty conversion rate at assignment time"
    )

    # Change tracking
    assignment_date: Mapped[str100]  # ISO datetime string when assignment was made
    changed_from_player_id: Mapped[int | None] = mapped_column(
        ForeignKey("player.player_id"),
        comment="Previous penalty taker if this is a change",
    )
    change_reason: Mapped[str | None] = mapped_column(
        String(500), comment="Reason for penalty taker change"
    )

    # Validation and performance tracking
    is_active: Mapped[bool] = mapped_column(
        default=True, comment="Whether this assignment is currently active"
    )
    validated_by_actual_penalty: Mapped[bool] = mapped_column(
        default=False,
        comment="Whether assignment was validated by actual penalty event",
    )
    validation_date: Mapped[str100 | None]  # ISO datetime when validation occurred

    # Analysis metadata
    analysis_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of analysis algorithm used"
    )
    data_quality_score: Mapped[float | None] = mapped_column(
        comment="Quality score of underlying data (0-1 scale)"
    )

    def __str__(self):
        primary_str = "Primary" if self.is_primary_taker else "Secondary"
        return f"PenaltyTakerHistory({primary_str} {self.player.name} for {self.team} {self.season}, confidence={self.confidence_score:.2f})"


class RotationRiskPrediction(Base):
    """Store rotation risk predictions for players with detailed factor breakdown."""

    __tablename__ = "rotation_risk_prediction"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    player: Mapped["Player"] = relationship()
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Core prediction
    rotation_risk: Mapped[float] = mapped_column(
        comment="Overall rotation risk score (0-1 scale)"
    )
    confidence: Mapped[float] = mapped_column(
        comment="Prediction confidence (0-1 scale)"
    )
    model_version: Mapped[str100] = mapped_column(
        comment="Version of prediction model used"
    )

    # Individual risk factors
    fixture_congestion: Mapped[float] = mapped_column(
        comment="Fixture congestion factor (0-1)"
    )
    fatigue_risk: Mapped[float] = mapped_column(
        comment="Player fatigue risk from recent minutes (0-1)"
    )
    age_factor: Mapped[float] = mapped_column(
        comment="Age-related rotation tendency (0-1)"
    )
    manager_rotation_tendency: Mapped[float] = mapped_column(
        comment="Manager's general rotation rate (0-1)"
    )
    position_rotation_rate: Mapped[float] = mapped_column(
        comment="Position-specific rotation rate (0-1)"
    )
    player_importance: Mapped[float] = mapped_column(
        comment="Player importance to team (0-1, higher = more important)"
    )
    competition_importance: Mapped[float] = mapped_column(
        comment="Importance of upcoming competition (0-1)"
    )
    team_depth: Mapped[float] = mapped_column(
        comment="Team depth in player's position (0-1)"
    )
    recovery_time: Mapped[float] = mapped_column(comment="Recovery time factor (0-1)")

    # Prediction metadata
    predicted_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when prediction was made"
    )
    data_quality_score: Mapped[float | None] = mapped_column(
        comment="Quality of underlying data (0-1)"
    )
    calculation_method: Mapped[str100] = mapped_column(
        default="ml_ensemble", comment="Method used for calculation"
    )

    # Validation tracking
    actual_outcome: Mapped[bool | None] = mapped_column(
        comment="Actual rotation outcome if known (True=rotated, False=started)"
    )
    prediction_accuracy: Mapped[float | None] = mapped_column(
        comment="Accuracy of this prediction if validated"
    )
    validated_at: Mapped[str100 | None] = mapped_column(
        comment="ISO datetime when outcome was validated"
    )

    # Define indexes for efficient querying
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_rotation_risk_player_gameweek", "player_id", "season", "gameweek"),
        Index("ix_rotation_risk_gameweek", "gameweek", "season"),
        # Risk-based filtering
        Index("ix_rotation_risk_score", "rotation_risk"),
        Index("ix_rotation_risk_high_risk", "rotation_risk", "confidence"),
        # Factor analysis
        Index("ix_rotation_risk_fatigue", "fatigue_risk"),
        Index("ix_rotation_risk_congestion", "fixture_congestion"),
        # Time-based queries
        Index("ix_rotation_risk_predicted_at", "predicted_at"),
        # Validation queries
        Index("ix_rotation_risk_validation", "actual_outcome", "prediction_accuracy"),
    )

    def __str__(self):
        return f"RotationRisk({self.player.name} GW{self.gameweek}: {self.rotation_risk:.2f} risk, {self.confidence:.2f} confidence)"


class ManagerRotationPattern(Base):
    """Store manager-specific rotation patterns and behaviors."""

    __tablename__ = "manager_rotation_pattern"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    team: Mapped[str100]
    season: Mapped[str100]
    manager_name: Mapped[str100 | None] = mapped_column(comment="Manager name if known")

    # Overall rotation metrics
    avg_rotation_rate: Mapped[float] = mapped_column(
        comment="Overall rotation frequency (0-1)"
    )
    congestion_response: Mapped[float] = mapped_column(
        comment="Increased rotation during fixture congestion (0-2)"
    )
    sample_size: Mapped[int] = mapped_column(comment="Number of fixtures analyzed")

    # Position-specific rotation rates
    gk_rotation_rate: Mapped[float] = mapped_column(comment="Goalkeeper rotation rate")
    def_rotation_rate: Mapped[float] = mapped_column(comment="Defender rotation rate")
    mid_rotation_rate: Mapped[float] = mapped_column(comment="Midfielder rotation rate")
    fwd_rotation_rate: Mapped[float] = mapped_column(comment="Forward rotation rate")

    # Competition preferences (team selection strength by competition)
    premier_league_priority: Mapped[float] = mapped_column(
        default=1.0, comment="Premier League team strength priority"
    )
    champions_league_priority: Mapped[float] = mapped_column(
        default=0.9, comment="Champions League team strength priority"
    )
    europa_league_priority: Mapped[float] = mapped_column(
        default=0.8, comment="Europa League team strength priority"
    )
    fa_cup_priority: Mapped[float] = mapped_column(
        default=0.6, comment="FA Cup team strength priority"
    )
    carabao_cup_priority: Mapped[float] = mapped_column(
        default=0.4, comment="Carabao Cup team strength priority"
    )

    # Behavioral indicators
    age_bias: Mapped[float] = mapped_column(
        default=0.0, comment="Tendency to rotate older players (-1 to 1)"
    )
    youth_integration: Mapped[float] = mapped_column(
        default=0.0, comment="Tendency to give young players chances (0-1)"
    )
    injury_caution: Mapped[float] = mapped_column(
        default=0.5, comment="Caution level with injury-prone players (0-1)"
    )

    # Analysis metadata
    analysis_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of analysis algorithm"
    )
    analyzed_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when analysis was performed"
    )
    data_quality_score: Mapped[float] = mapped_column(
        comment="Quality of underlying data (0-1)"
    )

    # Validation metrics
    prediction_accuracy: Mapped[float | None] = mapped_column(
        comment="Accuracy when using these patterns for prediction"
    )
    validation_period: Mapped[str100 | None] = mapped_column(
        comment="Period used for validation"
    )

    # Define indexes for manager analysis
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_manager_pattern_team_season", "team", "season"),
        Index("ix_manager_pattern_manager", "manager_name", "season"),
        # Rotation rate queries
        Index("ix_manager_pattern_rotation_rate", "avg_rotation_rate"),
        Index("ix_manager_pattern_congestion_response", "congestion_response"),
        # Position-specific queries
        Index(
            "ix_manager_pattern_position_rates",
            "def_rotation_rate",
            "mid_rotation_rate",
            "fwd_rotation_rate",
        ),
        # Time-based queries
        Index("ix_manager_pattern_analyzed_at", "analyzed_at"),
    )

    def __str__(self):
        manager_str = self.manager_name or "Unknown Manager"
        return f"ManagerPattern({manager_str} - {self.team} {self.season}: {self.avg_rotation_rate:.2f} avg rotation)"


class FixtureCongestion(Base):
    """Store fixture congestion analysis for teams and gameweeks."""

    __tablename__ = "fixture_congestion"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Core congestion metrics
    congestion_score: Mapped[float] = mapped_column(
        comment="Overall congestion level (0-1)"
    )
    fixtures_7_days: Mapped[int] = mapped_column(
        comment="Number of fixtures in 7-day window"
    )
    fixtures_14_days: Mapped[int] = mapped_column(
        comment="Number of fixtures in 14-day window"
    )
    travel_burden: Mapped[float] = mapped_column(comment="Travel-adjusted fixture load")
    recovery_time: Mapped[float] = mapped_column(comment="Days since last fixture")

    # Upcoming fixture details
    next_fixture_days: Mapped[int | None] = mapped_column(
        comment="Days until next fixture"
    )
    next_fixture_competition: Mapped[str100 | None] = mapped_column(
        comment="Type of next fixture competition"
    )
    next_fixture_is_away: Mapped[bool | None] = mapped_column(
        comment="Whether next fixture is away"
    )

    # Historical context
    season_fixture_load: Mapped[float] = mapped_column(
        comment="Season-to-date fixture load compared to typical"
    )
    congestion_rank: Mapped[int | None] = mapped_column(
        comment="Congestion rank among all teams this gameweek"
    )

    # Impact factors
    european_competition: Mapped[bool] = mapped_column(
        default=False, comment="Whether team is in European competition"
    )
    international_break_impact: Mapped[float] = mapped_column(
        default=0.0, comment="Impact of recent international break"
    )
    injury_crisis_multiplier: Mapped[float] = mapped_column(
        default=1.0, comment="Multiplier for injury-related squad limitations"
    )

    # Calculation metadata
    calculated_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when calculated"
    )
    calculation_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of calculation algorithm"
    )
    data_quality_score: Mapped[float] = mapped_column(
        comment="Quality of underlying fixture data (0-1)"
    )

    # Define indexes for congestion analysis
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_fixture_congestion_team_gameweek", "team", "season", "gameweek"),
        Index("ix_fixture_congestion_gameweek", "gameweek", "season"),
        # Congestion-based queries
        Index("ix_fixture_congestion_score", "congestion_score"),
        Index("ix_fixture_congestion_high", "congestion_score", "team"),
        # Recovery time analysis
        Index("ix_fixture_congestion_recovery", "recovery_time"),
        # Competition impact
        Index(
            "ix_fixture_congestion_european", "european_competition", "congestion_score"
        ),
        # Time-based queries
        Index("ix_fixture_congestion_calculated_at", "calculated_at"),
    )

    def __str__(self):
        return f"FixtureCongestion({self.team} GW{self.gameweek}: {self.congestion_score:.2f} score, {self.fixtures_7_days} fixtures in 7 days)"


class TeamHomeAdvantageData(Base):
    """Store team-specific home advantage calculations with Bayesian analysis."""

    __tablename__ = "team_home_advantage"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Core home advantage metrics
    raw_home_advantage: Mapped[float] = mapped_column(
        comment="Raw calculated home advantage (goals per game)"
    )
    adjusted_home_advantage: Mapped[float] = mapped_column(
        comment="Bayesian shrinkage adjusted home advantage"
    )
    league_baseline: Mapped[float] = mapped_column(
        comment="League-wide baseline home advantage"
    )

    # Bayesian analysis components
    shrinkage_factor: Mapped[float] = mapped_column(
        comment="Alpha parameter for Bayesian shrinkage (0-1)"
    )
    confidence: Mapped[float] = mapped_column(
        comment="Confidence in the estimate (0-1)"
    )

    # Sample size information
    home_matches: Mapped[int] = mapped_column(comment="Number of home matches analyzed")
    away_matches: Mapped[int] = mapped_column(comment="Number of away matches analyzed")

    # Performance breakdown
    home_goals_per_game: Mapped[float] = mapped_column(
        comment="Goals scored per home game"
    )
    away_goals_per_game: Mapped[float] = mapped_column(
        comment="Goals scored per away game"
    )
    home_goals_against_per_game: Mapped[float] = mapped_column(
        comment="Goals conceded per home game"
    )
    away_goals_against_per_game: Mapped[float] = mapped_column(
        comment="Goals conceded per away game"
    )

    # Adjustment factors
    stadium_factor: Mapped[float] = mapped_column(
        comment="Stadium-specific adjustment factor"
    )
    empty_stadium_adjustment: Mapped[float] = mapped_column(
        comment="COVID/empty stadium adjustment"
    )

    # Calculation metadata
    calculation_method: Mapped[str100] = mapped_column(
        default="bayesian_shrinkage", comment="Method used for calculation"
    )
    calculation_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of calculation algorithm"
    )
    calculated_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when calculated"
    )

    # Define indexes for efficient querying
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_team_home_advantage_team_season", "team", "season", "gameweek"),
        Index("ix_team_home_advantage_current", "team", "season", "gameweek"),
        # Advantage-based queries
        Index("ix_team_home_advantage_raw", "raw_home_advantage"),
        Index("ix_team_home_advantage_adjusted", "adjusted_home_advantage"),
        # Confidence-based filtering
        Index("ix_team_home_advantage_confidence", "confidence"),
        # Time-based queries
        Index("ix_team_home_advantage_calculated_at", "calculated_at"),
    )

    def __str__(self):
        return f"TeamHomeAdvantage({self.team} {self.season} GW{self.gameweek}: {self.adjusted_home_advantage:.3f})"


class PlayerHomeAwayData(Base):
    """Store player-specific home/away performance adjustments with statistical significance testing."""

    __tablename__ = "player_home_away"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    player: Mapped["Player"] = relationship()
    season: Mapped[str100]

    # Core adjustment factors
    home_adjustment: Mapped[float] = mapped_column(
        comment="Home performance adjustment factor (-1 to 1)"
    )
    away_adjustment: Mapped[float] = mapped_column(
        comment="Away performance adjustment factor (-1 to 1)"
    )

    # Statistical significance
    is_significant: Mapped[bool] = mapped_column(
        comment="Whether difference is statistically significant"
    )
    p_value: Mapped[float] = mapped_column(comment="P-value from statistical test")
    confidence: Mapped[float] = mapped_column(
        comment="Confidence in the adjustment (0-1)"
    )

    # Sample size information
    home_games: Mapped[int] = mapped_column(comment="Number of home games analyzed")
    away_games: Mapped[int] = mapped_column(comment="Number of away games analyzed")
    total_games: Mapped[int] = mapped_column(comment="Total games analyzed")

    # Performance metrics
    home_points_per_game: Mapped[float] = mapped_column(
        comment="Average FPL points per home game"
    )
    away_points_per_game: Mapped[float] = mapped_column(
        comment="Average FPL points per away game"
    )

    # Calculation metadata
    analysis_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of analysis algorithm"
    )
    analyzed_at: Mapped[str100] = mapped_column(comment="ISO datetime when analyzed")
    lookback_seasons: Mapped[int] = mapped_column(
        default=2, comment="Number of seasons analyzed"
    )

    # Define indexes for efficient querying
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_player_home_away_player_season", "player_id", "season"),
        # Significance-based filtering
        Index("ix_player_home_away_significant", "is_significant", "confidence"),
        # Adjustment-based queries
        Index("ix_player_home_away_home_adj", "home_adjustment"),
        Index("ix_player_home_away_away_adj", "away_adjustment"),
        # Sample size filtering
        Index("ix_player_home_away_total_games", "total_games"),
        # Time-based queries
        Index("ix_player_home_away_analyzed_at", "analyzed_at"),
    )

    def __str__(self):
        significance_str = "Significant" if self.is_significant else "Not Significant"
        return f"PlayerHomeAway({self.player.name} {self.season}: H{self.home_adjustment:.2%}/A{self.away_adjustment:.2%}, {significance_str})"


class HomeAwayAdjustmentHistory(Base):
    """Track history of home/away adjustments for validation and analysis."""

    __tablename__ = "home_away_adjustment_history"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    entity_type: Mapped[str100]  # "team" or "player"
    entity_id: Mapped[int]  # team_id or player_id
    entity_name: Mapped[str100]  # team name or player name
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Adjustment values
    home_adjustment: Mapped[float] = mapped_column(
        comment="Home adjustment factor applied"
    )
    away_adjustment: Mapped[float] = mapped_column(
        comment="Away adjustment factor applied"
    )

    # Application context
    prediction_type: Mapped[str100]  # "points", "goals", "assists", etc.
    base_prediction: Mapped[float] = mapped_column(
        comment="Original prediction before adjustment"
    )
    adjusted_prediction: Mapped[float] = mapped_column(
        comment="Final prediction after adjustment"
    )
    adjustment_impact: Mapped[float] = mapped_column(
        comment="Total impact of adjustment"
    )

    # Metadata
    adjustment_method: Mapped[str100] = mapped_column(
        comment="Method used for adjustment"
    )
    adjustment_version: Mapped[str100] = mapped_column(
        comment="Version of adjustment algorithm"
    )
    applied_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when adjustment was applied"
    )

    # Validation tracking (filled in later when actual results are known)
    actual_result: Mapped[float | None] = mapped_column(
        comment="Actual outcome for validation"
    )
    prediction_error: Mapped[float | None] = mapped_column(
        comment="Error in adjusted prediction"
    )
    improvement_over_base: Mapped[float | None] = mapped_column(
        comment="Improvement over base prediction"
    )
    validated_at: Mapped[str100 | None] = mapped_column(
        comment="ISO datetime when validated"
    )

    # Define indexes for historical analysis
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_home_away_history_entity", "entity_type", "entity_id", "season"),
        Index("ix_home_away_history_gameweek", "gameweek", "season"),
        # Impact analysis
        Index("ix_home_away_history_impact", "adjustment_impact"),
        Index("ix_home_away_history_improvement", "improvement_over_base"),
        # Validation queries
        Index("ix_home_away_history_validated", "actual_result", "prediction_error"),
        # Time-based queries
        Index("ix_home_away_history_applied_at", "applied_at"),
    )

    def __str__(self):
        return f"HomeAwayHistory({self.entity_type}:{self.entity_name} GW{self.gameweek}: {self.adjustment_impact:+.2f})"


class VenuePerformanceAnalysis(Base):
    """Detailed venue-specific performance analysis for teams and players."""

    __tablename__ = "venue_performance_analysis"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    entity_type: Mapped[str100]  # "team" or "player"
    entity_id: Mapped[int]
    entity_name: Mapped[str100]
    season: Mapped[str100]
    venue: Mapped[str100]  # "home" or "away"

    # Performance metrics
    games_played: Mapped[int] = mapped_column(
        comment="Number of games played at this venue"
    )
    total_points: Mapped[float] = mapped_column(
        comment="Total FPL points earned at this venue"
    )
    points_per_game: Mapped[float] = mapped_column(
        comment="Average points per game at this venue"
    )

    # Goal involvement metrics (for players)
    goals_scored: Mapped[int | None] = mapped_column(
        comment="Goals scored at this venue"
    )
    assists: Mapped[int | None] = mapped_column(comment="Assists at this venue")
    clean_sheets: Mapped[int | None] = mapped_column(
        comment="Clean sheets at this venue (for defenders/GK)"
    )

    # Team-level metrics
    goals_for: Mapped[int | None] = mapped_column(
        comment="Goals scored by team at this venue"
    )
    goals_against: Mapped[int | None] = mapped_column(
        comment="Goals conceded by team at this venue"
    )
    wins: Mapped[int | None] = mapped_column(comment="Wins at this venue")
    draws: Mapped[int | None] = mapped_column(comment="Draws at this venue")
    losses: Mapped[int | None] = mapped_column(comment="Losses at this venue")

    # Form indicators
    recent_form: Mapped[float] = mapped_column(
        comment="Recent form at this venue (last 5 games)"
    )
    trend: Mapped[float] = mapped_column(comment="Performance trend (-1 to 1)")

    # External factors
    opponent_strength_avg: Mapped[float | None] = mapped_column(
        comment="Average opponent strength faced"
    )
    fixture_difficulty_avg: Mapped[float | None] = mapped_column(
        comment="Average fixture difficulty"
    )

    # Statistical measures
    performance_variance: Mapped[float] = mapped_column(
        comment="Variance in performance at this venue"
    )
    consistency_score: Mapped[float] = mapped_column(
        comment="Consistency of performance (0-1)"
    )

    # Calculation metadata
    analysis_period_start: Mapped[str100] = mapped_column(
        comment="Start date of analysis period"
    )
    analysis_period_end: Mapped[str100] = mapped_column(
        comment="End date of analysis period"
    )
    analyzed_at: Mapped[str100] = mapped_column(comment="ISO datetime when analyzed")
    analysis_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of analysis algorithm"
    )

    # Define indexes for performance analysis
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_venue_performance_entity", "entity_type", "entity_id", "venue"),
        Index("ix_venue_performance_season", "season", "venue"),
        # Performance-based queries
        Index("ix_venue_performance_points", "points_per_game"),
        Index("ix_venue_performance_consistency", "consistency_score"),
        # Form analysis
        Index("ix_venue_performance_form", "recent_form"),
        Index("ix_venue_performance_trend", "trend"),
        # Time-based queries
        Index("ix_venue_performance_analyzed_at", "analyzed_at"),
    )

    def __str__(self):
        return f"VenuePerformance({self.entity_type}:{self.entity_name} {self.venue} {self.season}: {self.points_per_game:.1f} PPG)"


class SetPieceEvent(Base):
    """Store individual set piece events for detailed analysis."""

    __tablename__ = "set_piece_event"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    player: Mapped["Player"] = relationship()
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int]

    # Set piece details
    set_piece_type: Mapped[str100] = mapped_column(
        comment="Type of set piece (corner_left, free_kick_close, etc.)"
    )
    outcome: Mapped[str100] = mapped_column(
        comment="Outcome of set piece (goal, assist, key_pass, shot, none)"
    )
    successful: Mapped[bool] = mapped_column(
        default=False, comment="Whether set piece led to positive outcome"
    )

    # Match context
    opponent: Mapped[str100]
    venue: Mapped[str100] = mapped_column(comment="home or away")
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))
    fixture: Mapped["Fixture"] = relationship()
    minute: Mapped[int | None] = mapped_column(comment="Minute of set piece")

    # Set piece specific details
    foot_used: Mapped[str100 | None] = mapped_column(comment="left, right, or both")
    distance: Mapped[int | None] = mapped_column(
        comment="Distance in yards for free kicks"
    )
    area: Mapped[str100 | None] = mapped_column(
        comment="Area of pitch for corners/throws"
    )

    # Analysis metadata
    detected_method: Mapped[str100] = mapped_column(
        comment="How the event was detected (heuristic, video, manual)"
    )
    confidence: Mapped[float] = mapped_column(
        comment="Confidence in event detection (0-1)"
    )
    recorded_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when event was recorded"
    )

    # Define indexes for efficient querying
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_set_piece_event_player", "player_id", "season", "set_piece_type"),
        Index("ix_set_piece_event_team", "team", "season", "gameweek"),
        Index("ix_set_piece_event_type", "set_piece_type", "season"),
        # Outcome analysis
        Index("ix_set_piece_event_outcome", "outcome", "successful"),
        Index("ix_set_piece_event_venue", "venue", "set_piece_type"),
        # Time-based queries
        Index("ix_set_piece_event_gameweek", "season", "gameweek"),
        Index("ix_set_piece_event_recorded_at", "recorded_at"),
    )

    def __str__(self):
        return f"SetPieceEvent({self.player.name} {self.set_piece_type} vs {self.opponent}: {self.outcome})"


class SetPieceSpecialist(Base):
    """Store set piece specialist assignments with confidence and historical tracking."""

    __tablename__ = "set_piece_specialist"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    player: Mapped["Player"] = relationship(foreign_keys=[player_id])
    team: Mapped[str100]
    season: Mapped[str100]
    gameweek: Mapped[int | None] = mapped_column(
        comment="Gameweek when assignment was made (null for season-level)"
    )

    # Specialist assignment details
    set_piece_type: Mapped[str100] = mapped_column(
        comment="Type of set piece specialist"
    )
    is_primary: Mapped[bool] = mapped_column(
        default=False, comment="Whether player is primary specialist"
    )
    confidence_score: Mapped[float] = mapped_column(
        comment="Confidence score for this assignment (0-1)"
    )
    assignment_source: Mapped[str100] = mapped_column(
        comment="analysis, attributes, manual, external"
    )

    # Historical performance at assignment time
    total_taken: Mapped[int] = mapped_column(
        default=0, comment="Total set pieces taken at assignment time"
    )
    successful_outcomes: Mapped[int] = mapped_column(
        default=0, comment="Successful outcomes at assignment time"
    )
    goals_from: Mapped[int] = mapped_column(default=0, comment="Goals from set pieces")
    assists_from: Mapped[int] = mapped_column(
        default=0, comment="Assists from set pieces"
    )
    success_rate: Mapped[float | None] = mapped_column(
        comment="Success rate at assignment time"
    )

    # Change tracking
    assignment_date: Mapped[str100] = mapped_column(
        comment="ISO datetime when assignment was made"
    )
    changed_from_player_id: Mapped[int | None] = mapped_column(
        ForeignKey("player.player_id"),
        comment="Previous specialist if this is a change",
    )
    change_reason: Mapped[str | None] = mapped_column(
        String(500), comment="Reason for specialist change"
    )

    # Validation and performance tracking
    is_active: Mapped[bool] = mapped_column(
        default=True, comment="Whether this assignment is currently active"
    )
    validated_by_actual_event: Mapped[bool] = mapped_column(
        default=False, comment="Whether assignment was validated by actual set piece"
    )
    validation_date: Mapped[str100 | None] = mapped_column(
        comment="ISO datetime when validation occurred"
    )

    # Performance since assignment
    events_since_assignment: Mapped[int] = mapped_column(
        default=0, comment="Set pieces taken since assignment"
    )
    successful_since_assignment: Mapped[int] = mapped_column(
        default=0, comment="Successful outcomes since assignment"
    )

    # Analysis metadata
    analysis_version: Mapped[str100] = mapped_column(
        default="1.0.0", comment="Version of analysis algorithm used"
    )
    data_quality_score: Mapped[float | None] = mapped_column(
        comment="Quality score of underlying data (0-1)"
    )

    # Define indexes for specialist analysis
    __table_args__ = (
        # Primary lookup patterns
        Index(
            "ix_set_piece_specialist_player", "player_id", "set_piece_type", "season"
        ),
        Index("ix_set_piece_specialist_team", "team", "season", "set_piece_type"),
        Index("ix_set_piece_specialist_type", "set_piece_type", "is_primary"),
        # Assignment tracking
        Index("ix_set_piece_specialist_assignment", "assignment_date"),
        Index("ix_set_piece_specialist_change", "changed_from_player_id"),
        # Performance analysis
        Index("ix_set_piece_specialist_confidence", "confidence_score"),
        Index("ix_set_piece_specialist_success", "success_rate"),
        # Active assignments
        Index("ix_set_piece_specialist_active", "is_active", "season"),
        # Validation queries
        Index(
            "ix_set_piece_specialist_validated",
            "validated_by_actual_event",
            "validation_date",
        ),
    )

    def __str__(self):
        primary_str = "Primary" if self.is_primary else "Secondary"
        return f"SetPieceSpecialist({primary_str} {self.player.name} - {self.set_piece_type} for {self.team}, confidence={self.confidence_score:.2f})"


class SetPieceStatistics(Base):
    """Aggregated set piece statistics for players over specific periods."""

    __tablename__ = "set_piece_statistics"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    player: Mapped["Player"] = relationship()
    team: Mapped[str100]
    season: Mapped[str100]
    set_piece_type: Mapped[str100]

    # Aggregation period
    period_type: Mapped[str100] = mapped_column(
        comment="season, gameweek_range, last_n_games"
    )
    period_start: Mapped[str100 | None] = mapped_column(
        comment="Start of aggregation period"
    )
    period_end: Mapped[str100 | None] = mapped_column(
        comment="End of aggregation period"
    )
    games_included: Mapped[int] = mapped_column(
        comment="Number of games included in statistics"
    )

    # Core statistics
    total_attempts: Mapped[int] = mapped_column(
        default=0, comment="Total set pieces attempted"
    )
    successful_outcomes: Mapped[int] = mapped_column(
        default=0, comment="Set pieces with positive outcomes"
    )
    goals_scored: Mapped[int] = mapped_column(
        default=0, comment="Goals directly from set pieces"
    )
    assists_provided: Mapped[int] = mapped_column(
        default=0, comment="Assists from set pieces"
    )
    key_passes_made: Mapped[int] = mapped_column(
        default=0, comment="Key passes from set pieces"
    )
    shots_created: Mapped[int] = mapped_column(
        default=0, comment="Shots created from set pieces"
    )

    # Calculated rates
    success_rate: Mapped[float] = mapped_column(
        default=0.0, comment="Percentage of successful outcomes"
    )
    goals_per_attempt: Mapped[float] = mapped_column(
        default=0.0, comment="Goals per set piece attempt"
    )
    assists_per_attempt: Mapped[float] = mapped_column(
        default=0.0, comment="Assists per set piece attempt"
    )
    key_passes_per_attempt: Mapped[float] = mapped_column(
        default=0.0, comment="Key passes per set piece attempt"
    )

    # Venue breakdown
    home_attempts: Mapped[int] = mapped_column(
        default=0, comment="Set pieces attempted at home"
    )
    away_attempts: Mapped[int] = mapped_column(
        default=0, comment="Set pieces attempted away"
    )
    home_success_rate: Mapped[float] = mapped_column(
        default=0.0, comment="Success rate at home"
    )
    away_success_rate: Mapped[float] = mapped_column(
        default=0.0, comment="Success rate away"
    )

    # Advanced metrics
    average_distance: Mapped[float | None] = mapped_column(
        comment="Average distance for free kicks"
    )
    preferred_foot: Mapped[str100 | None] = mapped_column(
        comment="Most commonly used foot"
    )
    preferred_area: Mapped[str100 | None] = mapped_column(comment="Most targeted area")
    consistency_score: Mapped[float] = mapped_column(
        default=0.0, comment="Consistency of performance (0-1)"
    )

    # Time-based analysis
    recent_form: Mapped[float] = mapped_column(
        default=0.0, comment="Performance in last 5 attempts"
    )
    trend_direction: Mapped[str100] = mapped_column(
        default="stable", comment="improving, declining, stable"
    )
    trend_confidence: Mapped[float] = mapped_column(
        default=0.0, comment="Confidence in trend analysis"
    )

    # Calculation metadata
    calculated_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when statistics were calculated"
    )
    calculation_method: Mapped[str100] = mapped_column(
        default="standard", comment="Method used for calculation"
    )
    data_quality_score: Mapped[float] = mapped_column(
        comment="Quality of underlying data (0-1)"
    )

    # Define indexes for statistical analysis
    __table_args__ = (
        # Primary lookup patterns
        Index("ix_set_piece_stats_player", "player_id", "set_piece_type", "season"),
        Index("ix_set_piece_stats_team", "team", "season", "set_piece_type"),
        Index("ix_set_piece_stats_type", "set_piece_type", "season"),
        # Performance-based queries
        Index("ix_set_piece_stats_success_rate", "success_rate"),
        Index("ix_set_piece_stats_goals_rate", "goals_per_attempt"),
        Index("ix_set_piece_stats_consistency", "consistency_score"),
        # Trend analysis
        Index("ix_set_piece_stats_trend", "trend_direction", "trend_confidence"),
        Index("ix_set_piece_stats_form", "recent_form"),
        # Time-based queries
        Index("ix_set_piece_stats_calculated_at", "calculated_at"),
        Index("ix_set_piece_stats_period", "period_type", "period_start", "period_end"),
    )

    def __str__(self):
        return f"SetPieceStats({self.player.name} {self.set_piece_type} {self.season}: {self.success_rate:.1%} success, {self.total_attempts} attempts)"


class SetPieceAnalysisLog(Base):
    """Log of set piece analysis runs for audit and debugging."""

    __tablename__ = "set_piece_analysis_log"

    id: Mapped[intpk] = mapped_column(autoincrement=True)
    analysis_type: Mapped[str100] = mapped_column(comment="Type of analysis performed")
    analysis_version: Mapped[str100] = mapped_column(
        comment="Version of analysis algorithm"
    )

    # Scope of analysis
    team: Mapped[str100 | None] = mapped_column(
        comment="Team analyzed (null for all teams)"
    )
    season: Mapped[str100]
    set_piece_type: Mapped[str100 | None] = mapped_column(
        comment="Set piece type analyzed (null for all types)"
    )

    # Execution details
    started_at: Mapped[str100] = mapped_column(
        comment="ISO datetime when analysis started"
    )
    completed_at: Mapped[str100 | None] = mapped_column(
        comment="ISO datetime when analysis completed"
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        comment="Duration of analysis"
    )
    status: Mapped[str100] = mapped_column(
        comment="running, completed, failed, cancelled"
    )

    # Results summary
    players_analyzed: Mapped[int] = mapped_column(
        default=0, comment="Number of players analyzed"
    )
    events_processed: Mapped[int] = mapped_column(
        default=0, comment="Number of set piece events processed"
    )
    specialists_identified: Mapped[int] = mapped_column(
        default=0, comment="Number of specialists identified"
    )
    confidence_threshold: Mapped[float] = mapped_column(
        comment="Confidence threshold used"
    )

    # Data quality
    data_quality_score: Mapped[float | None] = mapped_column(
        comment="Overall data quality (0-1)"
    )
    events_with_low_confidence: Mapped[int] = mapped_column(
        default=0, comment="Events with confidence < 0.7"
    )
    missing_data_percentage: Mapped[float] = mapped_column(
        default=0.0, comment="Percentage of missing data"
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(
        String(1000), comment="Error message if analysis failed"
    )
    warnings_count: Mapped[int] = mapped_column(
        default=0, comment="Number of warnings generated"
    )

    # Performance metrics
    average_confidence: Mapped[float | None] = mapped_column(
        comment="Average confidence of assignments"
    )
    assignments_changed: Mapped[int] = mapped_column(
        default=0, comment="Number of specialist assignments changed"
    )

    def __str__(self):
        return f"SetPieceAnalysisLog({self.analysis_type} for {self.team or 'all teams'} {self.season}: {self.status})"


def get_connection_string() -> str:
    if AIRSENAL_DB_FILE and AIRSENAL_DB_URI:
        msg = "Please choose only ONE of AIRSENAL_DB_FILE and AIRSENAL_DB_URI"
        raise RuntimeError(msg)

    # postgres database specified by: AIRSENAL_DB{_URI, _USER, _PASSWORD}
    if AIRSENAL_DB_URI:
        if AIRSENAL_DB_PASSWORD is None:
            msg = "AIRSENAL_DB_PASSWORD must be defined when using a postgres database"
            raise KeyError(msg)
        if AIRSENAL_DB_USER is None:
            msg = "AIRSENAL_DB_USER must be defined when using a postgres database"
            raise KeyError(msg)

        return (
            f"postgresql://{AIRSENAL_DB_USER}:"
            f"{AIRSENAL_DB_PASSWORD}@{AIRSENAL_DB_URI}/airsenal"
        )

    # sqlite database in a local file with path specified by AIRSENAL_DB_FILE,
    # or AIRSENAL_HOME / data.db by default
    if not AIRSENAL_DB_FILE:
        db_file = str(AIRSENAL_HOME / "data.db")
        save_env("AIRSENAL_DB_FILE", db_file)
        return f"sqlite:///{db_file}"
    return f"sqlite:///{AIRSENAL_DB_FILE}"


def get_session():
    conn_str = get_connection_string()
    engine = create_engine(conn_str)

    Base.metadata.create_all(engine)
    # Bind the engine to the metadata of the Base class so that the
    # declaratives can be accessed through a DBSession instance
    # Note: Base.metadata.bind is deprecated in SQLAlchemy 2.0

    DBSession = sessionmaker(bind=engine, autoflush=False)
    return DBSession()


# global database session used by default throughout the package
session = get_session()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def clean_database():
    """
    Clean up database
    """
    engine = create_engine(get_connection_string())
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def database_is_empty(dbsession):
    """
    Basic check to determine whether the database is empty
    """
    return dbsession.query(Team).first() is None
