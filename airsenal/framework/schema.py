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
    xg_per_90: Mapped[float | None] = mapped_column(comment="Expected goals per 90 minutes")
    xa_per_90: Mapped[float | None] = mapped_column(comment="Expected assists per 90 minutes")
    xgi_per_90: Mapped[float | None] = mapped_column(comment="Expected goal involvements (xG + xA) per 90 minutes")

    # Form metrics - rolling averages of FPL points
    form_3_games: Mapped[float | None] = mapped_column(comment="Average FPL points over last 3 games")
    form_5_games: Mapped[float | None] = mapped_column(comment="Average FPL points over last 5 games")
    form_10_games: Mapped[float | None] = mapped_column(comment="Average FPL points over last 10 games")
    momentum: Mapped[float | None] = mapped_column(comment="Trend indicator for recent performance (-1 to 1)")

    # Fixture difficulty ratings for upcoming fixtures
    next_3_fixture_difficulty: Mapped[float | None] = mapped_column(comment="Average difficulty rating for next 3 fixtures (1-5 scale)")
    next_5_fixture_difficulty: Mapped[float | None] = mapped_column(comment="Average difficulty rating for next 5 fixtures (1-5 scale)")

    # Player role indicators
    is_penalty_taker: Mapped[bool] = mapped_column(default=False, comment="Whether player is primary penalty taker")
    is_free_kick_taker: Mapped[bool] = mapped_column(default=False, comment="Whether player is primary free kick taker")
    is_corner_taker: Mapped[bool] = mapped_column(default=False, comment="Whether player is primary corner taker")
    role_confidence: Mapped[float | None] = mapped_column(comment="Confidence score for set piece roles (0-1 scale)")

    # Advanced performance statistics per 90 minutes
    shots_per_90: Mapped[float | None] = mapped_column(comment="Total shots per 90 minutes")
    key_passes_per_90: Mapped[float | None] = mapped_column(comment="Key passes (passes leading to shots) per 90 minutes")
    tackles_per_90: Mapped[float | None] = mapped_column(comment="Successful tackles per 90 minutes")
    interceptions_per_90: Mapped[float | None] = mapped_column(comment="Interceptions per 90 minutes")
    clearances_per_90: Mapped[float | None] = mapped_column(comment="Clearances per 90 minutes")

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
    model_type: Mapped[str100]  # e.g., "NumpyroPlayerModel", "ExtendedDixonColesMatchPredictor"
    description: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[str100]  # ISO datetime string
    created_by: Mapped[str100]  # Author/creator
    is_active: Mapped[bool] = mapped_column(default=True)
    
    # Relationships
    versions: Mapped[list["ModelVersion"]] = relationship(back_populates="registry")

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
    training_params: Mapped[str | None] = mapped_column(String(1000))  # JSON string of parameters
    feature_set: Mapped[str | None] = mapped_column(String(500))  # Description of features used
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
    registry: Mapped["ModelRegistry"] = relationship(back_populates="versions")
    artifacts: Mapped[list["ModelArtifact"]] = relationship(back_populates="version")
    performance_records: Mapped[list["ModelPerformance"]] = relationship(back_populates="version")
    experiments: Mapped[list["ModelExperiment"]] = relationship(
        back_populates="model_version",
        foreign_keys="ModelExperiment.model_version_id"
    )

    def __str__(self):
        return f"ModelVersion({self.registry.model_name} v{self.version})"


class ModelArtifact(Base):
    """Storage metadata for model artifacts (serialized models, weights, etc.)"""
    __tablename__ = "model_artifact"
    
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    artifact_type: Mapped[str100]  # "model_weights", "full_model", "metadata", "training_state"
    file_path: Mapped[str | None] = mapped_column(String(500))  # Local file path
    s3_path: Mapped[str | None] = mapped_column(String(500))  # S3 URL if using cloud storage
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
        return f"ModelPerformance({self.version} on {self.dataset_type}: MAE={self.mae})"


class ModelExperiment(Base):
    """A/B testing experiments comparing model versions"""
    __tablename__ = "model_experiment"
    
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    experiment_name: Mapped[str100]
    description: Mapped[str | None] = mapped_column(String(500))
    
    # Experiment configuration
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"))
    control_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"))
    traffic_split: Mapped[float] = mapped_column(default=0.5)  # Percentage of traffic for this version
    
    # Experiment timeline
    start_date: Mapped[str100]  # ISO datetime string
    end_date: Mapped[str100 | None]  # ISO datetime string
    status: Mapped[str100]  # "planned", "running", "completed", "stopped"
    
    # Results
    winner_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"))
    confidence_level: Mapped[float | None]  # Statistical confidence in results
    
    # Metadata
    created_by: Mapped[str100]
    notes: Mapped[str | None] = mapped_column(String(1000))
    
    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="experiments",
        foreign_keys=[model_version_id]
    )

    def __str__(self):
        return f"ModelExperiment({self.experiment_name}: {self.status})"

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
    computed_features: Mapped[list["ComputedFeature"]] = relationship(back_populates="feature_definition")

    def __str__(self):
        return f"{self.name}:{self.version} ({self.feature_type})"


class ComputedFeature(Base):
    """Computed feature values for entities (players, teams, fixtures) at specific times."""
    __tablename__ = "computed_feature"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    feature_definition_id: Mapped[int] = mapped_column(ForeignKey("feature_definition.id"))
    feature_definition: Mapped["FeatureDefinition"] = relationship(back_populates="computed_features")
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
    cache_key: Mapped[str100]  # composite key: feature_name:entity_type:entity_id:context
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
    schema_version: Mapped[str100]  # Schema-specific version for tracking DB structure changes
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
    rollback_info: Mapped[str | None] = mapped_column(String(1000))  # JSON string with rollback details
    
    # Compatibility information
    min_app_version: Mapped[str100 | None]  # Minimum app version compatible with this schema
    max_app_version: Mapped[str100 | None]  # Maximum app version compatible with this schema
    compatibility_notes: Mapped[str | None] = mapped_column(String(500))
    
    # Status and validation
    is_active: Mapped[bool] = mapped_column(default=True)  # Whether this version is currently active
    validation_status: Mapped[str100] = mapped_column(default="pending")  # "pending", "validated", "failed"
    validation_errors: Mapped[str | None] = mapped_column(String(1000))  # JSON string of validation errors
    
    # Performance impact tracking
    migration_duration_seconds: Mapped[float | None]  # How long the migration took
    affected_tables: Mapped[str | None] = mapped_column(String(500))  # Comma-separated list of affected table names
    records_migrated: Mapped[int | None]  # Number of records affected by migration
    
    # Relationships with migration history
    migration_history: Mapped[list["MigrationHistory"]] = relationship(back_populates="database_version")

    def __str__(self):
        return f"DatabaseVersion(v{self.version}, schema:{self.schema_version}, applied:{self.applied_at})"


class MigrationHistory(Base):
    """Detailed history of all database migrations for audit and rollback purposes"""
    __tablename__ = "migration_history"
    
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    database_version_id: Mapped[int] = mapped_column(ForeignKey("database_version.id"))
    database_version: Mapped["DatabaseVersion"] = relationship(back_populates="migration_history")
    
    # Migration identification
    migration_name: Mapped[str100]  # Human-readable migration name
    migration_hash: Mapped[str100]  # Unique hash identifying this specific migration
    sequence_number: Mapped[int]  # Order of migration execution within a version
    
    # Execution details
    executed_at: Mapped[str100]  # ISO datetime string
    execution_duration_seconds: Mapped[float]
    executed_by: Mapped[str100]  # User or automated system
    execution_context: Mapped[str100]  # "deployment", "development", "testing", "rollback"
    
    # Migration content and impact
    migration_sql: Mapped[str | None] = mapped_column(String(10000))  # SQL executed (if applicable)
    tables_affected: Mapped[str | None] = mapped_column(String(500))  # Comma-separated table names
    records_before: Mapped[int | None]  # Total records before migration
    records_after: Mapped[int | None]  # Total records after migration
    records_changed: Mapped[int | None]  # Number of records modified
    
    # Success and error tracking
    status: Mapped[str100]  # "success", "failed", "partial", "rolled_back"
    error_message: Mapped[str | None] = mapped_column(String(1000))
    warning_count: Mapped[int] = mapped_column(default=0)
    
    # Rollback information
    rollback_sql: Mapped[str | None] = mapped_column(String(10000))  # SQL to rollback this migration
    rollback_tested: Mapped[bool] = mapped_column(default=False)  # Whether rollback has been tested
    rollback_notes: Mapped[str | None] = mapped_column(String(500))
    
    # Performance and monitoring
    memory_usage_mb: Mapped[float | None]  # Peak memory usage during migration
    cpu_usage_percent: Mapped[float | None]  # Average CPU usage during migration
    lock_conflicts: Mapped[int] = mapped_column(default=0)  # Number of lock conflicts encountered
    
    # Dependencies and prerequisites
    depends_on: Mapped[str | None] = mapped_column(String(500))  # Comma-separated list of prerequisite migrations
    blocks: Mapped[str | None] = mapped_column(String(500))  # Comma-separated list of migrations blocked by this one

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
    compatibility_level: Mapped[str100]  # "full", "limited", "deprecated", "incompatible"
    compatibility_notes: Mapped[str | None] = mapped_column(String(500))
    migration_required: Mapped[bool] = mapped_column(default=False)
    migration_priority: Mapped[str100] = mapped_column(default="normal")  # "critical", "high", "normal", "low"
    
    # Validation and testing
    tested_combinations: Mapped[str | None] = mapped_column(String(1000))  # JSON list of tested version combinations
    known_issues: Mapped[str | None] = mapped_column(String(1000))  # JSON list of known compatibility issues
    workarounds: Mapped[str | None] = mapped_column(String(1000))  # JSON list of available workarounds
    
    # Lifecycle management
    created_at: Mapped[str100]
    updated_at: Mapped[str100]
    deprecated_at: Mapped[str100 | None]
    removed_at: Mapped[str100 | None]
    
    # Performance impact
    performance_impact: Mapped[str100] = mapped_column(default="none")  # "none", "low", "medium", "high"
    performance_notes: Mapped[str | None] = mapped_column(String(500))

    def __str__(self):
        return f"SchemaCompatibility(app:{self.app_version_min}-{self.app_version_max}, schema:{self.schema_version_min}-{self.schema_version_max}, level:{self.compatibility_level})"


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
