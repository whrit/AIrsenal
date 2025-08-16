"""
Test suite for extended PlayerAttributes schema with advanced metrics.

Tests cover:
- New xG metrics fields (xg_per_90, xa_per_90, xgi_per_90)
- Form metrics (form_3_games, form_5_games, form_10_games, momentum)
- Fixture difficulty metrics
- Player role indicators
- Advanced statistics per 90 minutes
- Schema integrity and performance indexes
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from airsenal.framework.schema import Base, Player, PlayerAttributes


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_player(db_session):
    """Create a sample player for testing."""
    player = Player(player_id=1, fpl_api_id=123, name="Test Player")
    db_session.add(player)
    db_session.commit()
    return player


def test_extended_player_attributes_creation(db_session, sample_player):
    """Test creation of PlayerAttributes with all new extended fields."""
    # Create PlayerAttributes with all new fields populated
    attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=10,
        price=95,
        team="ARS",
        position="MID",
        
        # xG metrics
        xg_per_90=0.85,
        xa_per_90=0.42,
        xgi_per_90=1.27,
        
        # Form metrics
        form_3_games=8.5,
        form_5_games=7.2,
        form_10_games=6.8,
        momentum=0.15,
        
        # Fixture difficulty
        next_3_fixture_difficulty=2.8,
        next_5_fixture_difficulty=3.1,
        
        # Player roles
        is_penalty_taker=True,
        is_free_kick_taker=False,
        is_corner_taker=True,
        role_confidence=0.85,
        
        # Advanced stats
        shots_per_90=3.2,
        key_passes_per_90=2.1,
        tackles_per_90=1.8,
        interceptions_per_90=0.9,
        clearances_per_90=0.3,
    )
    
    db_session.add(attrs)
    db_session.commit()
    
    # Retrieve and verify all fields
    retrieved = db_session.query(PlayerAttributes).filter_by(
        player_id=sample_player.player_id
    ).first()
    
    assert retrieved is not None
    
    # Test xG metrics
    assert retrieved.xg_per_90 == 0.85
    assert retrieved.xa_per_90 == 0.42
    assert retrieved.xgi_per_90 == 1.27
    
    # Test form metrics
    assert retrieved.form_3_games == 8.5
    assert retrieved.form_5_games == 7.2
    assert retrieved.form_10_games == 6.8
    assert retrieved.momentum == 0.15
    
    # Test fixture difficulty
    assert retrieved.next_3_fixture_difficulty == 2.8
    assert retrieved.next_5_fixture_difficulty == 3.1
    
    # Test player roles
    assert retrieved.is_penalty_taker is True
    assert retrieved.is_free_kick_taker is False
    assert retrieved.is_corner_taker is True
    assert retrieved.role_confidence == 0.85
    
    # Test advanced stats
    assert retrieved.shots_per_90 == 3.2
    assert retrieved.key_passes_per_90 == 2.1
    assert retrieved.tackles_per_90 == 1.8
    assert retrieved.interceptions_per_90 == 0.9
    assert retrieved.clearances_per_90 == 0.3


def test_nullable_fields_with_none_values(db_session, sample_player):
    """Test that all new nullable fields can be set to None."""
    attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=1,
        price=50,
        team="ARS",
        position="DEF",
        
        # All nullable fields set to None
        xg_per_90=None,
        xa_per_90=None,
        xgi_per_90=None,
        form_3_games=None,
        form_5_games=None,
        form_10_games=None,
        momentum=None,
        next_3_fixture_difficulty=None,
        next_5_fixture_difficulty=None,
        role_confidence=None,
        shots_per_90=None,
        key_passes_per_90=None,
        tackles_per_90=None,
        interceptions_per_90=None,
        clearances_per_90=None,
    )
    
    db_session.add(attrs)
    db_session.commit()
    
    retrieved = db_session.query(PlayerAttributes).filter_by(
        player_id=sample_player.player_id
    ).first()
    
    # Verify all nullable fields are None
    assert retrieved.xg_per_90 is None
    assert retrieved.xa_per_90 is None
    assert retrieved.xgi_per_90 is None
    assert retrieved.form_3_games is None
    assert retrieved.form_5_games is None
    assert retrieved.form_10_games is None
    assert retrieved.momentum is None
    assert retrieved.next_3_fixture_difficulty is None
    assert retrieved.next_5_fixture_difficulty is None
    assert retrieved.role_confidence is None
    assert retrieved.shots_per_90 is None
    assert retrieved.key_passes_per_90 is None
    assert retrieved.tackles_per_90 is None
    assert retrieved.interceptions_per_90 is None
    assert retrieved.clearances_per_90 is None


def test_boolean_fields_default_values(db_session, sample_player):
    """Test that boolean role fields have correct default values."""
    attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=1,
        price=50,
        team="ARS",
        position="DEF",
        # Note: not explicitly setting boolean fields to test defaults
    )
    
    db_session.add(attrs)
    db_session.commit()
    
    retrieved = db_session.query(PlayerAttributes).filter_by(
        player_id=sample_player.player_id
    ).first()
    
    # Test default values for boolean fields
    assert retrieved.is_penalty_taker is False
    assert retrieved.is_free_kick_taker is False
    assert retrieved.is_corner_taker is False


def test_backward_compatibility(db_session, sample_player):
    """Test that existing code creating PlayerAttributes still works."""
    # Create PlayerAttributes using only original fields
    attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=1,
        price=50,
        team="ARS",
        position="DEF",
        chance_of_playing_next_round=100,
        news="Fit and available",
        transfers_in=1500,
        transfers_out=500,
    )
    
    db_session.add(attrs)
    db_session.commit()
    
    retrieved = db_session.query(PlayerAttributes).filter_by(
        player_id=sample_player.player_id
    ).first()
    
    # Verify original fields work
    assert retrieved.price == 50
    assert retrieved.team == "ARS"
    assert retrieved.position == "DEF"
    assert retrieved.chance_of_playing_next_round == 100
    assert retrieved.news == "Fit and available"
    
    # Verify new fields have appropriate default/null values
    assert retrieved.xg_per_90 is None
    assert retrieved.form_3_games is None
    assert retrieved.is_penalty_taker is False


def test_indexes_exist(db_session):
    """Test that the defined indexes exist in the database."""
    # Get the engine from the session
    engine = db_session.bind
    
    # Query the SQLite master table for indexes
    result = engine.execute(text(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='player_attributes'"
    ))
    
    index_names = [row[0] for row in result]
    
    # Check that our custom indexes exist
    expected_indexes = [
        "ix_player_season_gameweek",
        "ix_form_3_games",
        "ix_form_5_games", 
        "ix_xg_per_90",
        "ix_xgi_per_90",
        "ix_next_3_fixture_difficulty",
        "ix_position_penalty_taker",
        "ix_momentum",
    ]
    
    for expected_index in expected_indexes:
        assert expected_index in index_names, f"Index {expected_index} not found"


def test_query_performance_scenarios(db_session, sample_player):
    """Test common query scenarios that would benefit from the new indexes."""
    # Create multiple PlayerAttributes records
    for gw in range(1, 11):
        attrs = PlayerAttributes(
            player_id=sample_player.player_id,
            season="2324",
            gameweek=gw,
            price=50 + gw,
            team="ARS",
            position="MID",
            form_3_games=5.0 + (gw * 0.5),
            xg_per_90=0.5 + (gw * 0.1),
            next_3_fixture_difficulty=2.0 + (gw * 0.2),
            is_penalty_taker=(gw % 2 == 0),
            momentum=gw * 0.1,
        )
        db_session.add(attrs)
    
    db_session.commit()
    
    # Test queries that should benefit from indexes
    
    # 1. Query by form (should use ix_form_3_games)
    high_form_players = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.form_3_games >= 7.0
    ).all()
    assert len(high_form_players) > 0
    
    # 2. Query by xG metrics (should use ix_xg_per_90)
    high_xg_players = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.xg_per_90 >= 1.0
    ).all()
    assert len(high_xg_players) > 0
    
    # 3. Query by position and penalty taker (should use ix_position_penalty_taker)
    penalty_takers = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.position == "MID",
        PlayerAttributes.is_penalty_taker == True
    ).all()
    assert len(penalty_takers) > 0
    
    # 4. Query by player, season, gameweek (should use ix_player_season_gameweek)
    specific_record = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.player_id == sample_player.player_id,
        PlayerAttributes.season == "2324",
        PlayerAttributes.gameweek == 5
    ).first()
    assert specific_record is not None
    assert specific_record.gameweek == 5


def test_data_types_and_constraints(db_session, sample_player):
    """Test that the data types are correctly enforced."""
    # Test with valid data types
    attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=1,
        price=50,
        team="ARS",
        position="DEF",
        xg_per_90=1.5,  # float
        form_3_games=8.0,  # float
        is_penalty_taker=True,  # bool
        momentum=-0.5,  # negative float (valid for momentum)
        role_confidence=0.95,  # float between 0-1
    )
    
    db_session.add(attrs)
    db_session.commit()
    
    retrieved = db_session.query(PlayerAttributes).first()
    
    # Verify types are preserved
    assert isinstance(retrieved.xg_per_90, float)
    assert isinstance(retrieved.form_3_games, float)
    assert isinstance(retrieved.is_penalty_taker, bool)
    assert isinstance(retrieved.momentum, float)
    assert isinstance(retrieved.role_confidence, float)


def test_relationship_integrity(db_session, sample_player):
    """Test that the relationship between Player and PlayerAttributes is maintained."""
    attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=1,
        price=50,
        team="ARS",
        position="DEF",
        xg_per_90=0.8,
        form_3_games=6.5,
    )
    
    db_session.add(attrs)
    db_session.commit()
    
    # Test forward relationship (Player -> PlayerAttributes)
    player = db_session.query(Player).filter_by(player_id=sample_player.player_id).first()
    assert len(player.attributes) == 1
    assert player.attributes[0].xg_per_90 == 0.8
    
    # Test backward relationship (PlayerAttributes -> Player)
    attrs = db_session.query(PlayerAttributes).first()
    assert attrs.player.name == "Test Player"
    assert attrs.player.player_id == sample_player.player_id


def test_multiple_gameweeks_form_progression(db_session, sample_player):
    """Test a realistic scenario of form progression over multiple gameweeks."""
    form_data = [
        (1, 2.0, 2.0, 2.0),    # Early season, low form
        (5, 4.5, 3.5, 3.0),    # Improving
        (10, 8.0, 6.5, 5.5),   # Good form
        (15, 6.0, 7.0, 6.8),   # Slight dip but good overall
        (20, 9.5, 8.5, 7.2),   # Peak form
    ]
    
    for gw, form_3, form_5, form_10 in form_data:
        attrs = PlayerAttributes(
            player_id=sample_player.player_id,
            season="2324",
            gameweek=gw,
            price=50,
            team="ARS",
            position="MID",
            form_3_games=form_3,
            form_5_games=form_5,
            form_10_games=form_10,
            momentum=(form_3 - form_10) / 10,  # Simple momentum calculation
        )
        db_session.add(attrs)
    
    db_session.commit()
    
    # Query for players in excellent recent form
    excellent_form = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.form_3_games >= 8.0,
        PlayerAttributes.momentum > 0
    ).all()
    
    assert len(excellent_form) == 1
    assert excellent_form[0].gameweek == 20
    assert excellent_form[0].form_3_games == 9.5


def test_advanced_stats_realistic_values(db_session, sample_player):
    """Test with realistic values for different player positions."""
    
    # Test a striker's stats
    striker_attrs = PlayerAttributes(
        player_id=sample_player.player_id,
        season="2324",
        gameweek=1,
        price=100,
        team="ARS",
        position="FWD",
        xg_per_90=0.95,  # High xG for striker
        shots_per_90=4.2,  # Many shots
        key_passes_per_90=1.1,  # Fewer key passes
        tackles_per_90=0.3,  # Low defensive stats
        interceptions_per_90=0.2,
        clearances_per_90=0.1,
        is_penalty_taker=True,
    )
    
    db_session.add(striker_attrs)
    db_session.commit()
    
    # Query and verify striker characteristics
    striker = db_session.query(PlayerAttributes).filter_by(position="FWD").first()
    assert striker.xg_per_90 > 0.8  # High xG
    assert striker.shots_per_90 > 3.0  # Many shots
    assert striker.tackles_per_90 < 1.0  # Low defensive activity
    assert striker.is_penalty_taker is True


def test_fixture_difficulty_scenarios(db_session, sample_player):
    """Test fixture difficulty calculations for transfer planning."""
    # Create attributes with varying fixture difficulties
    difficulty_scenarios = [
        (1, 2.0, 2.5),  # Easy fixtures ahead
        (2, 4.5, 4.2),  # Difficult fixtures  
        (3, 3.0, 3.1),  # Average fixtures
    ]
    
    for gw, diff_3, diff_5 in difficulty_scenarios:
        attrs = PlayerAttributes(
            player_id=sample_player.player_id,
            season="2324",
            gameweek=gw,
            price=50,
            team="ARS",
            position="MID",
            next_3_fixture_difficulty=diff_3,
            next_5_fixture_difficulty=diff_5,
        )
        db_session.add(attrs)
    
    db_session.commit()
    
    # Query for players with easy upcoming fixtures (good for transfers)
    easy_fixtures = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.next_3_fixture_difficulty <= 2.5
    ).all()
    
    assert len(easy_fixtures) == 1
    assert easy_fixtures[0].gameweek == 1
    
    # Query for players with difficult fixtures (avoid transfers)
    difficult_fixtures = db_session.query(PlayerAttributes).filter(
        PlayerAttributes.next_3_fixture_difficulty >= 4.0
    ).all()
    
    assert len(difficult_fixtures) == 1
    assert difficult_fixtures[0].gameweek == 2