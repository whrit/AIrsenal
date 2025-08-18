"""
Factory_boy factories for Player and PlayerAttributes models.

Provides realistic test data generation for players and their attributes,
including all extended fields added in Sprint 00.
"""

import random

import factory
from sqlalchemy.orm import Session

from airsenal.framework.schema import Player, PlayerAttributes

from .fpl_data import (
    Position,
    generate_season_gameweek,
    get_random_team,
    get_realistic_player_name,
)


class PlayerFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating Player instances with realistic data."""

    class Meta:
        model = Player
        sqlalchemy_session_persistence = "commit"

    # Auto-increment player_id will be handled by the database
    fpl_api_id = factory.Sequence(
        lambda n: n + 1000
    )  # Start from 1000 to avoid conflicts
    name = factory.LazyFunction(
        lambda: get_realistic_player_name("MID")
    )  # Default to MID

    @factory.post_generation
    def set_position_specific_name(obj, create, extracted, **kwargs):
        """Set name based on position if PlayerAttributes are created."""
        # This will be updated when PlayerAttributes are linked


class PlayerAttributesFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating basic PlayerAttributes (backward compatible)."""

    class Meta:
        model = PlayerAttributes
        sqlalchemy_session_persistence = "commit"

    # Basic required fields
    player = factory.SubFactory(PlayerFactory)
    season = factory.LazyFunction(lambda: generate_season_gameweek()[0])
    gameweek = factory.LazyFunction(lambda: generate_season_gameweek()[1])
    team = factory.LazyFunction(lambda: get_random_team()[0])
    position = factory.fuzzy.FuzzyChoice([pos.value for pos in Position])
    price = factory.LazyAttribute(lambda obj: random.randint(40, 150))

    # Existing optional fields
    chance_of_playing_next_round = factory.fuzzy.FuzzyInteger(0, 100)
    news = factory.Faker("sentence", nb_words=4)
    return_gameweek = factory.Maybe(
        "news", yes_declaration=factory.fuzzy.FuzzyInteger(1, 10), no_declaration=None
    )
    transfers_balance = factory.fuzzy.FuzzyInteger(-1000, 1000)
    selected = factory.fuzzy.FuzzyInteger(0, 100)
    transfers_in = factory.fuzzy.FuzzyInteger(0, 10000)
    transfers_out = factory.fuzzy.FuzzyInteger(0, 5000)


class PlayerAttributesExtendedFactory(PlayerAttributesFactory):
    """
    Factory for creating PlayerAttributes with all extended fields.

    This factory generates realistic data for all 19 new fields added in Sprint 00,
    with position-appropriate statistical distributions.
    """

    # xG metrics - realistic values based on position
    xg_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xg_per_90")
    )
    xa_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xa_per_90")
    )
    xgi_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xgi_per_90")
    )

    # Form metrics - correlated with each other
    form_3_games = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "form_3_games")
    )
    form_5_games = factory.LazyAttribute(
        lambda obj: max(0.0, obj.form_3_games + random.uniform(-1.0, 0.5))
    )
    form_10_games = factory.LazyAttribute(
        lambda obj: max(0.0, obj.form_5_games + random.uniform(-0.8, 0.3))
    )
    momentum = factory.LazyAttribute(
        lambda obj: _calculate_momentum(obj.form_3_games, obj.form_10_games)
    )

    # Fixture difficulty - consistent for both 3 and 5 game horizons
    next_3_fixture_difficulty = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "next_3_fixture_difficulty")
    )
    next_5_fixture_difficulty = factory.LazyAttribute(
        lambda obj: obj.next_3_fixture_difficulty + random.uniform(-0.3, 0.3)
    )

    # Player roles - position dependent probabilities
    is_penalty_taker = factory.LazyAttribute(
        lambda obj: _get_role_assignment(obj.position, "is_penalty_taker")
    )
    is_free_kick_taker = factory.LazyAttribute(
        lambda obj: _get_role_assignment(obj.position, "is_free_kick_taker")
    )
    is_corner_taker = factory.LazyAttribute(
        lambda obj: _get_role_assignment(obj.position, "is_corner_taker")
    )
    role_confidence = factory.LazyAttribute(lambda obj: _calculate_role_confidence(obj))

    # Advanced stats per 90 - position realistic
    shots_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "shots_per_90")
    )
    key_passes_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "key_passes_per_90")
    )
    tackles_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "tackles_per_90")
    )
    interceptions_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "interceptions_per_90")
    )
    clearances_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "clearances_per_90")
    )

    # Update price to be influenced by form
    price = factory.LazyAttribute(
        lambda obj: _calculate_realistic_price(obj.position, obj.form_3_games)
    )

    @factory.post_generation
    def update_player_name(obj, create, extracted, **kwargs):
        """Update the player's name to match their position."""
        if create and obj.player:
            obj.player.name = get_realistic_player_name(obj.position)


class HighFormPlayerAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory for players in excellent form (for testing transfer suggestions)."""

    form_3_games = factory.fuzzy.FuzzyFloat(8.0, 12.0)
    form_5_games = factory.LazyAttribute(
        lambda obj: obj.form_3_games + random.uniform(-0.5, 0.5)
    )
    form_10_games = factory.LazyAttribute(
        lambda obj: obj.form_5_games + random.uniform(-1.0, 0.3)
    )
    momentum = factory.LazyAttribute(
        lambda obj: max(
            0.0, min(1.0, _calculate_momentum(obj.form_3_games, obj.form_10_games))
        )
    )

    # High form players often have good underlying stats
    xg_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xg_per_90") * 1.3
    )
    xgi_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xgi_per_90") * 1.2
    )


class LowFormPlayerAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory for players in poor form (for testing transfer out suggestions)."""

    form_3_games = factory.fuzzy.FuzzyFloat(0.0, 3.0)
    form_5_games = factory.LazyAttribute(
        lambda obj: obj.form_3_games + random.uniform(0.0, 1.0)
    )
    form_10_games = factory.LazyAttribute(
        lambda obj: obj.form_5_games + random.uniform(0.0, 1.5)
    )
    momentum = factory.LazyAttribute(
        lambda obj: min(0.0, _calculate_momentum(obj.form_3_games, obj.form_10_games))
    )

    # Poor form often reflected in underlying stats
    xg_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xg_per_90") * 0.7
    )
    xgi_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xgi_per_90") * 0.8
    )


class PenaltyTakerPlayerAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory for players who are penalty takers (valuable for FPL)."""

    is_penalty_taker = True
    role_confidence = factory.fuzzy.FuzzyFloat(0.7, 0.95)

    # Penalty takers often have good form and attacking stats
    form_3_games = factory.fuzzy.FuzzyFloat(5.0, 10.0)
    xg_per_90 = factory.LazyAttribute(
        lambda obj: _get_position_stat(obj.position, "xg_per_90") * 1.1
    )


class EasyFixturesPlayerAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory for players with easy upcoming fixtures."""

    next_3_fixture_difficulty = factory.fuzzy.FuzzyFloat(1.0, 2.5)
    next_5_fixture_difficulty = factory.LazyAttribute(
        lambda obj: obj.next_3_fixture_difficulty + random.uniform(-0.2, 0.3)
    )


class DifficultFixturesPlayerAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory for players with difficult upcoming fixtures."""

    next_3_fixture_difficulty = factory.fuzzy.FuzzyFloat(4.0, 5.0)
    next_5_fixture_difficulty = factory.LazyAttribute(
        lambda obj: obj.next_3_fixture_difficulty + random.uniform(-0.3, 0.2)
    )


class GoalkeeperAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory specifically for goalkeepers with position-appropriate stats."""

    position = "GK"

    # Goalkeepers have very low attacking stats
    xg_per_90 = factory.fuzzy.FuzzyFloat(0.0, 0.05)
    xa_per_90 = factory.fuzzy.FuzzyFloat(0.0, 0.02)
    shots_per_90 = factory.fuzzy.FuzzyFloat(0.0, 0.1)

    # But high distribution stats
    key_passes_per_90 = factory.fuzzy.FuzzyFloat(8.0, 20.0)
    clearances_per_90 = factory.fuzzy.FuzzyFloat(2.0, 8.0)

    # Rarely take set pieces
    is_penalty_taker = False
    is_free_kick_taker = factory.fuzzy.FuzzyChoice([True, False], weights=[5, 95])
    is_corner_taker = False


class DefenderAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory specifically for defenders."""

    position = "DEF"

    # Defenders have moderate attacking threat
    xg_per_90 = factory.fuzzy.FuzzyFloat(0.02, 0.25)
    shots_per_90 = factory.fuzzy.FuzzyFloat(0.3, 2.0)

    # High defensive stats
    tackles_per_90 = factory.fuzzy.FuzzyFloat(1.0, 4.0)
    interceptions_per_90 = factory.fuzzy.FuzzyFloat(1.0, 3.5)
    clearances_per_90 = factory.fuzzy.FuzzyFloat(1.5, 6.0)


class MidfielderAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory specifically for midfielders."""

    position = "MID"

    # Balanced stats with good creativity
    xa_per_90 = factory.fuzzy.FuzzyFloat(0.1, 0.8)
    key_passes_per_90 = factory.fuzzy.FuzzyFloat(1.5, 6.0)

    # Most likely to have set piece roles
    is_free_kick_taker = factory.fuzzy.FuzzyChoice([True, False], weights=[60, 40])
    is_corner_taker = factory.fuzzy.FuzzyChoice([True, False], weights=[50, 50])


class ForwardAttributesFactory(PlayerAttributesExtendedFactory):
    """Factory specifically for forwards."""

    position = "FWD"

    # High attacking threat
    xg_per_90 = factory.fuzzy.FuzzyFloat(0.2, 1.0)
    shots_per_90 = factory.fuzzy.FuzzyFloat(2.0, 6.0)

    # Low defensive contribution
    tackles_per_90 = factory.fuzzy.FuzzyFloat(0.0, 1.5)
    interceptions_per_90 = factory.fuzzy.FuzzyFloat(0.0, 1.0)
    clearances_per_90 = factory.fuzzy.FuzzyFloat(0.0, 0.8)

    # Often penalty takers
    is_penalty_taker = factory.fuzzy.FuzzyChoice([True, False], weights=[35, 65])


# Helper functions for realistic data generation
def _get_position_stat(position: str, stat_name: str) -> float:
    """Get a realistic statistical value for a position."""
    from .fpl_data import STATISTICAL_DISTRIBUTIONS

    if position not in STATISTICAL_DISTRIBUTIONS:
        position = "MID"  # Default fallback

    if stat_name not in STATISTICAL_DISTRIBUTIONS[position]:
        return 0.0

    distribution = STATISTICAL_DISTRIBUTIONS[position][stat_name]
    return distribution.generate()


def _get_role_assignment(position: str, role: str) -> bool:
    """Determine if a player should have a specific role based on position."""
    from .fpl_data import ROLE_PROBABILITIES

    if position not in ROLE_PROBABILITIES:
        position = "MID"

    probability = ROLE_PROBABILITIES[position].get(role, 0.0)
    return random.random() < probability


def _calculate_momentum(form_3: float, form_10: float) -> float:
    """Calculate momentum based on short vs long term form."""
    if form_10 == 0:
        return 0.0

    # Momentum is the difference between recent and long-term form
    momentum = (form_3 - form_10) / 10.0  # Normalize to -1 to 1 range
    return max(-1.0, min(1.0, momentum))


def _calculate_role_confidence(obj) -> float:
    """Calculate role confidence based on assigned roles and form."""
    role_count = sum(
        [
            getattr(obj, "is_penalty_taker", False),
            getattr(obj, "is_free_kick_taker", False),
            getattr(obj, "is_corner_taker", False),
        ]
    )

    if role_count == 0:
        return random.uniform(0.0, 0.3)

    # Base confidence increases with number of roles
    base_confidence = 0.4 + (role_count * 0.2)

    # Adjust based on form if available
    form_adjustment = 0.0
    if hasattr(obj, "form_3_games") and obj.form_3_games is not None:
        if obj.form_3_games > 6.0:
            form_adjustment = 0.1
        elif obj.form_3_games < 3.0:
            form_adjustment = -0.1

    confidence = base_confidence + form_adjustment + random.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, round(confidence, 2)))


def _calculate_realistic_price(position: str, form_3_games: float | None = None) -> int:
    """Calculate realistic price based on position and form."""
    from .fpl_data import generate_realistic_price

    return generate_realistic_price(position, form_3_games)


# Batch creation functions for testing scenarios
def create_squad_players(session: Session, num_players: int = 15) -> list[Player]:
    """Create a full FPL squad of 15 players with realistic distribution."""
    players = []

    # Standard FPL squad formation: 2 GK, 5 DEF, 5 MID, 3 FWD
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3

    for _i, position in enumerate(positions):
        factory_class = {
            "GK": GoalkeeperAttributesFactory,
            "DEF": DefenderAttributesFactory,
            "MID": MidfielderAttributesFactory,
            "FWD": ForwardAttributesFactory,
        }[position]

        attrs = factory_class(session=session)
        players.append(attrs.player)

    return players


def create_gameweek_players(
    session: Session, season: str, gameweek: int, num_players: int = 50
) -> list[PlayerAttributes]:
    """Create multiple players for a specific gameweek."""
    players = []

    for _ in range(num_players):
        position = random.choice(["GK", "DEF", "MID", "FWD"])
        attrs = PlayerAttributesExtendedFactory(
            session=session, season=season, gameweek=gameweek, position=position
        )
        players.append(attrs)

    return players


def create_form_comparison_players(session: Session) -> dict:
    """Create players for testing form-based comparisons."""
    return {
        "high_form": [
            HighFormPlayerAttributesFactory(session=session) for _ in range(5)
        ],
        "low_form": [LowFormPlayerAttributesFactory(session=session) for _ in range(5)],
        "penalty_takers": [
            PenaltyTakerPlayerAttributesFactory(session=session) for _ in range(3)
        ],
        "easy_fixtures": [
            EasyFixturesPlayerAttributesFactory(session=session) for _ in range(5)
        ],
        "difficult_fixtures": [
            DifficultFixturesPlayerAttributesFactory(session=session) for _ in range(5)
        ],
    }


def create_historical_progression(
    session: Session,
    player: Player,
    start_gameweek: int = 1,
    num_gameweeks: int = 10,
    season: str = "2324",
) -> list[PlayerAttributes]:
    """Create a progression of PlayerAttributes for a single player over multiple gameweeks."""
    from .fpl_data import generate_form_progression

    # Generate realistic form progression
    base_form = random.uniform(3.0, 8.0)
    form_progression = generate_form_progression(base_form, num_gameweeks)

    attributes = []
    for i in range(num_gameweeks):
        gameweek = start_gameweek + i
        form_3 = form_progression[i]

        attrs = PlayerAttributesExtendedFactory(
            session=session,
            player=player,
            season=season,
            gameweek=gameweek,
            form_3_games=form_3,
            form_5_games=max(0.0, form_3 + random.uniform(-1.0, 0.5)),
            form_10_games=max(0.0, form_3 + random.uniform(-1.5, 0.3)),
        )
        attributes.append(attrs)

    return attributes
