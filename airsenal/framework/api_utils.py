"""
Functions used by the AIrsenal API
"""

from flask import jsonify
from sqlalchemy.orm import scoped_session

from airsenal.framework.optimization_transfers import (
    make_optimum_double_transfer,
    make_optimum_single_transfer,
)
from airsenal.framework.schema import Player, SessionBudget, SessionSquad, session
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_fixtures_for_player,
    get_last_finished_gameweek,
    get_latest_prediction_tag,
    get_next_fixture_for_player,
    get_player,
    get_predicted_points_for_player,
    get_recent_scores_for_player,
    list_players,
    list_teams,
)

DBSESSION = scoped_session(session)


def remove_db_session(dbsession=DBSESSION):
    dbsession.remove()


def create_response(orig_response):
    """
    Add headers to the response
    """
    response = jsonify(orig_response)
    response.headers.add(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, x-auth",
    )
    return response


def reset_session_squad(session_id, dbsession=DBSESSION):
    """
    remove any rows with the given session ID and add a new budget of
    100M
    """
    # remove all players with this session id
    dbsession.query(SessionSquad).filter_by(session_id=session_id).delete()
    dbsession.commit()
    # now remove the budget, and make a new one
    dbsession.query(SessionBudget).filter_by(session_id=session_id).delete()
    dbsession.commit()
    sb = SessionBudget(session_id=session_id, budget=1000)
    dbsession.add(sb)
    dbsession.commit()
    return True


def list_players_for_api(team, position, dbsession=DBSESSION):
    """
    List players.  Just pass on to utils.list_players but
    specify the dbsession.
    """
    return list_players(team=team, position=position, dbsession=dbsession)


def list_teams_for_api(dbsession=DBSESSION):
    """
    List teams.  Just pass on to utils.list_teams but
    specify the season and  dbsession.
    """
    all_teams = [{"name": "all", "full_name": "all"}]
    all_teams += list_teams(season=CURRENT_SEASON, dbsession=dbsession)
    return all_teams


def combine_player_info(player_id, dbsession=DBSESSION):
    """
    Get player's name, club, recent scores, upcoming fixtures, and
    upcoming predictions if available
    """
    info_dict = {"player_id": player_id}
    p = get_player(player_id, dbsession=dbsession)
    if p is None:
        msg = f"Player with id {player_id} not found"
        raise RuntimeError(msg)
    info_dict["player_name"] = p.name
    team = p.team(CURRENT_SEASON, NEXT_GAMEWEEK)
    info_dict["team"] = team
    # get recent scores for the player
    rs = get_recent_scores_for_player(p, dbsession=dbsession)
    recent_scores = [{"gameweek": k, "score": v} for k, v in rs.items()]
    info_dict["recent_scores"] = recent_scores
    # get upcoming fixtures
    fixtures = get_fixtures_for_player(p, dbsession=dbsession)[:3]
    info_dict["fixtures"] = []
    for f in fixtures:
        home_or_away = "home" if f.home_team == team else "away"
        opponent = f.away_team if home_or_away == "home" else f.home_team
        info_dict["fixtures"].append(
            {"gameweek": f.gameweek, "opponent": opponent, "home_or_away": home_or_away}
        )
    try:
        tag = get_latest_prediction_tag(dbsession=dbsession)
        predicted_points = get_predicted_points_for_player(p, tag, dbsession=dbsession)
        info_dict["predictions"] = predicted_points
    except RuntimeError:
        pass
    return info_dict


def add_session_player(player_id, session_id, dbsession=DBSESSION):
    """
    Add a row in the SessionSquad table.
    """
    pids = [p["id"] for p in get_session_players(session_id, dbsession)]
    if player_id in pids:  # don't add the same player twice!
        return False
    st = SessionSquad(session_id=session_id, player_id=player_id)
    dbsession.add(st)
    dbsession.commit()
    return True


def remove_session_player(player_id, session_id, dbsession=DBSESSION):
    """
    Remove row from SessionSquad table.
    """
    pids = [p["id"] for p in get_session_players(session_id, dbsession)]
    player_id = int(player_id)
    if player_id not in pids:  # player not there
        return False
    (
        dbsession.query(SessionSquad)
        .filter_by(session_id=session_id, player_id=player_id)
        .delete()
    )
    dbsession.commit()
    return True


def list_players_teams_prices(
    position="all", team="all", dbsession=DBSESSION, gameweek=NEXT_GAMEWEEK
):
    """
    Return a list of players, each with their current team and price
    """
    return [
        (
            f"{p.name} "
            f"({p.team(CURRENT_SEASON, NEXT_GAMEWEEK)}): "
            f"{p.price(CURRENT_SEASON, NEXT_GAMEWEEK)}"
        )
        for p in list_players(
            position=position, team=team, dbsession=dbsession, gameweek=gameweek
        )
    ]


def get_session_budget(session_id, dbsession=DBSESSION):
    """
    query the sessionbudget table in the db - there should hopefully
    be one and only one row for this session_id
    """

    sb = dbsession.query(SessionBudget).filter_by(session_id=session_id).all()
    if len(sb) != 1:
        msg = f"{len(sb)}  SessionBudgets for session key {session_id}"
        raise RuntimeError(msg)
    return sb[0].budget


def set_session_budget(budget, session_id, dbsession=DBSESSION):
    """
    delete the existing entry for this session_id in the sessionbudget table,
    then enter a new row
    """
    print("Deleting old budget")
    dbsession.query(SessionBudget).filter_by(session_id=session_id).delete()
    dbsession.commit()
    print(f"Setting budget for {session_id} to {budget}")
    sb = SessionBudget(session_id=session_id, budget=budget)
    dbsession.add(sb)
    dbsession.commit()
    return True


def get_session_players(session_id, dbsession=DBSESSION):
    """
    query the dbsession for the list of players with the requested player_id
    """
    players = dbsession.query(SessionSquad).filter_by(session_id=session_id).all()
    return [
        {
            "id": p.player_id,
            "name": dbsession.query(Player)
            .filter_by(player_id=p.player_id)
            .first()
            .name,
        }
        for p in players
    ]


def validate_session_squad(session_id, dbsession=DBSESSION):
    """
    get the list of player_ids for this session_id, and see if we can
    make a valid 15-player squad out of it
    """
    budget = get_session_budget(session_id, dbsession)

    players = get_session_players(session_id, dbsession)
    if len(players) != 15:
        return False
    t = Squad(budget)
    for p in players:
        added_ok = t.add_player(p["id"], dbsession=dbsession)
        if not added_ok:
            return False
    return True


def fill_session_squad(team_id, session_id, dbsession=DBSESSION):
    """
    Use the FPL API to get list of players in an FPL squad with id=team_id,
    then fill the session squad with these players.
    """
    # first reset the squad
    reset_session_squad(session_id, dbsession)
    # now query the API
    players = fetcher.get_fpl_team_data(get_last_finished_gameweek(), team_id)["picks"]
    player_ids = [p["element"] for p in players]
    for pid in player_ids:
        add_session_player(pid, session_id, dbsession)
    team_history = fetcher.get_fpl_team_history_data()["current"]
    index = (
        get_last_finished_gameweek() - 1
    )  # as gameweek starts counting from 1 but list index starts at 0
    budget = team_history[index]["value"]
    set_session_budget(budget, session_id)
    return player_ids


def get_session_prediction(player_id, gw=None, pred_tag=None, dbsession=DBSESSION):
    """
    Query the fixture and predictedscore tables for a specified player
    """
    if not gw:
        gw = NEXT_GAMEWEEK
    if not pred_tag:
        pred_tag = get_latest_prediction_tag()
    return {
        "predicted_score": get_predicted_points_for_player(
            player_id, pred_tag, CURRENT_SEASON, dbsession
        )[gw],
        "fixture": get_next_fixture_for_player(player_id, CURRENT_SEASON, dbsession),
    }


def get_session_predictions(session_id, dbsession=DBSESSION):
    """
    Query the fixture and predictedscore tables for all
    players in our session squad
    """
    pids = [p["id"] for p in get_session_players(session_id, dbsession)]
    pred_tag = get_latest_prediction_tag()
    gw = NEXT_GAMEWEEK
    return {pid: get_session_prediction(pid, gw, pred_tag, dbsession) for pid in pids}


def best_transfer_suggestions(n_transfer, session_id, dbsession=DBSESSION):
    """
    Use our predicted playerscores to suggest the best transfers.
    """
    n_transfer = int(n_transfer)
    if n_transfer not in range(1, 3):
        msg = "Need to choose 1 or 2 transfers"
        raise RuntimeError(msg)
    if not validate_session_squad(session_id, dbsession):
        msg = "Cannot suggest transfer without complete squad"
        raise RuntimeError(msg)

    budget = get_session_budget(session_id, dbsession)
    players = [p["id"] for p in get_session_players(session_id, dbsession)]
    t = Squad(budget)
    for p in players:
        added_ok = t.add_player(p)
        if not added_ok:
            msg = f"Cannot add player {p}"
            raise RuntimeError(msg)
    pred_tag = get_latest_prediction_tag()
    if n_transfer == 1:
        _, pid_out, pid_in = make_optimum_single_transfer(t, pred_tag)
    elif n_transfer == 2:
        _, pid_out, pid_in = make_optimum_double_transfer(t, pred_tag)
    return {"transfers_out": pid_out, "transfers_in": pid_in}


def get_penalty_takers_for_team(team, season=None, dbsession=DBSESSION):
    """
    Get penalty takers for a specific team using the penalty taker identification system.
    
    Args:
        team: Team name
        season: Season to analyze (defaults to current season)
        dbsession: Database session
        
    Returns:
        List of penalty taker assignments with confidence scores
    """
    from airsenal.framework.player_roles import get_team_penalty_takers
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        penalty_takers = get_team_penalty_takers(team, season)
        return penalty_takers
    except Exception as e:
        # Return empty list if analysis fails
        print(f"Error getting penalty takers for {team}: {e}")
        return []


def get_all_penalty_takers_for_api(season=None, dbsession=DBSESSION):
    """
    Get penalty takers for all teams for API consumption.
    
    Args:
        season: Season to analyze (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary mapping team names to their penalty takers
    """
    from airsenal.framework.player_roles import get_all_penalty_takers
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        all_penalty_takers = get_all_penalty_takers(season)
        return all_penalty_takers
    except Exception as e:
        # Return empty dict if analysis fails
        print(f"Error getting all penalty takers: {e}")
        return {}


def get_penalty_taker_confidence(player_id, season=None, dbsession=DBSESSION):
    """
    Get penalty taker confidence score for a specific player.
    
    Args:
        player_id: Player ID
        season: Season to analyze (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with player penalty taker information
    """
    from airsenal.framework.player_roles import PenaltyTakerAnalyzer
    from airsenal.framework.utils import CURRENT_SEASON
    from airsenal.framework.schema import Player, PlayerAttributes
    from sqlalchemy import and_
    
    if not season:
        season = CURRENT_SEASON
    
    # Get player information
    player = dbsession.query(Player).filter(Player.player_id == player_id).first()
    if not player:
        return {"error": "Player not found"}
    
    # Get player attributes to find team
    attrs = dbsession.query(PlayerAttributes).filter(
        and_(
            PlayerAttributes.player_id == player_id,
            PlayerAttributes.season == season
        )
    ).first()
    
    if not attrs:
        return {"error": "Player attributes not found for season"}
    
    # Get penalty taker information for the team
    analyzer = PenaltyTakerAnalyzer(dbsession)
    team_penalty_takers = analyzer.identify_team_penalty_takers(attrs.team, season)
    
    # Find this player in the results
    for pt in team_penalty_takers:
        if pt['player_id'] == player_id:
            return {
                'player_id': player_id,
                'player_name': player.name,
                'team': attrs.team,
                'is_penalty_taker': True,
                'confidence': pt['confidence'],
                'is_primary': pt['is_primary'],
                'penalties_taken': pt['penalties_taken'],
                'success_rate': pt['success_rate'],
                'last_penalty_date': pt['last_penalty_date'].isoformat() if pt['last_penalty_date'] else None
            }
    
    # Player not identified as penalty taker
    return {
        'player_id': player_id,
        'player_name': player.name,
        'team': attrs.team,
        'is_penalty_taker': False,
        'confidence': 0.0,
        'is_primary': False,
        'penalties_taken': 0,
        'success_rate': 0.0,
        'last_penalty_date': None
    }


def get_fixture_difficulty_for_api(fixture_id, perspective_team=None, season=None, dbsession=DBSESSION):
    """
    Get fixture difficulty rating for API consumption.
    
    Args:
        fixture_id: Fixture ID
        perspective_team: Team to calculate difficulty from perspective of (optional)
        season: Season (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with fixture difficulty data
    """
    from airsenal.framework.fixture_difficulty import get_fixture_difficulty
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        difficulty_data = get_fixture_difficulty(
            fixture_id, perspective_team, season, dbsession
        )
        return difficulty_data
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"Error getting fixture difficulty for fixture {fixture_id}: {e}")
        return {"error": "Failed to calculate fixture difficulty"}


def get_team_fixture_difficulties_for_api(team, gameweek_start, gameweek_end, season=None, dbsession=DBSESSION):
    """
    Get fixture difficulties for a team over a period for API consumption.
    
    Args:
        team: Team name
        gameweek_start: Starting gameweek
        gameweek_end: Ending gameweek (inclusive)
        season: Season (defaults to current season)
        dbsession: Database session
        
    Returns:
        List of fixture difficulty dictionaries
    """
    from airsenal.framework.fixture_difficulty import get_team_fixture_difficulties
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        gameweek_start = int(gameweek_start)
        gameweek_end = int(gameweek_end)
        
        if gameweek_start > gameweek_end:
            return {"error": "gameweek_start must be <= gameweek_end"}
        
        if gameweek_end - gameweek_start > 20:
            return {"error": "Maximum range is 20 gameweeks"}
        
        difficulties = get_team_fixture_difficulties(
            team, gameweek_start, gameweek_end, season, dbsession
        )
        
        return {
            "team": team,
            "season": season,
            "gameweek_start": gameweek_start,
            "gameweek_end": gameweek_end,
            "fixtures": difficulties
        }
        
    except ValueError as e:
        return {"error": f"Invalid gameweek values: {e}"}
    except Exception as e:
        print(f"Error getting team fixture difficulties for {team}: {e}")
        return {"error": "Failed to calculate team fixture difficulties"}


def get_all_teams_fixture_difficulties_for_api(gameweek_start, gameweek_end, season=None, dbsession=DBSESSION):
    """
    Get average fixture difficulties for all teams over a period for API consumption.
    
    Args:
        gameweek_start: Starting gameweek
        gameweek_end: Ending gameweek (inclusive)
        season: Season (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary mapping team names to their average difficulties
    """
    from airsenal.framework.fixture_difficulty import get_all_teams_average_difficulty
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        gameweek_start = int(gameweek_start)
        gameweek_end = int(gameweek_end)
        
        if gameweek_start > gameweek_end:
            return {"error": "gameweek_start must be <= gameweek_end"}
        
        if gameweek_end - gameweek_start > 20:
            return {"error": "Maximum range is 20 gameweeks"}
        
        difficulties = get_all_teams_average_difficulty(
            gameweek_start, gameweek_end, season, dbsession
        )
        
        return {
            "season": season,
            "gameweek_start": gameweek_start,
            "gameweek_end": gameweek_end,
            "teams": difficulties
        }
        
    except ValueError as e:
        return {"error": f"Invalid gameweek values: {e}"}
    except Exception as e:
        print(f"Error getting all teams fixture difficulties: {e}")
        return {"error": "Failed to calculate fixture difficulties"}


def get_gameweek_fixture_difficulties_for_api(gameweek, season=None, dbsession=DBSESSION):
    """
    Get fixture difficulties for all fixtures in a specific gameweek for API consumption.
    
    Args:
        gameweek: Gameweek number
        season: Season (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with all fixtures and their difficulties for the gameweek
    """
    from airsenal.framework.fixture_difficulty import create_difficulty_calculator
    from airsenal.framework.utils import CURRENT_SEASON
    from airsenal.framework.schema import Fixture
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        gameweek = int(gameweek)
        
        # Get all fixtures for the gameweek
        fixtures = (
            dbsession.query(Fixture)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek == gameweek)
            .order_by(Fixture.date, Fixture.fixture_id)
            .all()
        )
        
        if not fixtures:
            return {
                "gameweek": gameweek,
                "season": season,
                "fixtures": [],
                "message": "No fixtures found for this gameweek"
            }
        
        # Calculate difficulties
        calculator = create_difficulty_calculator(season, dbsession)
        fixture_difficulties = []
        
        for fixture in fixtures:
            try:
                difficulty = calculator.calculate_fixture_difficulty(fixture)
                fixture_difficulties.append(difficulty)
            except Exception as e:
                print(f"Error calculating difficulty for fixture {fixture.fixture_id}: {e}")
                continue
        
        return {
            "gameweek": gameweek,
            "season": season,
            "fixture_count": len(fixture_difficulties),
            "fixtures": fixture_difficulties
        }
        
    except ValueError as e:
        return {"error": f"Invalid gameweek: {e}"}
    except Exception as e:
        print(f"Error getting gameweek fixture difficulties: {e}")
        return {"error": "Failed to calculate gameweek fixture difficulties"}


def validate_fixture_difficulty_predictions_for_api(season=None, dbsession=DBSESSION):
    """
    Validate fixture difficulty predictions against actual results for API consumption.
    
    Args:
        season: Season to validate (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with validation metrics
    """
    from airsenal.framework.fixture_difficulty import create_difficulty_calculator
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        calculator = create_difficulty_calculator(season, dbsession)
        validation_results = calculator.validate_predictions(season)
        
        return validation_results
        
    except Exception as e:
        print(f"Error validating fixture difficulty predictions: {e}")
        return {"error": "Failed to validate predictions"}


def get_team_strength_for_api(team, season=None, gameweek=None, dbsession=DBSESSION):
    """
    Get team strength data for API consumption.
    
    Args:
        team: Team name
        season: Season (defaults to current season)
        gameweek: Specific gameweek (defaults to most recent)
        dbsession: Database session
        
    Returns:
        Dictionary with team strength data
    """
    from airsenal.framework.team_strength import get_team_strength_for_api as core_get_team_strength
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        strength_data = core_get_team_strength(team, season, gameweek, dbsession)
        return strength_data
    except Exception as e:
        print(f"Error getting team strength for {team}: {e}")
        return {"error": f"Failed to get team strength for {team}"}


def get_all_team_strengths_for_api(season=None, gameweek=None, dbsession=DBSESSION):
    """
    Get team strengths for all teams for API consumption.
    
    Args:
        season: Season (defaults to current season)
        gameweek: Specific gameweek (defaults to most recent)
        dbsession: Database session
        
    Returns:
        Dictionary mapping team names to their strength data
    """
    from airsenal.framework.team_strength import get_all_team_strengths_for_api as core_get_all_strengths
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        all_strengths = core_get_all_strengths(season, gameweek, dbsession)
        return all_strengths
    except Exception as e:
        print(f"Error getting all team strengths: {e}")
        return {"error": "Failed to get team strengths"}


def update_team_strengths_for_api(season=None, gameweek=None, dbsession=DBSESSION):
    """
    Update team strengths for a specific gameweek for API consumption.
    
    Args:
        season: Season to update (defaults to current season)
        gameweek: Gameweek to update (defaults to last finished gameweek)
        dbsession: Database session
        
    Returns:
        Dictionary with update results
    """
    from airsenal.framework.team_strength import update_team_strengths_for_gameweek
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        update_results = update_team_strengths_for_gameweek(season, gameweek, dbsession)
        return update_results
    except Exception as e:
        print(f"Error updating team strengths: {e}")
        return {"error": f"Failed to update team strengths: {e}"}


def validate_team_strength_model_for_api(season=None, dbsession=DBSESSION):
    """
    Validate team strength model performance for API consumption.
    
    Args:
        season: Season to validate (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with validation metrics
    """
    from airsenal.framework.team_strength import validate_team_strength_model
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        validation_results = validate_team_strength_model(season, dbsession)
        return validation_results
    except Exception as e:
        print(f"Error validating team strength model: {e}")
        return {"error": f"Model validation failed: {e}"}


def compare_team_strengths_for_api(team1, team2, season=None, gameweek=None, dbsession=DBSESSION):
    """
    Compare strengths between two teams for API consumption.
    
    Args:
        team1: First team name
        team2: Second team name
        season: Season (defaults to current season)
        gameweek: Specific gameweek (defaults to most recent)
        dbsession: Database session
        
    Returns:
        Dictionary with team strength comparison
    """
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        # Get strengths for both teams
        team1_strength = get_team_strength_for_api(team1, season, gameweek, dbsession)
        team2_strength = get_team_strength_for_api(team2, season, gameweek, dbsession)
        
        if "error" in team1_strength:
            return {"error": f"Could not get strength for {team1}"}
        if "error" in team2_strength:
            return {"error": f"Could not get strength for {team2}"}
        
        # Calculate head-to-head comparisons
        home_advantage = team1_strength["attacking_strength"]["home"] - team2_strength["defensive_strength"]["away"]
        away_advantage = team2_strength["attacking_strength"]["away"] - team1_strength["defensive_strength"]["home"]
        
        neutral_comparison = (
            (team1_strength["overall_strength"]["home"] + team1_strength["overall_strength"]["away"]) / 2 -
            (team2_strength["overall_strength"]["home"] + team2_strength["overall_strength"]["away"]) / 2
        )
        
        return {
            "comparison": {
                "team1": team1,
                "team2": team2,
                "season": season,
                "gameweek": gameweek
            },
            "head_to_head": {
                "team1_home_advantage": float(home_advantage),
                "team2_away_strength": float(away_advantage),
                "neutral_ground_advantage": float(neutral_comparison),
                "predicted_winner": team1 if neutral_comparison > 0 else team2,
                "confidence": abs(float(neutral_comparison))
            },
            "individual_strengths": {
                team1: team1_strength,
                team2: team2_strength
            }
        }
        
    except Exception as e:
        print(f"Error comparing team strengths: {e}")
        return {"error": f"Failed to compare team strengths: {e}"}


def get_team_strength_trends_for_api(team, season=None, num_gameweeks=10, dbsession=DBSESSION):
    """
    Get team strength trends over time for API consumption.
    
    Args:
        team: Team name
        season: Season (defaults to current season)
        num_gameweeks: Number of recent gameweeks to analyze
        dbsession: Database session
        
    Returns:
        Dictionary with team strength trends
    """
    from airsenal.framework.schema import TeamStrengthHistory
    from airsenal.framework.utils import CURRENT_SEASON
    from sqlalchemy import desc
    
    if not season:
        season = CURRENT_SEASON
    
    try:
        # Get historical strength data
        strength_history = (
            dbsession.query(TeamStrengthHistory)
            .filter(TeamStrengthHistory.team == team)
            .filter(TeamStrengthHistory.season == season)
            .order_by(desc(TeamStrengthHistory.gameweek))
            .limit(num_gameweeks)
            .all()
        )
        
        if not strength_history:
            return {"error": f"No strength history found for {team}"}
        
        # Format trend data
        trends = []
        for strength in reversed(strength_history):  # Chronological order
            trends.append({
                "gameweek": strength.gameweek,
                "attacking_strength_home": strength.attacking_strength_home,
                "attacking_strength_away": strength.attacking_strength_away,
                "defensive_strength_home": strength.defensive_strength_home,
                "defensive_strength_away": strength.defensive_strength_away,
                "overall_strength_home": strength.overall_strength_home,
                "overall_strength_away": strength.overall_strength_away,
                "calculated_at": strength.calculated_at,
                "changes": {
                    "attacking_home": strength.attacking_strength_home_change,
                    "attacking_away": strength.attacking_strength_away_change,
                    "defensive_home": strength.defensive_strength_home_change,
                    "defensive_away": strength.defensive_strength_away_change
                }
            })
        
        # Calculate trend statistics
        if len(trends) > 1:
            # Simple trend calculation (slope of overall strength)
            overall_home_values = [t["overall_strength_home"] for t in trends]
            overall_away_values = [t["overall_strength_away"] for t in trends]
            
            import numpy as np
            from scipy import stats
            
            x = np.arange(len(trends))
            home_trend = stats.linregress(x, overall_home_values)[0] if len(overall_home_values) > 1 else 0
            away_trend = stats.linregress(x, overall_away_values)[0] if len(overall_away_values) > 1 else 0
            
            trend_summary = {
                "home_trend": float(home_trend),
                "away_trend": float(away_trend),
                "overall_trend": float((home_trend + away_trend) / 2),
                "trend_direction": "improving" if (home_trend + away_trend) > 0 else "declining",
                "volatility": float(np.std(overall_home_values + overall_away_values))
            }
        else:
            trend_summary = {
                "home_trend": 0.0,
                "away_trend": 0.0,
                "overall_trend": 0.0,
                "trend_direction": "stable",
                "volatility": 0.0
            }
        
        return {
            "team": team,
            "season": season,
            "gameweeks_analyzed": len(trends),
            "trend_summary": trend_summary,
            "historical_data": trends
        }
        
    except Exception as e:
        print(f"Error getting team strength trends for {team}: {e}")
        return {"error": f"Failed to get strength trends for {team}"}


def get_rotation_risk_for_player_api(player_id, gameweek=None, season=None, dbsession=DBSESSION):
    """
    Get rotation risk prediction for a specific player for API consumption.
    
    Args:
        player_id: Player database ID
        gameweek: Target gameweek (defaults to next gameweek)
        season: Season string (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with rotation risk prediction and factors
    """
    from airsenal.framework.rotation_risk import RotationRiskCalculator
    from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK
    
    if not season:
        season = CURRENT_SEASON
    if not gameweek:
        gameweek = NEXT_GAMEWEEK
        
    try:
        calculator = RotationRiskCalculator(dbsession=dbsession)
        prediction = calculator.calculate_rotation_risk(player_id, gameweek, season)
        
        # Format for API response
        return {
            "player_id": player_id,
            "gameweek": gameweek,
            "season": season,
            "rotation_risk": prediction["rotation_risk"],
            "confidence": prediction["confidence"],
            "risk_level": _get_risk_level_description(prediction["rotation_risk"]),
            "factors": prediction.get("risk_factors", {}),
            "model_version": prediction.get("model_version", "1.0.0"),
            "calculated_at": prediction.get("calculation_time")
        }
        
    except Exception as e:
        print(f"Error calculating rotation risk for player {player_id}: {e}")
        return {
            "error": f"Failed to calculate rotation risk for player {player_id}",
            "player_id": player_id,
            "gameweek": gameweek,
            "season": season
        }


def get_team_rotation_risks_for_api(team, gameweek=None, season=None, dbsession=DBSESSION):
    """
    Get rotation risks for all players in a team for API consumption.
    
    Args:
        team: Team abbreviation (e.g., 'ARS', 'MCI')
        gameweek: Target gameweek (defaults to next gameweek)
        season: Season string (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with rotation risks for all team players
    """
    from airsenal.framework.rotation_risk import get_team_rotation_risks
    from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK
    
    if not season:
        season = CURRENT_SEASON
    if not gameweek:
        gameweek = NEXT_GAMEWEEK
        
    try:
        rotation_risks = get_team_rotation_risks(team, gameweek, season)
        
        # Format risks and add player names
        formatted_risks = []
        for player_id, prediction in rotation_risks.items():
            player = get_player(player_id, dbsession=dbsession)
            player_name = player.name if player else f"Player {player_id}"
            
            formatted_risks.append({
                "player_id": player_id,
                "player_name": player_name,
                "rotation_risk": prediction["rotation_risk"],
                "confidence": prediction["confidence"],
                "risk_level": _get_risk_level_description(prediction["rotation_risk"]),
                "key_factors": _get_key_risk_factors(prediction.get("risk_factors", {}))
            })
        
        # Sort by rotation risk (highest first)
        formatted_risks.sort(key=lambda x: x["rotation_risk"], reverse=True)
        
        return {
            "team": team,
            "gameweek": gameweek,
            "season": season,
            "player_count": len(formatted_risks),
            "avg_rotation_risk": sum(p["rotation_risk"] for p in formatted_risks) / len(formatted_risks) if formatted_risks else 0,
            "high_risk_players": [p for p in formatted_risks if p["rotation_risk"] > 0.6],
            "low_risk_players": [p for p in formatted_risks if p["rotation_risk"] < 0.3],
            "players": formatted_risks
        }
        
    except Exception as e:
        print(f"Error calculating team rotation risks for {team}: {e}")
        return {
            "error": f"Failed to calculate rotation risks for team {team}",
            "team": team,
            "gameweek": gameweek,
            "season": season
        }


def get_high_rotation_risk_players_for_api(gameweek=None, season=None, threshold=0.6, limit=20, dbsession=DBSESSION):
    """
    Get players with highest rotation risk across all teams for API consumption.
    
    Args:
        gameweek: Target gameweek (defaults to next gameweek)
        season: Season string (defaults to current season)
        threshold: Minimum rotation risk threshold (default 0.6)
        limit: Maximum number of players to return (default 20)
        dbsession: Database session
        
    Returns:
        Dictionary with high-risk players across all teams
    """
    from airsenal.framework.rotation_risk import RotationRiskCalculator
    from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK
    
    if not season:
        season = CURRENT_SEASON
    if not gameweek:
        gameweek = NEXT_GAMEWEEK
        
    try:
        calculator = RotationRiskCalculator(dbsession=dbsession)
        
        # Get all active players for the season
        active_players = (dbsession.query(Player)
                         .join(PlayerAttributes)
                         .filter(
                             PlayerAttributes.season == season,
                             PlayerAttributes.gameweek <= gameweek
                         )
                         .distinct()
                         .limit(500)  # Reasonable limit to avoid performance issues
                         .all())
        
        player_ids = [p.player_id for p in active_players]
        
        # Calculate rotation risks for all players
        rotation_risks = calculator.calculate_batch_rotation_risks(player_ids, gameweek, season)
        
        # Filter and format high-risk players
        high_risk_players = []
        for player_id, prediction in rotation_risks.items():
            if prediction["rotation_risk"] >= threshold:
                player = get_player(player_id, dbsession=dbsession)
                if player:
                    # Get current team
                    attrs = player.get_gameweek_attributes(season, gameweek)
                    current_team = attrs.team if attrs and not isinstance(attrs, tuple) else "Unknown"
                    current_position = attrs.position if attrs and not isinstance(attrs, tuple) else "Unknown"
                    
                    high_risk_players.append({
                        "player_id": player_id,
                        "player_name": player.name,
                        "team": current_team,
                        "position": current_position,
                        "rotation_risk": prediction["rotation_risk"],
                        "confidence": prediction["confidence"],
                        "risk_level": _get_risk_level_description(prediction["rotation_risk"]),
                        "key_factors": _get_key_risk_factors(prediction.get("risk_factors", {}))
                    })
        
        # Sort by rotation risk (highest first) and limit results
        high_risk_players.sort(key=lambda x: x["rotation_risk"], reverse=True)
        high_risk_players = high_risk_players[:limit]
        
        return {
            "gameweek": gameweek,
            "season": season,
            "threshold": threshold,
            "player_count": len(high_risk_players),
            "avg_risk": sum(p["rotation_risk"] for p in high_risk_players) / len(high_risk_players) if high_risk_players else 0,
            "players": high_risk_players
        }
        
    except Exception as e:
        print(f"Error getting high rotation risk players: {e}")
        return {
            "error": "Failed to get high rotation risk players",
            "gameweek": gameweek,
            "season": season,
            "threshold": threshold
        }


def get_manager_rotation_patterns_for_api(team, season=None, dbsession=DBSESSION):
    """
    Get manager rotation patterns for a specific team for API consumption.
    
    Args:
        team: Team abbreviation (e.g., 'ARS', 'MCI')
        season: Season string (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with manager rotation patterns and tendencies
    """
    from airsenal.framework.rotation_risk import ManagerPatternAnalyzer
    from airsenal.framework.utils import CURRENT_SEASON
    
    if not season:
        season = CURRENT_SEASON
        
    try:
        analyzer = ManagerPatternAnalyzer(dbsession=dbsession)
        patterns = analyzer.analyze_manager_patterns(team, season)
        
        return {
            "team": team,
            "season": season,
            "rotation_patterns": {
                "avg_rotation_rate": patterns["avg_rotation_rate"],
                "congestion_response": patterns["congestion_response"],
                "position_preferences": patterns["position_preferences"],
                "competition_priorities": patterns["competition_priorities"]
            },
            "behavioral_analysis": {
                "rotation_frequency": _get_rotation_frequency_description(patterns["avg_rotation_rate"]),
                "congestion_strategy": _get_congestion_strategy_description(patterns["congestion_response"]),
                "most_rotated_position": max(patterns["position_preferences"].items(), key=lambda x: x[1])[0],
                "least_rotated_position": min(patterns["position_preferences"].items(), key=lambda x: x[1])[0]
            },
            "sample_size": patterns.get("sample_size", 0)
        }
        
    except Exception as e:
        print(f"Error getting manager patterns for {team}: {e}")
        return {
            "error": f"Failed to get manager patterns for team {team}",
            "team": team,
            "season": season
        }


def get_fixture_congestion_for_api(team, gameweek=None, season=None, dbsession=DBSESSION):
    """
    Get fixture congestion analysis for a team for API consumption.
    
    Args:
        team: Team abbreviation (e.g., 'ARS', 'MCI')
        gameweek: Target gameweek (defaults to next gameweek)
        season: Season string (defaults to current season)
        dbsession: Database session
        
    Returns:
        Dictionary with fixture congestion metrics
    """
    from airsenal.framework.rotation_risk import FixtureCongestionAnalyzer
    from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK
    
    if not season:
        season = CURRENT_SEASON
    if not gameweek:
        gameweek = NEXT_GAMEWEEK
        
    try:
        analyzer = FixtureCongestionAnalyzer(dbsession=dbsession)
        congestion = analyzer.calculate_congestion_score(team, gameweek, season)
        
        return {
            "team": team,
            "gameweek": gameweek,
            "season": season,
            "congestion_metrics": {
                "overall_score": congestion["congestion_score"],
                "fixtures_7_days": congestion["fixtures_7_days"],
                "fixtures_14_days": congestion["fixtures_14_days"],
                "travel_burden": congestion["travel_burden"],
                "recovery_time": congestion["recovery_time"]
            },
            "congestion_level": _get_congestion_level_description(congestion["congestion_score"]),
            "recommendations": _get_congestion_recommendations(congestion)
        }
        
    except Exception as e:
        print(f"Error getting fixture congestion for {team}: {e}")
        return {
            "error": f"Failed to get fixture congestion for team {team}",
            "team": team,
            "gameweek": gameweek,
            "season": season
        }


# Helper functions for API formatting

def _get_risk_level_description(risk_score):
    """Convert rotation risk score to descriptive level."""
    if risk_score >= 0.8:
        return "Very High"
    elif risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.4:
        return "Moderate"
    elif risk_score >= 0.2:
        return "Low"
    else:
        return "Very Low"


def _get_key_risk_factors(factors):
    """Extract and format the top 3 risk factors."""
    if not factors:
        return []
        
    # Sort factors by value (excluding negative importance factors)
    positive_factors = {k: v for k, v in factors.items() 
                       if not k.endswith('_importance') and v > 0.1}
    
    sorted_factors = sorted(positive_factors.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 3 factors with descriptions
    top_factors = []
    for factor, value in sorted_factors[:3]:
        description = _get_factor_description(factor, value)
        top_factors.append({
            "factor": factor,
            "value": value,
            "description": description
        })
    
    return top_factors


def _get_factor_description(factor, value):
    """Get human-readable description for a risk factor."""
    descriptions = {
        "fixture_congestion": f"Fixture congestion ({value:.1%})",
        "fatigue_risk": f"Player fatigue ({value:.1%})",
        "manager_rotation_tendency": f"Manager rotation rate ({value:.1%})",
        "age_factor": f"Age-related risk ({value:.1%})",
        "team_depth": f"Squad depth ({value:.1%})",
        "position_rotation_rate": f"Position rotation rate ({value:.1%})",
        "recovery_time": f"Short recovery time ({value:.1%})"
    }
    return descriptions.get(factor, f"{factor.replace('_', ' ').title()} ({value:.1%})")


def _get_rotation_frequency_description(rate):
    """Convert rotation rate to descriptive text."""
    if rate >= 0.5:
        return "Very High - Frequent rotation"
    elif rate >= 0.3:
        return "High - Regular rotation"
    elif rate >= 0.2:
        return "Moderate - Occasional rotation"
    elif rate >= 0.1:
        return "Low - Minimal rotation"
    else:
        return "Very Low - Rarely rotates"


def _get_congestion_strategy_description(response):
    """Convert congestion response to descriptive text."""
    if response >= 1.5:
        return "Highly Responsive - Significantly increases rotation during congestion"
    elif response >= 1.2:
        return "Responsive - Increases rotation during congestion"
    elif response >= 0.8:
        return "Moderately Responsive - Some increase in rotation during congestion"
    else:
        return "Unresponsive - Rotation not significantly affected by congestion"


def _get_congestion_level_description(score):
    """Convert congestion score to descriptive level."""
    if score >= 0.8:
        return "Extreme"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    elif score >= 0.2:
        return "Low"
    else:
        return "Minimal"


def _get_congestion_recommendations(congestion):
    """Generate recommendations based on congestion metrics."""
    recommendations = []
    
    if congestion["congestion_score"] >= 0.6:
        recommendations.append("Expect increased rotation due to fixture congestion")
    
    if congestion["recovery_time"] <= 3:
        recommendations.append("Short recovery time increases rotation likelihood")
        
    if congestion["fixtures_7_days"] >= 3:
        recommendations.append("Multiple fixtures in short period - high rotation risk")
        
    if congestion["travel_burden"] >= 2.0:
        recommendations.append("Travel burden may affect team selection")
    
    if not recommendations:
        recommendations.append("Congestion levels are manageable - normal team selection expected")
        
    return recommendations
