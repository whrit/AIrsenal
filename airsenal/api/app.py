"""
API for calling airsenal functions.
HTTP requests to the endpoints defined here will give rise
to calls to functions in api_utils.py
"""

import json
from uuid import uuid4

from flask import Blueprint, Flask, jsonify, request, session
from flask_cors import CORS
from flask_session import Session

from airsenal.api.exceptions import ApiException
from airsenal.framework.api_utils import (
    add_session_player,
    best_transfer_suggestions,
    combine_player_info,
    compare_team_strengths_for_api,
    create_response,
    fill_session_squad,
    get_all_penalty_takers_for_api,
    get_all_team_strengths_for_api,
    get_all_teams_fixture_difficulties_for_api,
    get_fixture_congestion_for_api,
    get_fixture_difficulty_for_api,
    get_gameweek_fixture_difficulties_for_api,
    get_high_rotation_risk_players_for_api,
    get_manager_rotation_patterns_for_api,
    get_penalty_taker_confidence,
    get_penalty_takers_for_team,
    get_rotation_risk_for_player_api,
    get_session_budget,
    get_session_players,
    get_session_predictions,
    get_team_fixture_difficulties_for_api,
    get_team_rotation_risks_for_api,
    get_team_strength_for_api,
    get_team_strength_trends_for_api,
    list_players_for_api,
    list_teams_for_api,
    remove_db_session,
    remove_session_player,
    set_session_budget,
    update_team_strengths_for_api,
    validate_fixture_difficulty_predictions_for_api,
    validate_session_squad,
    validate_team_strength_model_for_api,
)


def get_session_id():
    """
    Get the ID from the flask_session Session instance if
    it exists, otherwise just get a default string, which
    will enable us to test some functionality just via python requests.
    """
    print(f"Session keys {session.keys()}")
    if "key" in session:
        return session["key"]
    return "DEFAULT_SESSION_ID"


# Use a flask blueprint rather than creating the app directly
# so that we can also make a test app

blueprint = Blueprint("airsenal", __name__)


@blueprint.errorhandler(ApiException)
def handle_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@blueprint.teardown_request
def remove_session(ex=None):  # noqa: ARG001
    remove_db_session()


@blueprint.route("/teams", methods=["GET"])
def get_team_list():
    """
    Return a list of all teams for the current season
    """
    team_list = list_teams_for_api()
    return create_response(team_list)


@blueprint.route("/players/<team>/<pos>", methods=["GET"])
def get_player_list(team, pos):
    """
    Return a list of all players in that team and/or position
    """
    player_list = [
        {"id": p.player_id, "name": p.name}
        for p in list_players_for_api(position=pos, team=team)
    ]
    return create_response(player_list)


@blueprint.route("/new")
def set_session_key():
    """
    Create a new and unique session ID
    """
    key = str(uuid4())
    session["key"] = key
    return create_response(key)


def reset_session_team(param):
    pass


@blueprint.route("/team/new")
def reset_team():
    """
    Remove all players from the DB table with this session_id and
    reset the budget to 100M
    """
    reset_session_team(get_session_id())
    return create_response("OK")


@blueprint.route("/player/<player_id>")
def get_player_info(player_id):
    """
    Return a dict containing player's name, team, recent points,
    and upcoming fixtures and predictions.
    """
    player_info = combine_player_info(player_id)
    return create_response(player_info)


@blueprint.route("/team/add/<player_id>")
def add_player(player_id):
    """
    Add a selected player to this session's squad.
    """
    added_ok = add_session_player(player_id, session_id=get_session_id())
    return create_response(added_ok)


@blueprint.route("/team/remove/<player_id>")
def remove_player(player_id):
    """
    Remove selected player to this session's squad.
    """
    removed_ok = remove_session_player(player_id, session_id=get_session_id())
    return create_response(removed_ok)


@blueprint.route("/team/list", methods=["GET"])
def list_session_players():
    """
    List all players currently in this session's squad.
    """
    player_list = get_session_players(session_id=get_session_id())
    return create_response(player_list)


@blueprint.route("/team/pred", methods=["GET"])
def list_session_predictions():
    """
    Get predicted points for all players in this sessions squad
    """
    pred_dict = get_session_predictions(session_id=get_session_id())
    return create_response(pred_dict)


@blueprint.route("/team/validate", methods=["GET"])
def validate_session_players():
    """
    Check that the squad has 15 players, and obeys constraints.
    """
    valid = validate_session_squad(session_id=get_session_id())
    return create_response(valid)


@blueprint.route("/team/fill/<team_id>")
def fill_team_from_team_id(team_id):
    """
    Use the ID of a team in the FPL API to fill a squad for this session.
    """
    player_ids = fill_session_squad(team_id=team_id, session_id=get_session_id())
    return create_response(player_ids)


@blueprint.route("/team/optimize/<n_transfers>")
def get_optimum_transfers(n_transfers):
    """
    Find the best n_transfers transfers for the next gameweek.
    """
    transfers = best_transfer_suggestions(n_transfers, session_id=get_session_id())
    return create_response(transfers)


@blueprint.route("/budget", methods=["GET", "POST"])
def session_budget():
    """
    Set or get the budget for this team.
    """
    if request.method != "POST":
        return create_response(get_session_budget(get_session_id()))

    data = json.loads(request.data.decode("utf-8"))
    budget = data["budget"]
    set_session_budget(budget, get_session_id())
    return create_response("OK")


@blueprint.route("/penalty_takers", methods=["GET"])
def get_all_penalty_takers():
    """
    Get penalty takers for all teams in the current season.
    
    Returns:
        Dictionary mapping team names to their penalty takers with confidence scores
    """
    penalty_takers = get_all_penalty_takers_for_api()
    return create_response(penalty_takers)


@blueprint.route("/penalty_takers/<team>", methods=["GET"])
def get_team_penalty_takers(team):
    """
    Get penalty takers for a specific team.
    
    Args:
        team: Team name (3-letter code or full name)
        
    Returns:
        List of penalty taker assignments for the team
    """
    season = request.args.get('season')  # Optional season parameter
    penalty_takers = get_penalty_takers_for_team(team, season)
    return create_response(penalty_takers)


@blueprint.route("/player/<player_id>/penalty_info", methods=["GET"])
def get_player_penalty_info(player_id):
    """
    Get penalty taker information for a specific player.
    
    Args:
        player_id: Player ID
        
    Returns:
        Dictionary with penalty taker confidence and statistics
    """
    season = request.args.get('season')  # Optional season parameter
    penalty_info = get_penalty_taker_confidence(int(player_id), season)
    return create_response(penalty_info)


@blueprint.route("/fixture/<fixture_id>/difficulty", methods=["GET"])
def get_fixture_difficulty(fixture_id):
    """
    Get fixture difficulty rating for a specific fixture.
    
    Args:
        fixture_id: Fixture ID
        
    Query parameters:
        team: Calculate difficulty from this team's perspective (optional)
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with fixture difficulty rating and component breakdown
    """
    perspective_team = request.args.get('team')
    season = request.args.get('season')
    
    difficulty_data = get_fixture_difficulty_for_api(
        int(fixture_id), perspective_team, season
    )
    return create_response(difficulty_data)


@blueprint.route("/team/<team>/fixture_difficulties", methods=["GET"])
def get_team_fixture_difficulties(team):
    """
    Get fixture difficulties for a team over a period.
    
    Args:
        team: Team name (3-letter code or full name)
        
    Query parameters:
        gameweek_start: Starting gameweek (required)
        gameweek_end: Ending gameweek (required)
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with team's fixture difficulties
    """
    gameweek_start = request.args.get('gameweek_start')
    gameweek_end = request.args.get('gameweek_end')
    season = request.args.get('season')
    
    if not gameweek_start or not gameweek_end:
        return create_response({
            "error": "gameweek_start and gameweek_end parameters are required"
        })
    
    difficulties = get_team_fixture_difficulties_for_api(
        team, gameweek_start, gameweek_end, season
    )
    return create_response(difficulties)


@blueprint.route("/fixture_difficulties/all_teams", methods=["GET"])
def get_all_teams_fixture_difficulties():
    """
    Get average fixture difficulties for all teams over a period.
    
    Query parameters:
        gameweek_start: Starting gameweek (required)
        gameweek_end: Ending gameweek (required)
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary mapping team names to their average difficulties
    """
    gameweek_start = request.args.get('gameweek_start')
    gameweek_end = request.args.get('gameweek_end')
    season = request.args.get('season')
    
    if not gameweek_start or not gameweek_end:
        return create_response({
            "error": "gameweek_start and gameweek_end parameters are required"
        })
    
    difficulties = get_all_teams_fixture_difficulties_for_api(
        gameweek_start, gameweek_end, season
    )
    return create_response(difficulties)


@blueprint.route("/gameweek/<gameweek>/fixture_difficulties", methods=["GET"])
def get_gameweek_fixture_difficulties(gameweek):
    """
    Get fixture difficulties for all fixtures in a specific gameweek.
    
    Args:
        gameweek: Gameweek number
        
    Query parameters:
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with all fixtures and their difficulties for the gameweek
    """
    season = request.args.get('season')
    
    difficulties = get_gameweek_fixture_difficulties_for_api(gameweek, season)
    return create_response(difficulties)


@blueprint.route("/fixture_difficulties/validate", methods=["GET"])
def validate_fixture_difficulty_predictions():
    """
    Validate fixture difficulty predictions against actual results.
    
    Query parameters:
        season: Season to validate (optional, defaults to current season)
        
    Returns:
        Dictionary with validation metrics including accuracy and correlation
    """
    season = request.args.get('season')
    
    validation_results = validate_fixture_difficulty_predictions_for_api(season)
    return create_response(validation_results)


@blueprint.route("/team_strength/<team>", methods=["GET"])
def get_team_strength(team):
    """
    Get comprehensive team strength analysis for a specific team.
    
    Args:
        team: Team name (3-letter code or full name)
        
    Query parameters:
        season: Season (optional, defaults to current season)
        gameweek: Specific gameweek (optional, defaults to most recent)
        
    Returns:
        Dictionary with comprehensive team strength data including:
        - Attacking and defensive strengths (home/away)
        - Uncertainty quantification
        - Expected goals metrics
        - Form indicators
        - Model metadata
    """
    season = request.args.get('season')
    gameweek = request.args.get('gameweek')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    strength_data = get_team_strength_for_api(team, season, gameweek)
    return create_response(strength_data)


@blueprint.route("/team_strength", methods=["GET"])
def get_all_team_strengths():
    """
    Get team strengths for all teams in the league.
    
    Query parameters:
        season: Season (optional, defaults to current season)
        gameweek: Specific gameweek (optional, defaults to most recent)
        
    Returns:
        Dictionary mapping team names to their strength data
    """
    season = request.args.get('season')
    gameweek = request.args.get('gameweek')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    all_strengths = get_all_team_strengths_for_api(season, gameweek)
    return create_response(all_strengths)


@blueprint.route("/team_strength/compare/<team1>/<team2>", methods=["GET"])
def compare_team_strengths(team1, team2):
    """
    Compare strengths between two teams with head-to-head analysis.
    
    Args:
        team1: First team name
        team2: Second team name
        
    Query parameters:
        season: Season (optional, defaults to current season)
        gameweek: Specific gameweek (optional, defaults to most recent)
        
    Returns:
        Dictionary with detailed team strength comparison including:
        - Head-to-head advantage calculations
        - Venue-specific comparisons
        - Predicted match outcomes
        - Individual team strength profiles
    """
    season = request.args.get('season')
    gameweek = request.args.get('gameweek')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    comparison_data = compare_team_strengths_for_api(team1, team2, season, gameweek)
    return create_response(comparison_data)


@blueprint.route("/team_strength/<team>/trends", methods=["GET"])
def get_team_strength_trends(team):
    """
    Get historical team strength trends and trajectory analysis.
    
    Args:
        team: Team name (3-letter code or full name)
        
    Query parameters:
        season: Season (optional, defaults to current season)
        num_gameweeks: Number of recent gameweeks to analyze (optional, default 10)
        
    Returns:
        Dictionary with team strength trends including:
        - Historical strength values over time
        - Trend direction and volatility
        - Change indicators
        - Performance trajectory
    """
    season = request.args.get('season')
    num_gameweeks = request.args.get('num_gameweeks', '10')
    
    try:
        num_gameweeks = int(num_gameweeks)
        if num_gameweeks < 1 or num_gameweeks > 38:
            return create_response({"error": "num_gameweeks must be between 1 and 38"})
    except ValueError:
        return create_response({"error": "Invalid num_gameweeks parameter"})
    
    trends_data = get_team_strength_trends_for_api(team, season, num_gameweeks)
    return create_response(trends_data)


@blueprint.route("/team_strength/update", methods=["POST"])
def update_team_strengths():
    """
    Update team strengths for all teams in the current gameweek.
    
    This endpoint triggers a full recalculation of team strengths using
    the latest match data and Bayesian inference.
    
    Query parameters:
        season: Season to update (optional, defaults to current season)
        gameweek: Gameweek to update (optional, defaults to last finished gameweek)
        
    Returns:
        Dictionary with update results including teams updated and timing
    """
    season = request.args.get('season')
    gameweek = request.args.get('gameweek')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    update_results = update_team_strengths_for_api(season, gameweek)
    return create_response(update_results)


@blueprint.route("/team_strength/validate", methods=["GET"])
def validate_team_strength_model():
    """
    Validate the team strength model performance against actual results.
    
    This endpoint checks whether the team strength ratings correlate with
    actual match outcomes to ensure model accuracy meets requirements.
    
    Query parameters:
        season: Season to validate (optional, defaults to current season)
        
    Returns:
        Dictionary with validation metrics including:
        - Pearson and Spearman correlations
        - Prediction accuracy
        - Mean absolute error and RMSE
        - Compliance with >0.7 correlation target
    """
    season = request.args.get('season')
    
    validation_results = validate_team_strength_model_for_api(season)
    return create_response(validation_results)


# Rotation Risk Endpoints

@blueprint.route("/rotation_risk/player/<int:player_id>", methods=["GET"])
def get_player_rotation_risk(player_id):
    """
    Get rotation risk prediction for a specific player.
    
    Args:
        player_id: Player database ID
        
    Query parameters:
        gameweek: Target gameweek (optional, defaults to next gameweek)
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with rotation risk prediction including:
        - Overall rotation risk score (0-1)
        - Confidence level
        - Risk factors breakdown
        - Risk level description
    """
    gameweek = request.args.get('gameweek')
    season = request.args.get('season')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    rotation_risk = get_rotation_risk_for_player_api(player_id, gameweek, season)
    return create_response(rotation_risk)


@blueprint.route("/rotation_risk/team/<team>", methods=["GET"])
def get_team_rotation_risks(team):
    """
    Get rotation risks for all players in a team.
    
    Args:
        team: Team abbreviation (e.g., 'ARS', 'MCI')
        
    Query parameters:
        gameweek: Target gameweek (optional, defaults to next gameweek)
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with team rotation risks including:
        - All players with their rotation risks
        - Team averages and summaries
        - High-risk and low-risk player lists
    """
    gameweek = request.args.get('gameweek')
    season = request.args.get('season')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    team_risks = get_team_rotation_risks_for_api(team, gameweek, season)
    return create_response(team_risks)


@blueprint.route("/rotation_risk/high_risk", methods=["GET"])
def get_high_rotation_risk_players():
    """
    Get players with highest rotation risk across all teams.
    
    Query parameters:
        gameweek: Target gameweek (optional, defaults to next gameweek)
        season: Season (optional, defaults to current season)
        threshold: Minimum rotation risk threshold (optional, default 0.6)
        limit: Maximum number of players to return (optional, default 20)
        
    Returns:
        Dictionary with high-risk players including:
        - Players exceeding rotation risk threshold
        - Risk factors for each player
        - Team and position information
    """
    gameweek = request.args.get('gameweek')
    season = request.args.get('season')
    threshold = request.args.get('threshold', '0.6')
    limit = request.args.get('limit', '20')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    try:
        threshold = float(threshold)
        limit = int(limit)
        if threshold < 0 or threshold > 1:
            return create_response({"error": "Threshold must be between 0 and 1"})
        if limit < 1 or limit > 100:
            return create_response({"error": "Limit must be between 1 and 100"})
    except ValueError:
        return create_response({"error": "Invalid threshold or limit parameter"})
    
    high_risk_players = get_high_rotation_risk_players_for_api(gameweek, season, threshold, limit)
    return create_response(high_risk_players)


@blueprint.route("/rotation_risk/manager_patterns/<team>", methods=["GET"])
def get_manager_rotation_patterns(team):
    """
    Get manager rotation patterns and behavioral analysis for a team.
    
    Args:
        team: Team abbreviation (e.g., 'ARS', 'MCI')
        
    Query parameters:
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with manager patterns including:
        - Overall rotation tendencies
        - Position-specific rotation rates
        - Competition priorities
        - Behavioral analysis and descriptions
    """
    season = request.args.get('season')
    
    manager_patterns = get_manager_rotation_patterns_for_api(team, season)
    return create_response(manager_patterns)


@blueprint.route("/rotation_risk/fixture_congestion/<team>", methods=["GET"])
def get_fixture_congestion(team):
    """
    Get fixture congestion analysis for a team.
    
    Args:
        team: Team abbreviation (e.g., 'ARS', 'MCI')
        
    Query parameters:
        gameweek: Target gameweek (optional, defaults to next gameweek)
        season: Season (optional, defaults to current season)
        
    Returns:
        Dictionary with congestion analysis including:
        - Overall congestion score
        - Fixture counts in various time windows
        - Travel burden analysis
        - Recovery time metrics
        - Recommendations based on congestion level
    """
    gameweek = request.args.get('gameweek')
    season = request.args.get('season')
    
    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            return create_response({"error": "Invalid gameweek parameter"})
    
    congestion_data = get_fixture_congestion_for_api(team, gameweek, season)
    return create_response(congestion_data)


def create_app(name=__name__):
    app = Flask(name)
    app.config["SESSION_TYPE"] = "filesystem"
    app.secret_key = "blah"
    CORS(app, supports_credentials=True)
    app.register_blueprint(blueprint)
    Session(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5002, debug=True)
