"""
Realistic FPL player names, teams, and statistical data for test fixtures.

This module provides realistic data based on actual Premier League players and teams
to create more meaningful test scenarios that mirror real-world FPL data.
"""

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Premier League teams for 2024/25 season with abbreviations
PREMIER_LEAGUE_TEAMS = {
    "ARS": "Arsenal",
    "AVL": "Aston Villa", 
    "BOU": "AFC Bournemouth",
    "BRE": "Brentford",
    "BHA": "Brighton & Hove Albion",
    "CHE": "Chelsea",
    "CRY": "Crystal Palace",
    "EVE": "Everton",
    "FUL": "Fulham",
    "IPS": "Ipswich Town",
    "LEI": "Leicester City",
    "LIV": "Liverpool",
    "MCI": "Manchester City",
    "MUN": "Manchester United",
    "NEW": "Newcastle United",
    "NFO": "Nottingham Forest",
    "SOU": "Southampton",
    "TOT": "Tottenham Hotspur",
    "WHU": "West Ham United",
    "WOL": "Wolverhampton Wanderers",
}

# Realistic player names categorized by position
REALISTIC_PLAYERS = {
    "GK": [
        "Aaron Ramsdale", "Alisson Becker", "Andre Onana", "Bernd Leno",
        "David Raya", "Dean Henderson", "Ederson", "Emiliano Martinez",
        "Fraser Forster", "Guglielmo Vicario", "James Trafford", "Jason Steele",
        "Jordan Pickford", "Jose Sa", "Kepa Arrizabalaga", "Lukasz Fabianski",
        "Mark Flekken", "Matz Sels", "Nick Pope", "Neto Santos",
        "Robert Sanchez", "Sam Johnstone", "Thiago Silva", "Thomas Kaminski",
        "Tim Krul", "Vicente Guaita", "Wes Foderingham", "Alphonse Areola",
    ],
    "DEF": [
        # Arsenal
        "William Saliba", "Gabriel Magalhaes", "Ben White", "Jurrien Timber",
        "Takehiro Tomiyasu", "Kieran Tierney", "Oleksandr Zinchenko",
        # Liverpool
        "Virgil van Dijk", "Ibrahima Konate", "Joe Gomez", "Andrew Robertson",
        "Trent Alexander-Arnold", "Kostas Tsimikas", "Jarell Quansah",
        # Manchester City
        "Ruben Dias", "John Stones", "Nathan Ake", "Josko Gvardiol",
        "Kyle Walker", "Rico Lewis", "Manuel Akanji",
        # Chelsea
        "Thiago Silva", "Wesley Fofana", "Benoit Badiashile", "Levi Colwill",
        "Reece James", "Ben Chilwell", "Marc Cucurella", "Malo Gusto",
        # Newcastle
        "Sven Botman", "Fabian Schar", "Dan Burn", "Kieran Trippier",
        "Tino Livramento", "Lewis Hall", "Emil Krafth",
        # Tottenham
        "Cristian Romero", "Micky van de Ven", "Pedro Porro", "Destiny Udogie",
        "Radu Dragusin", "Ben Davies", "Djed Spence",
        # Manchester United
        "Lisandro Martinez", "Raphael Varane", "Harry Maguire", "Luke Shaw",
        "Diogo Dalot", "Aaron Wan-Bissaka", "Tyrell Malacia",
        # Aston Villa
        "Ezri Konsa", "Pau Torres", "Diego Carlos", "Lucas Digne",
        "Matty Cash", "Alex Moreno",
        # Brighton
        "Lewis Dunk", "Joel Veltman", "Jan Paul van Hecke", "Pervis Estupinan",
        "Tariq Lamptey", "Igor Julio", "Adam Webster",
        # West Ham
        "Kurt Zouma", "Nayef Aguerd", "Vladimir Coufal", "Aaron Cresswell",
        "Emerson Palmieri", "Ben Johnson",
        # Others
        "James Tarkowski", "Jarrad Branthwaite", "Nathan Collins", "Max Kilman",
        "Craig Dawson", "Joachim Andersen", "Marc Guehi", "Tyrick Mitchell",
        "Joel Ward", "Nathaniel Clyne", "Ryan Manning", "Rayan Ait-Nouri",
    ],
    "MID": [
        # Arsenal
        "Martin Odegaard", "Declan Rice", "Kai Havertz", "Thomas Partey",
        "Jorginho", "Mikel Merino", "Emile Smith Rowe", "Fabio Vieira",
        # Liverpool
        "Mohamed Salah", "Luis Diaz", "Dominik Szoboszlai", "Alexis Mac Allister",
        "Curtis Jones", "Ryan Gravenberch", "Wataru Endo", "Harvey Elliott",
        # Manchester City
        "Kevin De Bruyne", "Bernardo Silva", "Phil Foden", "Ilkay Gundogan",
        "Mateo Kovacic", "Jeremy Doku", "Jack Grealish", "Matheus Nunes",
        # Chelsea
        "Enzo Fernandez", "Moises Caicedo", "Conor Gallagher", "Christopher Nkunku",
        "Cole Palmer", "Raheem Sterling", "Mykhailo Mudryk", "Carney Chukwuemeka",
        # Tottenham
        "James Maddison", "Dejan Kulusevski", "Pape Matar Sarr", "Yves Bissouma",
        "Rodrigo Bentancur", "Brennan Johnson", "Manor Solomon",
        # Manchester United
        "Bruno Fernandes", "Casemiro", "Christian Eriksen", "Mason Mount",
        "Kobbie Mainoo", "Scott McTominay", "Antony", "Marcus Rashford",
        # Newcastle
        "Bruno Guimaraes", "Sandro Tonali", "Joelinton", "Alexander Isak",
        "Anthony Gordon", "Harvey Barnes", "Sean Longstaff", "Elliot Anderson",
        # Aston Villa
        "John McGinn", "Douglas Luiz", "Boubacar Kamara", "Leon Bailey",
        "Jacob Ramsey", "Moussa Diaby", "Youri Tielemans", "Morgan Rogers",
        # Brighton
        "Pascal Gross", "Kaoru Mitoma", "Simon Adingra", "Carlos Baleba",
        "Billy Gilmour", "Adam Lallana", "Facundo Buonanotte", "Julio Enciso",
        # West Ham
        "Jarrod Bowen", "Lucas Paqueta", "Tomas Soucek", "James Ward-Prowse",
        "Mohammed Kudus", "Pablo Fornals", "Said Benrahma", "Flynn Downes",
        # Others
        "Abdoulaye Doucoure", "Amadou Onana", "James Garner", "Dwight McNeil",
        "Jack Harrison", "Idrissa Gueye", "Vitinho", "Andreas Pereira",
        "Alex Iwobi", "Tom Cairney", "Harrison Reed", "Timothy Castagne",
        "Wilfred Ndidi", "Kiernan Dewsbury-Hall", "Abdul Fatawu", "Stephy Mavididi",
    ],
    "FWD": [
        # Top-tier forwards
        "Erling Haaland", "Darwin Nunez", "Gabriel Jesus", "Alexander Isak",
        "Ivan Toney", "Ollie Watkins", "Dominic Solanke", "Richarlison",
        "Nicolas Jackson", "Callum Wilson", "Danny Welbeck", "Michail Antonio",
        # Mid-tier forwards
        "Evan Ferguson", "Jamie Vardy", "Matheus Cunha", "Jean-Philippe Mateta",
        "Beto", "Rodrigo Muniz", "Cameron Archer", "Liam Delap",
        "Chris Wood", "Jhon Duran", "Adam Armstrong", "Ben Brereton Diaz",
        # Emerging/Squad forwards
        "Elijah Adebayo", "Yoane Wissa", "Neal Maupay", "Armando Broja",
        "Anthony Martial", "Folarin Balogun", "Eddie Nketiah", "Diogo Jota",
        "Luis Suarez", "Cody Gakpo", "Roberto Firmino", "Gabriel Martinelli",
        "Brennan Johnson", "Morgan Gibbs-White", "Eberechi Eze", "Wilfried Zaha",
        "Carlton Morris", "Troy Deeney", "Ashley Barnes", "Che Adams",
    ]
}

@dataclass
class StatisticalDistribution:
    """Statistical distribution parameters for realistic data generation."""
    mean: float
    std: float
    min_val: float = 0.0
    max_val: float = None
    
    def generate(self) -> float:
        """Generate a random value from this distribution."""
        value = random.gauss(self.mean, self.std)
        value = max(value, self.min_val)
        if self.max_val is not None:
            value = min(value, self.max_val)
        return round(value, 2)

class Position(Enum):
    GK = "GK"
    DEF = "DEF" 
    MID = "MID"
    FWD = "FWD"

# Realistic statistical distributions by position for new PlayerAttributes fields
STATISTICAL_DISTRIBUTIONS = {
    "GK": {
        "xg_per_90": StatisticalDistribution(0.02, 0.01, 0.0, 0.1),
        "xa_per_90": StatisticalDistribution(0.01, 0.005, 0.0, 0.05),
        "xgi_per_90": StatisticalDistribution(0.03, 0.015, 0.0, 0.15),
        "form_3_games": StatisticalDistribution(4.2, 1.8, 0.0, 15.0),
        "form_5_games": StatisticalDistribution(4.1, 1.6, 0.0, 12.0),
        "form_10_games": StatisticalDistribution(4.0, 1.4, 0.0, 10.0),
        "momentum": StatisticalDistribution(0.0, 0.3, -1.0, 1.0),
        "next_3_fixture_difficulty": StatisticalDistribution(3.0, 0.8, 1.0, 5.0),
        "next_5_fixture_difficulty": StatisticalDistribution(3.0, 0.7, 1.0, 5.0),
        "role_confidence": StatisticalDistribution(0.1, 0.05, 0.0, 0.3),
        "shots_per_90": StatisticalDistribution(0.1, 0.05, 0.0, 0.5),
        "key_passes_per_90": StatisticalDistribution(12.5, 4.2, 5.0, 25.0),
        "tackles_per_90": StatisticalDistribution(0.3, 0.2, 0.0, 1.0),
        "interceptions_per_90": StatisticalDistribution(1.2, 0.5, 0.0, 3.0),
        "clearances_per_90": StatisticalDistribution(4.8, 2.1, 1.0, 12.0),
        "price_range": (40, 65),  # GK prices in 0.1m units
    },
    "DEF": {
        "xg_per_90": StatisticalDistribution(0.08, 0.04, 0.0, 0.3),
        "xa_per_90": StatisticalDistribution(0.15, 0.08, 0.0, 0.5),
        "xgi_per_90": StatisticalDistribution(0.23, 0.11, 0.0, 0.8),
        "form_3_games": StatisticalDistribution(4.8, 2.1, 0.0, 15.0),
        "form_5_games": StatisticalDistribution(4.7, 1.9, 0.0, 12.0),
        "form_10_games": StatisticalDistribution(4.6, 1.7, 0.0, 10.0),
        "momentum": StatisticalDistribution(0.0, 0.4, -1.0, 1.0),
        "next_3_fixture_difficulty": StatisticalDistribution(3.0, 0.8, 1.0, 5.0),
        "next_5_fixture_difficulty": StatisticalDistribution(3.0, 0.7, 1.0, 5.0),
        "role_confidence": StatisticalDistribution(0.25, 0.15, 0.0, 0.7),
        "shots_per_90": StatisticalDistribution(0.8, 0.4, 0.0, 2.5),
        "key_passes_per_90": StatisticalDistribution(1.2, 0.6, 0.0, 3.5),
        "tackles_per_90": StatisticalDistribution(2.1, 0.8, 0.5, 5.0),
        "interceptions_per_90": StatisticalDistribution(1.8, 0.7, 0.2, 4.0),
        "clearances_per_90": StatisticalDistribution(3.2, 1.4, 0.5, 8.0),
        "price_range": (40, 85),  # DEF prices
    },
    "MID": {
        "xg_per_90": StatisticalDistribution(0.25, 0.15, 0.0, 0.8),
        "xa_per_90": StatisticalDistribution(0.35, 0.20, 0.0, 1.0),
        "xgi_per_90": StatisticalDistribution(0.60, 0.30, 0.0, 1.5),
        "form_3_games": StatisticalDistribution(5.4, 2.4, 0.0, 20.0),
        "form_5_games": StatisticalDistribution(5.3, 2.2, 0.0, 16.0),
        "form_10_games": StatisticalDistribution(5.2, 2.0, 0.0, 14.0),
        "momentum": StatisticalDistribution(0.0, 0.5, -1.0, 1.0),
        "next_3_fixture_difficulty": StatisticalDistribution(3.0, 0.8, 1.0, 5.0),
        "next_5_fixture_difficulty": StatisticalDistribution(3.0, 0.7, 1.0, 5.0),
        "role_confidence": StatisticalDistribution(0.45, 0.25, 0.0, 0.9),
        "shots_per_90": StatisticalDistribution(2.1, 1.0, 0.2, 6.0),
        "key_passes_per_90": StatisticalDistribution(2.8, 1.4, 0.5, 8.0),
        "tackles_per_90": StatisticalDistribution(1.6, 0.8, 0.2, 4.5),
        "interceptions_per_90": StatisticalDistribution(1.2, 0.6, 0.1, 3.5),
        "clearances_per_90": StatisticalDistribution(0.8, 0.5, 0.0, 3.0),
        "price_range": (45, 140),  # MID prices (wide range for different types)
    },
    "FWD": {
        "xg_per_90": StatisticalDistribution(0.45, 0.25, 0.0, 1.2),
        "xa_per_90": StatisticalDistribution(0.20, 0.12, 0.0, 0.6),
        "xgi_per_90": StatisticalDistribution(0.65, 0.30, 0.0, 1.5),
        "form_3_games": StatisticalDistribution(5.8, 2.6, 0.0, 25.0),
        "form_5_games": StatisticalDistribution(5.7, 2.4, 0.0, 20.0),
        "form_10_games": StatisticalDistribution(5.6, 2.2, 0.0, 18.0),
        "momentum": StatisticalDistribution(0.0, 0.6, -1.0, 1.0),
        "next_3_fixture_difficulty": StatisticalDistribution(3.0, 0.8, 1.0, 5.0),
        "next_5_fixture_difficulty": StatisticalDistribution(3.0, 0.7, 1.0, 5.0),
        "role_confidence": StatisticalDistribution(0.55, 0.25, 0.0, 0.95),
        "shots_per_90": StatisticalDistribution(3.2, 1.4, 0.5, 8.0),
        "key_passes_per_90": StatisticalDistribution(1.5, 0.8, 0.1, 4.0),
        "tackles_per_90": StatisticalDistribution(0.6, 0.4, 0.0, 2.0),
        "interceptions_per_90": StatisticalDistribution(0.4, 0.3, 0.0, 1.5),
        "clearances_per_90": StatisticalDistribution(0.2, 0.2, 0.0, 1.0),
        "price_range": (50, 150),  # FWD prices (highest range)
    }
}

# Fixture difficulty ratings for different team matchups
FIXTURE_DIFFICULTY_MATRIX = {
    # Top 6 teams are generally more difficult
    ("ARS", "LIV"): 4.5, ("ARS", "MCI"): 5.0, ("ARS", "CHE"): 4.0,
    ("LIV", "MCI"): 4.8, ("LIV", "CHE"): 4.2, ("MCI", "CHE"): 4.3,
    
    # Mid-table teams
    ("AVL", "NEW"): 3.2, ("BHA", "WHU"): 3.0, ("FUL", "BOU"): 2.8,
    
    # Newly promoted/relegation candidates
    ("IPS", "SOU"): 2.5, ("LEI", "NFO"): 2.7,
}

def get_realistic_player_name(position: str) -> str:
    """Get a random realistic player name for the given position."""
    if position not in REALISTIC_PLAYERS:
        position = "MID"  # Default fallback
    return random.choice(REALISTIC_PLAYERS[position])

def get_random_team() -> Tuple[str, str]:
    """Get a random team code and full name."""
    team_code = random.choice(list(PREMIER_LEAGUE_TEAMS.keys()))
    return team_code, PREMIER_LEAGUE_TEAMS[team_code]

def get_position_stats(position: str) -> Dict[str, float]:
    """Generate realistic statistics for a player in the given position."""
    if position not in STATISTICAL_DISTRIBUTIONS:
        position = "MID"  # Default fallback
    
    position_stats = STATISTICAL_DISTRIBUTIONS[position]
    stats = {}
    
    for stat_name, distribution in position_stats.items():
        if stat_name == "price_range":
            continue  # Handle separately
        stats[stat_name] = distribution.generate()
    
    return stats

def get_position_price_range(position: str) -> Tuple[int, int]:
    """Get realistic price range for a player position."""
    if position not in STATISTICAL_DISTRIBUTIONS:
        position = "MID"
    return STATISTICAL_DISTRIBUTIONS[position]["price_range"]

def generate_realistic_price(position: str, form_3_games: float = None) -> int:
    """Generate a realistic price based on position and form."""
    price_range = get_position_price_range(position)
    base_price = random.randint(price_range[0], price_range[1])
    
    # Adjust price based on form if provided
    if form_3_games is not None:
        if form_3_games > 7.0:  # High form
            base_price += random.randint(5, 15)
        elif form_3_games < 3.0:  # Poor form
            base_price -= random.randint(5, 10)
    
    # Ensure price stays within reasonable bounds
    return max(price_range[0], min(price_range[1] + 20, base_price))

def get_fixture_difficulty(home_team: str, away_team: str, home_advantage: bool = True) -> float:
    """Calculate fixture difficulty rating between two teams."""
    # Check if we have a specific rating for this matchup
    matchup = (home_team, away_team)
    reverse_matchup = (away_team, home_team)
    
    if matchup in FIXTURE_DIFFICULTY_MATRIX:
        base_difficulty = FIXTURE_DIFFICULTY_MATRIX[matchup]
    elif reverse_matchup in FIXTURE_DIFFICULTY_MATRIX:
        base_difficulty = FIXTURE_DIFFICULTY_MATRIX[reverse_matchup]
    else:
        # Generate based on team "strength" (simplified approach)
        top_6 = ["ARS", "LIV", "MCI", "CHE", "TOT", "MUN"]
        
        home_strength = 4.5 if home_team in top_6 else 3.0
        away_strength = 4.0 if away_team in top_6 else 2.8
        
        base_difficulty = (home_strength + away_strength) / 2
    
    # Apply home advantage
    if home_advantage:
        base_difficulty *= 0.9  # Slightly easier at home
    
    # Add some randomness
    base_difficulty += random.uniform(-0.3, 0.3)
    
    return max(1.0, min(5.0, round(base_difficulty, 1)))

def generate_season_gameweek() -> Tuple[str, int]:
    """Generate a realistic season and gameweek combination."""
    seasons = ["2324", "2425", "2526"]  # Current and upcoming seasons
    season = random.choice(seasons)
    gameweek = random.randint(1, 38)
    return season, gameweek

def generate_form_progression(base_form: float, num_weeks: int = 10) -> List[float]:
    """Generate a realistic form progression over multiple gameweeks."""
    progression = [base_form]
    
    for _ in range(num_weeks - 1):
        # Form tends to regress to mean (5.0) with some randomness
        change = random.uniform(-1.5, 1.5)
        trend_to_mean = (5.0 - progression[-1]) * 0.1
        
        new_form = progression[-1] + change + trend_to_mean
        new_form = max(0.0, min(15.0, new_form))
        progression.append(round(new_form, 1))
    
    return progression

# Role probability distributions
ROLE_PROBABILITIES = {
    "GK": {
        "is_penalty_taker": 0.02,  # Very rare
        "is_free_kick_taker": 0.05,
        "is_corner_taker": 0.01,
    },
    "DEF": {
        "is_penalty_taker": 0.15,  # Some center-backs take penalties
        "is_free_kick_taker": 0.25,
        "is_corner_taker": 0.10,
    },
    "MID": {
        "is_penalty_taker": 0.40,  # Most penalty takers are midfielders
        "is_free_kick_taker": 0.60,
        "is_corner_taker": 0.50,
    },
    "FWD": {
        "is_penalty_taker": 0.35,  # Strikers often take penalties
        "is_free_kick_taker": 0.30,
        "is_corner_taker": 0.05,  # Rare for forwards
    },
}

def get_player_roles(position: str) -> Dict[str, bool]:
    """Generate realistic player role assignments based on position."""
    if position not in ROLE_PROBABILITIES:
        position = "MID"
    
    probs = ROLE_PROBABILITIES[position]
    return {
        "is_penalty_taker": random.random() < probs["is_penalty_taker"],
        "is_free_kick_taker": random.random() < probs["is_free_kick_taker"], 
        "is_corner_taker": random.random() < probs["is_corner_taker"],
    }

# Utility function to get all data for a realistic player
def generate_realistic_player_data(position: str = None, team: str = None) -> Dict:
    """Generate a complete set of realistic data for a player."""
    if position is None:
        position = random.choice(list(Position)).value
    
    if team is None:
        team, _ = get_random_team()
    
    # Get basic stats and roles
    stats = get_position_stats(position)
    roles = get_player_roles(position)
    season, gameweek = generate_season_gameweek()
    
    # Calculate role confidence based on roles
    role_count = sum(roles.values())
    if role_count > 0:
        base_confidence = 0.6 + (role_count * 0.15)
        confidence_noise = random.uniform(-0.2, 0.2)
        role_confidence = max(0.0, min(1.0, base_confidence + confidence_noise))
    else:
        role_confidence = random.uniform(0.0, 0.3)
    
    stats["role_confidence"] = round(role_confidence, 2)
    
    return {
        "name": get_realistic_player_name(position),
        "position": position,
        "team": team,
        "season": season,
        "gameweek": gameweek,
        "price": generate_realistic_price(position, stats.get("form_3_games")),
        **stats,
        **roles,
    }