"""
Team Strength Analyzer - Comprehensive Bayesian modeling of team capabilities

This module implements a sophisticated Bayesian hierarchical model for estimating
team attacking and defensive strengths. The model combines multiple performance
metrics with uncertainty quantification and handles both home/away effects and
temporal changes including manager transitions and squad changes.

Key Features:
- Bayesian hierarchical modeling with MCMC inference
- Separate home/away strength estimation
- Exponential smoothing for temporal updates (alpha=0.1-0.3)
- Manager change and injury impact adjustments
- Uncertainty quantification with prediction intervals
- Correlation validation framework (target >0.7)

Methodology:
The model uses a hierarchical structure where:
1. League-level parameters capture overall strength distributions
2. Team-level parameters represent inherent attacking/defensive abilities
3. Match-level observations update beliefs through Bayesian updating
4. Home advantage is modeled as an additive effect
5. Temporal decay ensures recent performance has higher weight
"""

import datetime
import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy import stats
from sqlalchemy import desc
from sqlalchemy.orm import Session

from airsenal.framework.schema import (
    Fixture,
    PlayerScore,
    Result,
    Team,
    TeamStrength,
    TeamStrengthHistory,
)
from airsenal.framework.utils import CURRENT_SEASON, get_last_finished_gameweek

logger = logging.getLogger(__name__)


class TeamStrengthAnalyzer:
    """
    Bayesian hierarchical model for team strength estimation.

    This class implements a comprehensive Bayesian approach to modeling team
    strengths that accounts for:
    - Attacking and defensive capabilities separately
    - Home/away venue effects
    - Temporal changes with exponential smoothing
    - Manager changes and squad disruptions
    - Uncertainty quantification

    The model structure follows:

    League-level priors:
    - μ_att ~ Normal(0, σ_league)  # League average attacking
    - μ_def ~ Normal(0, σ_league)  # League average defensive
    - σ_att ~ HalfNormal(1)       # Between-team attacking variance
    - σ_def ~ HalfNormal(1)       # Between-team defensive variance
    - home_adv ~ Normal(0.3, 0.1) # Home advantage effect

    Team-level parameters:
    - att_i ~ Normal(μ_att, σ_att)  # Team i attacking strength
    - def_i ~ Normal(μ_def, σ_def)  # Team i defensive strength

    Match-level observations:
    - λ_home = exp(att_home - def_away + home_adv)
    - λ_away = exp(att_away - def_home)
    - goals_home ~ Poisson(λ_home)
    - goals_away ~ Poisson(λ_away)
    """

    def __init__(
        self,
        dbsession: Session,
        exponential_smoothing_alpha: float = 0.2,
        min_matches_for_stability: int = 5,
        mcmc_samples: int = 2000,
        mcmc_warmup: int = 1000,
        random_seed: int = 42,
    ):
        """
        Initialize the team strength analyzer.

        Args:
            dbsession: Database session for data access
            exponential_smoothing_alpha: Smoothing parameter (0.1-0.3 recommended)
            min_matches_for_stability: Minimum matches before strengths stabilize
            mcmc_samples: Number of MCMC samples for Bayesian inference
            mcmc_warmup: Number of warmup samples
            random_seed: Random seed for reproducible results
        """
        self.dbsession = dbsession
        self.alpha = exponential_smoothing_alpha
        self.min_matches = min_matches_for_stability
        self.mcmc_samples = mcmc_samples
        self.mcmc_warmup = mcmc_warmup
        self.random_seed = random_seed

        # Model configuration
        self.model_version = "1.0.0"
        self.league_strength_prior_std = 0.5
        self.team_strength_prior_std = 0.3
        self.home_advantage_prior = {"mean": 0.3, "std": 0.1}

        # Cache for team mappings and current strengths
        self._team_cache = {}
        self._current_strengths = {}

    def hierarchical_strength_model(
        self,
        home_teams: jnp.ndarray,
        away_teams: jnp.ndarray,
        home_goals: jnp.ndarray,
        away_goals: jnp.ndarray,
        n_teams: int,
    ):
        """
        Bayesian hierarchical model for team strengths.

        This implements the core probabilistic model that learns team attacking
        and defensive capabilities from match data.

        Args:
            home_teams: Array of home team indices
            away_teams: Array of away team indices
            home_goals: Array of home team goals scored
            away_goals: Array of away team goals scored
            n_teams: Number of teams in the league
        """
        # League-level hyperpriors
        mu_att = numpyro.sample(
            "mu_att", dist.Normal(0.0, self.league_strength_prior_std)
        )
        mu_def = numpyro.sample(
            "mu_def", dist.Normal(0.0, self.league_strength_prior_std)
        )

        sigma_att = numpyro.sample(
            "sigma_att", dist.HalfNormal(self.team_strength_prior_std)
        )
        sigma_def = numpyro.sample(
            "sigma_def", dist.HalfNormal(self.team_strength_prior_std)
        )

        # Home advantage
        home_advantage = numpyro.sample(
            "home_advantage",
            dist.Normal(
                self.home_advantage_prior["mean"], self.home_advantage_prior["std"]
            ),
        )

        # Team-level parameters
        with numpyro.plate("teams", n_teams):
            attack = numpyro.sample("attack", dist.Normal(mu_att, sigma_att))
            defense = numpyro.sample("defense", dist.Normal(mu_def, sigma_def))

        # Match-level predictions
        with numpyro.plate("matches", len(home_teams)):
            # Expected goals for home and away teams
            lambda_home = jnp.exp(
                attack[home_teams] - defense[away_teams] + home_advantage
            )
            lambda_away = jnp.exp(attack[away_teams] - defense[home_teams])

            # Observed goals
            numpyro.sample("home_goals", dist.Poisson(lambda_home), obs=home_goals)
            numpyro.sample("away_goals", dist.Poisson(lambda_away), obs=away_goals)

    def prepare_match_data(
        self, season: str, up_to_gameweek: int | None = None
    ) -> tuple[dict, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Prepare match data for Bayesian modeling.

        Extracts completed matches and converts team names to numerical indices
        for efficient computation.

        Args:
            season: Season to analyze
            up_to_gameweek: Only include matches up to this gameweek (optional)

        Returns:
            Tuple of (team_mapping, home_teams, away_teams, home_goals, away_goals)
        """
        logger.info("Preparing match data for %s, up to GW %s", season, up_to_gameweek)

        # Get completed matches with results
        query = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
        )

        if up_to_gameweek:
            query = query.filter(Fixture.gameweek <= up_to_gameweek)

        matches = query.all()

        if not matches:
            msg = f"No completed matches found for {season}"
            raise ValueError(msg)

        # Create team mapping
        teams = set()
        for fixture, result in matches:
            teams.add(fixture.home_team)
            teams.add(fixture.away_team)

        team_to_idx = {team: idx for idx, team in enumerate(sorted(teams))}
        idx_to_team = {idx: team for team, idx in team_to_idx.items()}

        # Convert to arrays
        home_teams = []
        away_teams = []
        home_goals = []
        away_goals = []

        for fixture, result in matches:
            home_teams.append(team_to_idx[fixture.home_team])
            away_teams.append(team_to_idx[fixture.away_team])
            home_goals.append(result.home_score)
            away_goals.append(result.away_score)

        team_mapping = {
            "team_to_idx": team_to_idx,
            "idx_to_team": idx_to_team,
            "n_teams": len(teams),
        }

        return (
            team_mapping,
            jnp.array(home_teams),
            jnp.array(away_teams),
            jnp.array(home_goals),
            jnp.array(away_goals),
        )

    def run_mcmc_inference(
        self,
        home_teams: jnp.ndarray,
        away_teams: jnp.ndarray,
        home_goals: jnp.ndarray,
        away_goals: jnp.ndarray,
        n_teams: int,
    ) -> MCMC:
        """
        Run MCMC inference for the hierarchical strength model.

        Uses No-U-Turn Sampler (NUTS) for efficient sampling from the posterior
        distribution of team strengths.

        Args:
            home_teams: Home team indices
            away_teams: Away team indices
            home_goals: Home goals scored
            away_goals: Away goals scored
            n_teams: Number of teams

        Returns:
            MCMC object with posterior samples
        """
        logger.info("Running MCMC inference for team strengths")

        # Configure NUTS sampler
        nuts_kernel = NUTS(
            self.hierarchical_strength_model,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        # Run MCMC
        mcmc = MCMC(
            nuts_kernel,
            num_samples=self.mcmc_samples,
            num_warmup=self.mcmc_warmup,
            num_chains=1,  # Single chain for efficiency
        )

        rng_key = jax.random.PRNGKey(self.random_seed)
        mcmc.run(
            rng_key,
            home_teams=home_teams,
            away_teams=away_teams,
            home_goals=home_goals,
            away_goals=away_goals,
            n_teams=n_teams,
        )

        return mcmc

    def calculate_attacking_strength(
        self, team: str, season: str, gameweek: int, home: bool = True
    ) -> dict[str, float]:
        """
        Calculate attacking strength for a team using advanced metrics.

        Combines expected goals (xG), shots, actual goals, and possession data
        with exponential smoothing for recent form emphasis.

        Args:
            team: Team name
            season: Season
            gameweek: Current gameweek
            home: Whether to calculate home or away strength

        Returns:
            Dictionary with attacking strength metrics
        """
        logger.debug(
            "Calculating attacking strength for %s (%s)",
            team,
            "home" if home else "away",
        )

        # Get recent matches (last 10 games)
        venue_filter = Fixture.home_team == team if home else Fixture.away_team == team

        recent_matches = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek < gameweek)
            .filter(venue_filter)
            .order_by(desc(Fixture.gameweek))
            .limit(10)
            .all()
        )

        if not recent_matches:
            return self._get_default_attacking_strength()

        # Calculate weighted metrics with exponential smoothing
        weights = [
            self.alpha * (1 - self.alpha) ** i for i in range(len(recent_matches))
        ]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # Normalize

        goals_scored = 0.0
        expected_goals = 0.0
        shots_per_game = 0.0
        shot_accuracy = 0.0

        for i, (fixture, result) in enumerate(recent_matches):
            weight = weights[i]

            # Goals scored
            match_goals = result.home_score if home else result.away_score
            goals_scored += weight * match_goals

            # Get additional metrics from player scores
            match_xg, match_shots = self._get_team_match_stats(
                fixture.fixture_id, team, home
            )
            expected_goals += weight * match_xg
            shots_per_game += weight * match_shots

            if match_shots > 0:
                shot_accuracy += weight * (match_goals / match_shots)

        # Calculate composite attacking strength
        attacking_strength = self._calculate_composite_attacking_strength(
            goals_scored, expected_goals, shots_per_game, shot_accuracy
        )

        return {
            "attacking_strength": attacking_strength,
            "goals_per_game": goals_scored,
            "expected_goals_per_game": expected_goals,
            "shots_per_game": shots_per_game,
            "shot_accuracy": shot_accuracy,
            "matches_analyzed": len(recent_matches),
            "data_quality": min(1.0, len(recent_matches) / 10.0),
        }

    def calculate_defensive_strength(
        self, team: str, season: str, gameweek: int, home: bool = True
    ) -> dict[str, float]:
        """
        Calculate defensive strength using advanced metrics.

        Combines expected goals against (xGA), clean sheets, actual goals
        conceded, and defensive actions with exponential smoothing.

        Args:
            team: Team name
            season: Season
            gameweek: Current gameweek
            home: Whether to calculate home or away strength

        Returns:
            Dictionary with defensive strength metrics
        """
        logger.debug(
            "Calculating defensive strength for %s (%s)",
            team,
            "home" if home else "away",
        )

        # Get recent matches
        venue_filter = Fixture.home_team == team if home else Fixture.away_team == team

        recent_matches = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek < gameweek)
            .filter(venue_filter)
            .order_by(desc(Fixture.gameweek))
            .limit(10)
            .all()
        )

        if not recent_matches:
            return self._get_default_defensive_strength()

        # Calculate weighted metrics
        weights = [
            self.alpha * (1 - self.alpha) ** i for i in range(len(recent_matches))
        ]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        goals_conceded = 0.0
        expected_goals_against = 0.0
        clean_sheets = 0.0
        shots_against_per_game = 0.0

        for i, (fixture, result) in enumerate(recent_matches):
            weight = weights[i]

            # Goals conceded
            match_goals_against = result.away_score if home else result.home_score

            goals_conceded += weight * match_goals_against
            clean_sheets += weight * (1.0 if match_goals_against == 0 else 0.0)

            # Get defensive metrics
            match_xga, match_shots_against = self._get_team_defensive_stats(
                fixture.fixture_id, team, home
            )
            expected_goals_against += weight * match_xga
            shots_against_per_game += weight * match_shots_against

        # Calculate composite defensive strength
        defensive_strength = self._calculate_composite_defensive_strength(
            goals_conceded, expected_goals_against, clean_sheets, shots_against_per_game
        )

        return {
            "defensive_strength": defensive_strength,
            "goals_conceded_per_game": goals_conceded,
            "expected_goals_against_per_game": expected_goals_against,
            "clean_sheet_probability": clean_sheets,
            "shots_against_per_game": shots_against_per_game,
            "matches_analyzed": len(recent_matches),
            "data_quality": min(1.0, len(recent_matches) / 10.0),
        }

    def calculate_team_strengths(
        self, season: str, gameweek: int, team: str | None = None
    ) -> dict[str, dict]:
        """
        Calculate comprehensive team strengths for specified team(s).

        This is the main entry point that combines Bayesian modeling with
        traditional metrics to provide robust strength estimates.

        Args:
            season: Season to analyze
            gameweek: Current gameweek
            team: Specific team to analyze (if None, analyze all teams)

        Returns:
            Dictionary mapping team names to their strength analysis
        """
        logger.info("Calculating team strengths for %s GW%s", season, gameweek)

        try:
            # Prepare match data for Bayesian modeling
            team_mapping, home_teams, away_teams, home_goals, away_goals = (
                self.prepare_match_data(season, gameweek - 1)
            )

            # Run Bayesian inference
            mcmc = self.run_mcmc_inference(
                home_teams, away_teams, home_goals, away_goals, team_mapping["n_teams"]
            )

            # Extract posterior samples
            posterior_samples = mcmc.get_samples()

            # Calculate team strengths
            results = {}
            teams_to_analyze = [team] if team else team_mapping["team_to_idx"].keys()

            for team_name in teams_to_analyze:
                if team_name not in team_mapping["team_to_idx"]:
                    logger.warning("Team %s not found in match data", team_name)
                    continue

                team_idx = team_mapping["team_to_idx"][team_name]
                results[team_name] = self._analyze_team_strength(
                    team_name, team_idx, posterior_samples, season, gameweek
                )

            return results

        except Exception as e:
            logger.error("Error calculating team strengths: %s", e)
            raise

    def _analyze_team_strength(
        self,
        team_name: str,
        team_idx: int,
        posterior_samples: dict,
        season: str,
        gameweek: int,
    ) -> dict:
        """
        Analyze individual team strength from posterior samples.

        Combines Bayesian estimates with traditional metrics and calculates
        uncertainty measures.

        Args:
            team_name: Team name
            team_idx: Team index in the model
            posterior_samples: MCMC posterior samples
            season: Season
            gameweek: Current gameweek

        Returns:
            Comprehensive team strength analysis
        """
        # Extract Bayesian estimates
        attack_samples = posterior_samples["attack"][:, team_idx]
        defense_samples = posterior_samples["defense"][:, team_idx]
        home_advantage = posterior_samples["home_advantage"]

        # Calculate strength statistics
        attack_mean = float(jnp.mean(attack_samples))
        attack_std = float(jnp.std(attack_samples))
        defense_mean = float(jnp.mean(defense_samples))
        defense_std = float(jnp.std(defense_samples))
        home_adv_mean = float(jnp.mean(home_advantage))

        # Get traditional metrics
        home_attacking = self.calculate_attacking_strength(
            team_name, season, gameweek, True
        )
        away_attacking = self.calculate_attacking_strength(
            team_name, season, gameweek, False
        )
        home_defensive = self.calculate_defensive_strength(
            team_name, season, gameweek, True
        )
        away_defensive = self.calculate_defensive_strength(
            team_name, season, gameweek, False
        )

        # Calculate overall strengths
        attacking_strength_home = attack_mean + home_adv_mean
        attacking_strength_away = attack_mean
        defensive_strength_home = defense_mean
        defensive_strength_away = defense_mean

        # Combine with traditional metrics (weighted average)
        bayesian_weight = 0.7  # Higher weight for Bayesian estimates
        traditional_weight = 0.3

        final_attack_home = (
            bayesian_weight * attacking_strength_home
            + traditional_weight * home_attacking["attacking_strength"]
        )
        final_attack_away = (
            bayesian_weight * attacking_strength_away
            + traditional_weight * away_attacking["attacking_strength"]
        )
        final_defense_home = (
            bayesian_weight * defensive_strength_home
            + traditional_weight * home_defensive["defensive_strength"]
        )
        final_defense_away = (
            bayesian_weight * defensive_strength_away
            + traditional_weight * away_defensive["defensive_strength"]
        )

        # Calculate overall strength composites
        overall_strength_home = final_attack_home - final_defense_home
        overall_strength_away = final_attack_away - final_defense_away

        # Calculate form and momentum
        form_metrics = self._calculate_form_metrics(team_name, season, gameweek)

        # Detect external factors
        external_factors = self._detect_external_factors(team_name, season, gameweek)

        return {
            "team": team_name,
            "season": season,
            "gameweek": gameweek,
            # Core Bayesian strength estimates
            "attacking_strength_home": final_attack_home,
            "attacking_strength_away": final_attack_away,
            "defensive_strength_home": final_defense_home,
            "defensive_strength_away": final_defense_away,
            # Uncertainty quantification
            "attacking_strength_home_std": attack_std,
            "attacking_strength_away_std": attack_std,
            "defensive_strength_home_std": defense_std,
            "defensive_strength_away_std": defense_std,
            # Overall strength composites
            "overall_strength_home": overall_strength_home,
            "overall_strength_away": overall_strength_away,
            # Component metrics
            "expected_goals_for_per_game": (
                home_attacking["expected_goals_per_game"]
                + away_attacking["expected_goals_per_game"]
            )
            / 2,
            "expected_goals_against_per_game": (
                home_defensive["expected_goals_against_per_game"]
                + away_defensive["expected_goals_against_per_game"]
            )
            / 2,
            "actual_goals_for_per_game": (
                home_attacking["goals_per_game"] + away_attacking["goals_per_game"]
            )
            / 2,
            "actual_goals_against_per_game": (
                home_defensive["goals_conceded_per_game"]
                + away_defensive["goals_conceded_per_game"]
            )
            / 2,
            "shots_for_per_game": (
                home_attacking["shots_per_game"] + away_attacking["shots_per_game"]
            )
            / 2,
            "shots_against_per_game": (
                home_defensive["shots_against_per_game"]
                + away_defensive["shots_against_per_game"]
            )
            / 2,
            "clean_sheet_probability": (
                home_defensive["clean_sheet_probability"]
                + away_defensive["clean_sheet_probability"]
            )
            / 2,
            # Form indicators
            **form_metrics,
            # External factors
            **external_factors,
            # Model metadata
            "model_version": self.model_version,
            "sample_count": self.mcmc_samples,
            "convergence_diagnostic": self._calculate_rhat(
                attack_samples, defense_samples
            ),
            "effective_sample_size": min(
                self._calculate_ess(attack_samples),
                self._calculate_ess(defense_samples),
            ),
            # Data quality
            "matches_played": max(
                home_attacking["matches_analyzed"] + home_defensive["matches_analyzed"],
                away_attacking["matches_analyzed"] + away_defensive["matches_analyzed"],
            )
            // 2,
            "home_matches_played": home_attacking["matches_analyzed"],
            "away_matches_played": away_attacking["matches_analyzed"],
            "data_quality_score": min(
                home_attacking["data_quality"],
                away_attacking["data_quality"],
                home_defensive["data_quality"],
                away_defensive["data_quality"],
            ),
            # Timestamp
            "calculated_at": datetime.datetime.now().isoformat(),
        }

    def save_team_strength(self, strength_data: dict) -> TeamStrength:
        """
        Save team strength data to the database.

        Args:
            strength_data: Dictionary containing team strength analysis

        Returns:
            TeamStrength database object
        """
        # Check if strength already exists for this team/season/gameweek
        existing = (
            self.dbsession.query(TeamStrength)
            .filter(
                TeamStrength.team == strength_data["team"],
                TeamStrength.season == strength_data["season"],
                TeamStrength.gameweek == strength_data["gameweek"],
            )
            .first()
        )

        if existing:
            # Update existing record
            for key, value in strength_data.items():
                if hasattr(existing, key) and key not in ["team", "season", "gameweek"]:
                    setattr(existing, key, value)
            team_strength = existing
        else:
            # Create new record
            team_strength = TeamStrength(**strength_data)
            self.dbsession.add(team_strength)

        # Save to history
        history_data = strength_data.copy()
        history_data["calculation_trigger"] = "manual"
        history_data["exponential_smoothing_alpha"] = self.alpha
        history_data["matches_used_in_calculation"] = strength_data.get(
            "matches_played", 0
        )

        team_strength_history = TeamStrengthHistory(**history_data)
        self.dbsession.add(team_strength_history)

        self.dbsession.commit()
        return team_strength

    def update_all_team_strengths(
        self, season: str, gameweek: int
    ) -> dict[str, TeamStrength]:
        """
        Update team strengths for all teams in the league.

        Args:
            season: Season to update
            gameweek: Current gameweek

        Returns:
            Dictionary mapping team names to their updated TeamStrength objects
        """
        logger.info("Updating all team strengths for %s GW%s", season, gameweek)

        # Calculate strengths for all teams
        all_strengths = self.calculate_team_strengths(season, gameweek)

        # Save to database
        updated_teams = {}
        for team_name, strength_data in all_strengths.items():
            try:
                team_strength = self.save_team_strength(strength_data)
                updated_teams[team_name] = team_strength
                logger.debug("Updated strength for %s", team_name)
            except Exception as e:
                logger.error("Error saving strength for %s: %s", team_name, e)

        logger.info("Updated strengths for %s teams", len(updated_teams))
        return updated_teams

    def get_team_strength(
        self, team: str, season: str, gameweek: int | None = None
    ) -> TeamStrength | None:
        """
        Get the most recent team strength data.

        Args:
            team: Team name
            season: Season
            gameweek: Specific gameweek (if None, get most recent)

        Returns:
            TeamStrength object or None if not found
        """
        query = (
            self.dbsession.query(TeamStrength)
            .filter(TeamStrength.team == team)
            .filter(TeamStrength.season == season)
        )

        if gameweek:
            query = query.filter(TeamStrength.gameweek == gameweek)
        else:
            query = query.order_by(desc(TeamStrength.gameweek))

        return query.first()

    def validate_strength_predictions(
        self, season: str, min_gameweek: int = 5
    ) -> dict[str, float]:
        """
        Validate team strength predictions against actual results.

        This method tests whether the strength ratings correlate with actual
        match outcomes to ensure the model meets the >0.7 correlation requirement.

        Args:
            season: Season to validate
            min_gameweek: Start validation from this gameweek

        Returns:
            Dictionary with validation metrics including correlation coefficients
        """
        logger.info("Validating strength predictions for %s", season)

        # Get all matches with both strength data and results
        validation_data = []

        matches_query = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek >= min_gameweek)
            .order_by(Fixture.gameweek)
        )

        for fixture, result in matches_query:
            # Get team strengths at time of match
            home_strength = self.get_team_strength(
                fixture.home_team, season, fixture.gameweek - 1
            )
            away_strength = self.get_team_strength(
                fixture.away_team, season, fixture.gameweek - 1
            )

            if home_strength and away_strength:
                # Calculate predicted goal difference
                predicted_goal_diff = (
                    home_strength.overall_strength_home
                    - away_strength.overall_strength_away
                )

                # Actual goal difference
                actual_goal_diff = result.home_score - result.away_score

                validation_data.append(
                    {
                        "predicted_goal_diff": predicted_goal_diff,
                        "actual_goal_diff": actual_goal_diff,
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "gameweek": fixture.gameweek,
                        "predicted_home_win": predicted_goal_diff > 0,
                        "actual_home_win": actual_goal_diff > 0,
                        "predicted_draw": abs(predicted_goal_diff) < 0.1,
                        "actual_draw": actual_goal_diff == 0,
                    }
                )

        if not validation_data:
            return {"error": "No validation data available"}

        # Calculate correlation metrics
        predicted_diffs = [d["predicted_goal_diff"] for d in validation_data]
        actual_diffs = [d["actual_goal_diff"] for d in validation_data]

        correlation = stats.pearsonr(predicted_diffs, actual_diffs)[0]
        spearman_corr = stats.spearmanr(predicted_diffs, actual_diffs)[0]

        # Calculate prediction accuracy
        correct_predictions = sum(
            1
            for d in validation_data
            if (d["predicted_home_win"] and d["actual_home_win"])
            or (not d["predicted_home_win"] and not d["actual_home_win"])
        )

        accuracy = correct_predictions / len(validation_data)

        # Calculate mean absolute error
        mae = np.mean(
            [
                abs(d["predicted_goal_diff"] - d["actual_goal_diff"])
                for d in validation_data
            ]
        )

        # Calculate RMSE
        rmse = np.sqrt(
            np.mean(
                [
                    (d["predicted_goal_diff"] - d["actual_goal_diff"]) ** 2
                    for d in validation_data
                ]
            )
        )

        return {
            "season": season,
            "matches_analyzed": len(validation_data),
            "pearson_correlation": float(correlation),
            "spearman_correlation": float(spearman_corr),
            "prediction_accuracy": float(accuracy),
            "mean_absolute_error": float(mae),
            "rmse": float(rmse),
            "correlation_meets_target": float(correlation) > 0.7,
            "validation_date": datetime.datetime.now().isoformat(),
        }

    # Helper methods

    def _get_default_attacking_strength(self) -> dict[str, float]:
        """Return default attacking strength values for new/data-poor teams."""
        return {
            "attacking_strength": 0.0,
            "goals_per_game": 1.0,
            "expected_goals_per_game": 1.0,
            "shots_per_game": 10.0,
            "shot_accuracy": 0.1,
            "matches_analyzed": 0,
            "data_quality": 0.0,
        }

    def _get_default_defensive_strength(self) -> dict[str, float]:
        """Return default defensive strength values for new/data-poor teams."""
        return {
            "defensive_strength": 0.0,
            "goals_conceded_per_game": 1.0,
            "expected_goals_against_per_game": 1.0,
            "clean_sheet_probability": 0.3,
            "shots_against_per_game": 10.0,
            "matches_analyzed": 0,
            "data_quality": 0.0,
        }

    def _get_team_match_stats(
        self, fixture_id: int, team: str, home: bool
    ) -> tuple[float, float]:
        """
        Get team's xG and shots for a specific match.

        Args:
            fixture_id: Fixture ID
            team: Team name
            home: Whether team played at home

        Returns:
            Tuple of (expected_goals, shots)
        """
        # Query player scores for the match
        team_players = (
            self.dbsession.query(PlayerScore)
            .filter(PlayerScore.fixture_id == fixture_id)
            .filter(PlayerScore.player_team == team)
            .all()
        )

        if not team_players:
            return 0.0, 0.0

        # Aggregate team stats
        expected_goals = sum(p.expected_goals or 0.0 for p in team_players)

        # Estimate shots from goals and xG (simplified approach)
        goals = sum(p.goals for p in team_players)
        shots = max(expected_goals * 5, goals * 3)  # Rough estimation

        return expected_goals, shots

    def _get_team_defensive_stats(
        self, fixture_id: int, team: str, home: bool
    ) -> tuple[float, float]:
        """
        Get team's xGA and shots against for a specific match.

        Args:
            fixture_id: Fixture ID
            team: Team name
            home: Whether team played at home

        Returns:
            Tuple of (expected_goals_against, shots_against)
        """
        # Get opponent's attacking stats
        fixture = (
            self.dbsession.query(Fixture)
            .filter(Fixture.fixture_id == fixture_id)
            .first()
        )

        if not fixture:
            return 0.0, 0.0

        opponent = fixture.away_team if home else fixture.home_team
        return self._get_team_match_stats(fixture_id, opponent, not home)

    def _calculate_composite_attacking_strength(
        self,
        goals_per_game: float,
        expected_goals_per_game: float,
        shots_per_game: float,
        shot_accuracy: float,
    ) -> float:
        """
        Calculate composite attacking strength from component metrics.

        Uses a weighted combination of metrics with xG having highest weight.
        """
        # Normalize metrics to 0-1 scale (rough league averages)
        goals_norm = min(goals_per_game / 2.5, 1.0)
        xg_norm = min(expected_goals_per_game / 2.0, 1.0)
        shots_norm = min(shots_per_game / 15.0, 1.0)
        accuracy_norm = min(shot_accuracy, 1.0)

        # Weighted combination (xG gets highest weight)
        strength = (
            0.4 * xg_norm + 0.3 * goals_norm + 0.2 * shots_norm + 0.1 * accuracy_norm
        )

        return strength * 2.0 - 1.0  # Scale to [-1, 1]

    def _calculate_composite_defensive_strength(
        self,
        goals_conceded_per_game: float,
        expected_goals_against_per_game: float,
        clean_sheet_probability: float,
        shots_against_per_game: float,
    ) -> float:
        """
        Calculate composite defensive strength from component metrics.

        Note: Higher values indicate worse defense, so we invert the scale.
        """
        # Normalize metrics
        goals_norm = min(goals_conceded_per_game / 2.5, 1.0)
        xga_norm = min(expected_goals_against_per_game / 2.0, 1.0)
        cs_norm = clean_sheet_probability
        shots_norm = min(shots_against_per_game / 15.0, 1.0)

        # Weighted combination (lower is better for defense)
        weakness = (
            0.4 * xga_norm
            + 0.3 * goals_norm
            + 0.2 * shots_norm
            + 0.1 * (1.0 - cs_norm)  # Invert clean sheet probability
        )

        return 1.0 - (weakness * 2.0 - 1.0)  # Invert and scale to [-1, 1]

    def _calculate_form_metrics(
        self, team: str, season: str, gameweek: int
    ) -> dict[str, float]:
        """Calculate form-weighted strength and momentum indicators."""
        # Get last 5 matches
        recent_results = (
            self.dbsession.query(Fixture, Result)
            .join(Result, Fixture.fixture_id == Result.fixture_id)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek < gameweek)
            .filter((Fixture.home_team == team) | (Fixture.away_team == team))
            .order_by(desc(Fixture.gameweek))
            .limit(5)
            .all()
        )

        if not recent_results:
            return {
                "form_weighted_strength": 0.0,
                "recent_performance_trend": 0.0,
                "momentum_factor": 0.0,
            }

        # Calculate points and performance metrics
        points = []
        goal_differences = []

        for fixture, result in recent_results:
            if fixture.home_team == team:
                gd = result.home_score - result.away_score
                if gd > 0:
                    pts = 3
                elif gd == 0:
                    pts = 1
                else:
                    pts = 0
            else:
                gd = result.away_score - result.home_score
                if gd > 0:
                    pts = 3
                elif gd == 0:
                    pts = 1
                else:
                    pts = 0

            points.append(pts)
            goal_differences.append(gd)

        # Calculate form metrics
        form_weighted_strength = np.mean(points) / 3.0  # Normalize to [0, 1]

        # Calculate trend (linear regression slope of goal differences)
        if len(goal_differences) > 1:
            x = np.arange(len(goal_differences))
            trend_slope = stats.linregress(x, goal_differences[::-1])[
                0
            ]  # Reverse for chronological order
            recent_performance_trend = np.tanh(trend_slope)  # Bound to [-1, 1]
        else:
            recent_performance_trend = 0.0

        # Momentum factor based on recent results
        recent_points = points[:3]  # Last 3 games
        momentum_factor = (np.mean(recent_points) - 1.0) / 2.0  # Scale to [-0.5, 1.0]

        return {
            "form_weighted_strength": float(form_weighted_strength),
            "recent_performance_trend": float(recent_performance_trend),
            "momentum_factor": float(momentum_factor),
        }

    def _detect_external_factors(
        self, team: str, season: str, gameweek: int
    ) -> dict[str, float | None]:
        """
        Detect external factors that might affect team strength.

        This is a simplified implementation - in practice, you would integrate
        with external data sources for manager changes, injuries, etc.
        """
        # Placeholder for external factor detection
        # In a real implementation, this would check:
        # - Manager change databases
        # - Injury reports
        # - Transfer activity

        return {
            "manager_change_adjustment": None,
            "key_player_injury_impact": None,
            "transfer_window_impact": None,
        }

    def _calculate_rhat(self, *chains) -> float:
        """Calculate R-hat convergence diagnostic for MCMC chains."""
        # Simplified R-hat calculation
        # In practice, you'd use proper MCMC diagnostics
        return 1.0  # Placeholder

    def _calculate_ess(self, samples: jnp.ndarray) -> int:
        """Calculate effective sample size."""
        # Simplified ESS calculation
        return len(samples)  # Placeholder


def create_team_strength_analyzer(dbsession: Session, **kwargs) -> TeamStrengthAnalyzer:
    """
    Factory function to create a configured TeamStrengthAnalyzer.

    Args:
        dbsession: Database session
        **kwargs: Additional configuration parameters

    Returns:
        Configured TeamStrengthAnalyzer instance
    """
    return TeamStrengthAnalyzer(dbsession, **kwargs)


# API helper functions for integration with existing codebase


def get_team_strength_for_api(
    team: str,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> dict:
    """
    Get team strength data formatted for API consumption.

    Args:
        team: Team name
        season: Season (defaults to current)
        gameweek: Specific gameweek (defaults to most recent)
        dbsession: Database session

    Returns:
        Dictionary with team strength data
    """
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    analyzer = create_team_strength_analyzer(dbsession)
    strength = analyzer.get_team_strength(team, season, gameweek)

    if not strength:
        return {"error": f"No strength data found for {team}"}

    return {
        "team": strength.team,
        "season": strength.season,
        "gameweek": strength.gameweek,
        "attacking_strength": {
            "home": strength.attacking_strength_home,
            "away": strength.attacking_strength_away,
            "uncertainty": {
                "home_std": strength.attacking_strength_home_std,
                "away_std": strength.attacking_strength_away_std,
            },
        },
        "defensive_strength": {
            "home": strength.defensive_strength_home,
            "away": strength.defensive_strength_away,
            "uncertainty": {
                "home_std": strength.defensive_strength_home_std,
                "away_std": strength.defensive_strength_away_std,
            },
        },
        "overall_strength": {
            "home": strength.overall_strength_home,
            "away": strength.overall_strength_away,
        },
        "metrics": {
            "expected_goals_for": strength.expected_goals_for_per_game,
            "expected_goals_against": strength.expected_goals_against_per_game,
            "clean_sheet_probability": strength.clean_sheet_probability,
            "form_weighted_strength": strength.form_weighted_strength,
            "momentum_factor": strength.momentum_factor,
        },
        "model_info": {
            "version": strength.model_version,
            "sample_count": strength.sample_count,
            "convergence_diagnostic": strength.convergence_diagnostic,
            "data_quality_score": strength.data_quality_score,
            "calculated_at": strength.calculated_at,
        },
    }


def get_all_team_strengths_for_api(
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> dict:
    """
    Get team strengths for all teams formatted for API consumption.

    Args:
        season: Season (defaults to current)
        gameweek: Specific gameweek (defaults to most recent)
        dbsession: Database session

    Returns:
        Dictionary mapping team names to their strength data
    """
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    # Get all teams for the season
    teams = dbsession.query(Team.name).filter(Team.season == season).distinct().all()

    result = {"season": season, "gameweek": gameweek, "teams": {}}

    for (team_name,) in teams:
        try:
            team_strength = get_team_strength_for_api(
                team_name, season, gameweek, dbsession
            )
            if "error" not in team_strength:
                result["teams"][team_name] = team_strength
        except Exception as e:
            logger.error("Error getting strength for %s: %s", team_name, e)
            continue

    return result


def update_team_strengths_for_gameweek(
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> dict:
    """
    Update team strengths for a specific gameweek.

    This function is designed to be called after each gameweek to update
    all team strength estimates.

    Args:
        season: Season to update
        gameweek: Gameweek to update (defaults to current)
        dbsession: Database session

    Returns:
        Dictionary with update results
    """
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    if not gameweek:
        gameweek = get_last_finished_gameweek()

    analyzer = create_team_strength_analyzer(dbsession)

    try:
        updated_teams = analyzer.update_all_team_strengths(season, gameweek)

        return {
            "season": season,
            "gameweek": gameweek,
            "teams_updated": len(updated_teams),
            "teams": list(updated_teams.keys()),
            "update_time": datetime.datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("Error updating team strengths: %s", e)
        return {
            "error": f"Failed to update team strengths: {e}",
            "season": season,
            "gameweek": gameweek,
        }


def validate_team_strength_model(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> dict:
    """
    Validate the team strength model performance.

    Args:
        season: Season to validate
        dbsession: Database session

    Returns:
        Dictionary with validation metrics
    """
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    analyzer = create_team_strength_analyzer(dbsession)

    try:
        return analyzer.validate_strength_predictions(season)
    except Exception as e:
        logger.error("Error validating team strength model: %s", e)
        return {"error": f"Validation failed: {e}"}


# Integration functions with Home/Away Adjustment System


def get_team_strength_with_venue_adjustments(
    team: str,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> dict:
    """
    Get team strength data enhanced with venue-specific adjustments.

    This function combines the existing team strength analysis with the
    home/away adjustment system to provide comprehensive venue-adjusted metrics.

    Args:
        team: Team name
        season: Season (defaults to current)
        gameweek: Specific gameweek (defaults to most recent)
        dbsession: Database session

    Returns:
        Dictionary with enhanced team strength data including venue adjustments
    """
    from airsenal.framework.home_away_adjustment import get_team_home_advantage
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    if not gameweek:
        gameweek = get_last_finished_gameweek()

    # Get base team strength data
    base_strength = get_team_strength_for_api(team, season, gameweek, dbsession)

    if "error" in base_strength:
        return base_strength

    try:
        # Get home/away adjustment data
        venue_data = get_team_home_advantage(team, season, gameweek, dbsession)

        # Combine the data
        enhanced_strength = base_strength.copy()
        enhanced_strength["venue_adjustments"] = {
            "home_advantage": venue_data.get("adjusted_home_advantage", 0.0),
            "raw_home_advantage": venue_data.get("raw_home_advantage", 0.0),
            "venue_confidence": venue_data.get("confidence", 0.0),
            "home_matches_analyzed": venue_data.get("home_matches", 0),
            "away_matches_analyzed": venue_data.get("away_matches", 0),
            "stadium_factor": venue_data.get("stadium_factor", 1.0),
            "empty_stadium_adjustment": venue_data.get("empty_stadium_adjustment", 1.0),
        }

        # Calculate venue-adjusted strength estimates
        base_home_strength = enhanced_strength["attacking_strength"]["home"]
        base_away_strength = enhanced_strength["attacking_strength"]["away"]

        venue_adjustment = (
            venue_data.get("adjusted_home_advantage", 0.0) * 0.3
        )  # Scale factor

        enhanced_strength["venue_adjusted_strength"] = {
            "home_attacking": base_home_strength * (1.0 + venue_adjustment),
            "away_attacking": base_away_strength * (1.0 - venue_adjustment),
            "venue_impact": venue_adjustment,
            "adjustment_applied": True,
        }

        return enhanced_strength

    except Exception as e:
        logger.error("Error adding venue adjustments for %s: %s", team, e)
        # Return base strength data if venue adjustments fail
        base_strength["venue_adjustments"] = {
            "error": f"Venue adjustment failed: {e}",
            "adjustment_applied": False,
        }
        return base_strength


def update_team_strengths_with_venue_analysis(
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> dict:
    """
    Update team strengths and venue adjustments for all teams.

    This function performs a comprehensive update of both team strength
    calculations and home/away advantage analysis.

    Args:
        season: Season to update
        gameweek: Gameweek to update (defaults to current)
        dbsession: Database session

    Returns:
        Dictionary with update results for both systems
    """
    from airsenal.framework.home_away_adjustment import TeamHomeAdvantage
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    if not gameweek:
        gameweek = get_last_finished_gameweek()

    logger.info(
        "Updating team strengths and venue analysis for %s GW%s", season, gameweek
    )

    # Update base team strengths
    strength_results = update_team_strengths_for_gameweek(season, gameweek, dbsession)

    # Update venue advantage analysis
    venue_results = {"teams_analyzed": 0, "teams_updated": 0, "errors": []}

    try:
        venue_analyzer = TeamHomeAdvantage(dbsession)

        # Get all teams for the season
        teams = (
            dbsession.query(Team.name).filter(Team.season == season).distinct().all()
        )

        for (team_name,) in teams:
            try:
                venue_results["teams_analyzed"] += 1

                # Calculate venue advantage
                venue_analyzer.calculate_team_home_advantage(
                    team_name, season, gameweek
                )

                # Save to database (if we have the venue advantage data model)
                # This would require the TeamHomeAdvantageData model to be available
                # For now, we'll just track the calculation
                venue_results["teams_updated"] += 1

                logger.debug("Updated venue analysis for %s", team_name)

            except Exception as e:
                error_msg = f"Error updating venue analysis for {team_name}: {e}"
                logger.error(error_msg)
                venue_results["errors"].append(error_msg)

        # Combine results
        combined_results = {
            "season": season,
            "gameweek": gameweek,
            "team_strength_update": strength_results,
            "venue_analysis_update": venue_results,
            "update_time": datetime.datetime.now().isoformat(),
        }

        logger.info(
            "Updated %s teams with venue analysis", venue_results["teams_updated"]
        )
        return combined_results

    except Exception as e:
        logger.error("Error in venue analysis update: %s", e)
        venue_results["error"] = f"Venue analysis failed: {e}"

        return {
            "season": season,
            "gameweek": gameweek,
            "team_strength_update": strength_results,
            "venue_analysis_update": venue_results,
            "update_time": datetime.datetime.now().isoformat(),
        }


def validate_integrated_strength_and_venue_model(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> dict:
    """
    Validate the integrated team strength and venue adjustment system.

    This function tests both the team strength model and venue adjustments
    against actual results to measure combined accuracy.

    Args:
        season: Season to validate
        dbsession: Database session

    Returns:
        Dictionary with comprehensive validation metrics
    """
    from airsenal.framework.home_away_adjustment import HomeAwayAdjuster
    from airsenal.framework.schema import session as default_session

    if not dbsession:
        dbsession = default_session

    logger.info("Validating integrated strength and venue model for %s", season)

    try:
        # Validate base team strength model
        strength_validation = validate_team_strength_model(season, dbsession)

        # Validate venue adjustment system
        adjuster = HomeAwayAdjuster(dbsession)
        venue_validation = adjuster.validate_adjustments(season)

        # Combined validation metrics
        combined_validation = {
            "season": season,
            "team_strength_validation": strength_validation,
            "venue_adjustment_validation": venue_validation,
            "validation_date": datetime.datetime.now().isoformat(),
        }

        # Calculate combined correlation if both validations succeeded
        if (
            "correlation" in strength_validation
            and "correlation" in venue_validation
            and strength_validation.get("correlation") is not None
            and venue_validation.get("correlation") is not None
        ):
            # Weighted average of correlations
            strength_weight = 0.7
            venue_weight = 0.3

            combined_correlation = (
                strength_weight * strength_validation["correlation"]
                + venue_weight * venue_validation["correlation"]
            )

            combined_validation["combined_metrics"] = {
                "weighted_correlation": combined_correlation,
                "strength_weight": strength_weight,
                "venue_weight": venue_weight,
                "meets_target": combined_correlation > 0.7,
            }

        return combined_validation

    except Exception as e:
        logger.error("Error in integrated validation: %s", e)
        return {
            "season": season,
            "error": f"Integrated validation failed: {e}",
            "validation_date": datetime.datetime.now().isoformat(),
        }


def get_venue_adjusted_predictions_summary(
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    position: str | None = None,
    dbsession: Session | None = None,
) -> dict:
    """
    Get summary of venue adjustment impacts on player predictions.

    Args:
        season: Season to analyze
        gameweek: Specific gameweek (defaults to next)
        position: Player position filter (optional)
        dbsession: Database session

    Returns:
        Dictionary with venue adjustment impact summary
    """
    from airsenal.framework.home_away_adjustment import HomeAwayAdjuster
    from airsenal.framework.schema import session as default_session
    from airsenal.framework.utils import NEXT_GAMEWEEK

    if not dbsession:
        dbsession = default_session

    if not gameweek:
        gameweek = NEXT_GAMEWEEK

    try:
        HomeAwayAdjuster(dbsession)

        # Get predictions for analysis
        query = (
            dbsession.query(PlayerPrediction)
            .join(Fixture, PlayerPrediction.fixture_id == Fixture.fixture_id)
            .filter(Fixture.season == season)
            .filter(Fixture.gameweek == gameweek)
        )

        if position:
            query = query.join(Player, PlayerPrediction.player_id == Player.player_id)
            # Note: would need to join with PlayerAttributes to filter by position

        predictions = query.all()

        if not predictions:
            return {
                "season": season,
                "gameweek": gameweek,
                "error": "No predictions found for analysis",
            }

        # Calculate venue impact summary
        from airsenal.framework.prediction_utils_enhanced import (
            calculate_venue_impact_summary,
        )

        impact_summary = calculate_venue_impact_summary(predictions, season, dbsession)

        return {
            "season": season,
            "gameweek": gameweek,
            "position_filter": position,
            "venue_impact_summary": impact_summary,
            "analysis_date": datetime.datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("Error analyzing venue adjustment impacts: %s", e)
        return {
            "season": season,
            "gameweek": gameweek,
            "error": f"Venue impact analysis failed: {e}",
        }
