# Team Strength Analyzer - Bayesian Methodology Documentation

## Overview

The Team Strength Analyzer implements a sophisticated Bayesian hierarchical model to estimate attacking and defensive capabilities of football teams. This document outlines the theoretical foundation, implementation details, and validation framework for the system.

## Theoretical Foundation

### Bayesian Hierarchical Model Structure

The team strength model follows a three-level hierarchy:

1. **League Level**: Global parameters that capture overall league characteristics
2. **Team Level**: Individual team capabilities relative to the league
3. **Match Level**: Specific game outcomes that update our beliefs

### Mathematical Formulation

#### League-Level Priors

```
μ_att ~ Normal(0, σ_league)     # League average attacking strength
μ_def ~ Normal(0, σ_league)     # League average defensive strength
σ_att ~ HalfNormal(0.3)         # Between-team attacking variance
σ_def ~ HalfNormal(0.3)         # Between-team defensive variance
home_adv ~ Normal(0.3, 0.1)     # Home advantage effect
```

Where:
- `σ_league = 0.5` represents the prior belief about overall league strength variability
- Home advantage is set with mean 0.3 goals based on historical analysis

#### Team-Level Parameters

```
att_i ~ Normal(μ_att, σ_att)    # Team i attacking strength
def_i ~ Normal(μ_def, σ_def)    # Team i defensive strength
```

These parameters represent the latent attacking and defensive abilities of each team, drawn from league-wide distributions.

#### Match-Level Likelihood

```
λ_home = exp(att_home - def_away + home_adv)
λ_away = exp(att_away - def_home)
goals_home ~ Poisson(λ_home)
goals_away ~ Poisson(λ_away)
```

The Poisson distribution models goal scoring, with rates determined by the difference between attacking and defensive strengths.

### Home Advantage Modeling

Home advantage is modeled as an additive effect that boosts the home team's attacking strength. The model learns this effect from data rather than assuming a fixed value.

### Temporal Updates with Exponential Smoothing

To incorporate recent form, the model uses exponential smoothing with parameter α ∈ [0.1, 0.3]:

```
strength_new = α × recent_performance + (1-α) × strength_old
```

This ensures recent matches have higher influence while maintaining stability.

## Implementation Details

### MCMC Inference

The model uses No-U-Turn Sampler (NUTS) for efficient exploration of the posterior distribution:

- **Samples**: 2000 post-warmup samples
- **Warmup**: 1000 samples for adaptation
- **Chains**: Single chain for efficiency
- **Diagnostics**: R-hat and effective sample size monitoring

### Convergence Diagnostics

- **R-hat < 1.1**: Indicates convergence across chains
- **ESS > 400**: Minimum effective sample size for reliable inference
- **Trace plots**: Visual inspection of sampling behavior

### Uncertainty Quantification

The Bayesian approach naturally provides uncertainty estimates:

- **Posterior means**: Point estimates of team strengths
- **Posterior standard deviations**: Uncertainty in strength estimates
- **Prediction intervals**: Confidence bounds for future performance

## Feature Engineering

### Component Metrics

The system combines multiple performance indicators:

#### Attacking Metrics
- **Expected Goals (xG)**: Primary indicator (40% weight)
- **Actual Goals**: Secondary validation (30% weight)
- **Shots per Game**: Volume indicator (20% weight)
- **Shot Accuracy**: Efficiency measure (10% weight)

#### Defensive Metrics
- **Expected Goals Against (xGA)**: Primary indicator (40% weight)
- **Actual Goals Conceded**: Secondary validation (30% weight)
- **Shots Against**: Pressure indicator (20% weight)
- **Clean Sheet Probability**: Success measure (10% weight)

### Form and Momentum

Recent performance indicators include:

- **Form Weighted Strength**: Points-based recent performance
- **Performance Trend**: Linear trend in goal difference
- **Momentum Factor**: Acceleration in recent results

## Validation Framework

### Correlation Requirements

The system must achieve **Pearson correlation > 0.7** with actual match results to ensure predictive validity.

### Validation Metrics

1. **Pearson Correlation**: Linear relationship with outcomes
2. **Spearman Correlation**: Rank-order relationship
3. **Prediction Accuracy**: Correct winner prediction rate
4. **Mean Absolute Error**: Average prediction error
5. **Root Mean Square Error**: Penalty for large errors

### Validation Process

```python
def validate_predictions(season):
    predictions = []
    actuals = []
    
    for match in completed_matches:
        # Get pre-match team strengths
        home_strength = get_team_strength(home_team, pre_match_gameweek)
        away_strength = get_team_strength(away_team, pre_match_gameweek)
        
        # Calculate predicted goal difference
        predicted_diff = (home_strength.overall_home - 
                         away_strength.overall_away)
        actual_diff = match.home_score - match.away_score
        
        predictions.append(predicted_diff)
        actuals.append(actual_diff)
    
    correlation = pearsonr(predictions, actuals)[0]
    return correlation > 0.7
```

## Data Quality Considerations

### Minimum Data Requirements

- **Minimum matches**: 5 games for stable estimates
- **Sample size**: At least 10 observations for Bayesian inference
- **Data recency**: Exponential decay with α = 0.2 for temporal weighting

### Quality Scoring

Data quality score (0-1) based on:
- Number of matches available
- Recency of data
- Completeness of statistical metrics
- Absence of major disruptions (manager changes, etc.)

## External Factor Adjustments

### Manager Changes

When a new manager is appointed:
- Apply 10-20% uncertainty increase to strength estimates
- Gradual confidence recovery over 5-8 matches
- Consider tactical style changes in adjustment

### Key Player Injuries

Impact assessment based on:
- Player's contribution to team xG/xGA
- Position importance (goalkeepers weighted higher)
- Expected absence duration
- Availability of replacements

### Transfer Window Effects

- Major signings: Gradual strength increases
- Key departures: Immediate strength adjustments
- Net spend correlation with strength changes

## Model Performance Optimization

### Computational Efficiency

- **Vectorization**: JAX-based operations for GPU acceleration
- **Caching**: Store computed team mappings and preprocessed data
- **Incremental Updates**: Only recompute when new data available

### Memory Management

- Efficient sample storage using compressed arrays
- Periodic cleanup of historical data
- Streaming inference for large datasets

## API Integration

### Endpoint Design

The system provides RESTful endpoints:

```
GET /team_strength/{team}              # Individual team analysis
GET /team_strength                     # All teams comparison
GET /team_strength/compare/{t1}/{t2}   # Head-to-head analysis
GET /team_strength/{team}/trends       # Historical trends
POST /team_strength/update             # Trigger recalculation
GET /team_strength/validate            # Model validation
```

### Response Format

```json
{
  "team": "ARS",
  "season": "2023-24",
  "gameweek": 10,
  "attacking_strength": {
    "home": 1.15,
    "away": 0.92,
    "uncertainty": {
      "home_std": 0.18,
      "away_std": 0.15
    }
  },
  "defensive_strength": {
    "home": 0.85,
    "away": 1.08,
    "uncertainty": {
      "home_std": 0.12,
      "away_std": 0.16
    }
  },
  "overall_strength": {
    "home": 1.08,
    "away": 0.89
  },
  "model_info": {
    "version": "1.0.0",
    "sample_count": 2000,
    "convergence_diagnostic": 1.02,
    "data_quality_score": 0.88
  }
}
```

## Database Schema

### TeamStrength Table

Stores current strength estimates with full uncertainty quantification:

```sql
CREATE TABLE team_strength (
    id INTEGER PRIMARY KEY,
    team VARCHAR(100),
    season VARCHAR(100),
    gameweek INTEGER,
    attacking_strength_home FLOAT,
    attacking_strength_away FLOAT,
    defensive_strength_home FLOAT,
    defensive_strength_away FLOAT,
    attacking_strength_home_std FLOAT,
    attacking_strength_away_std FLOAT,
    defensive_strength_home_std FLOAT,
    defensive_strength_away_std FLOAT,
    -- Additional metrics...
    calculated_at VARCHAR(100)
);
```

### TeamStrengthHistory Table

Maintains historical records for trend analysis:

```sql
CREATE TABLE team_strength_history (
    id INTEGER PRIMARY KEY,
    team VARCHAR(100),
    season VARCHAR(100),
    gameweek INTEGER,
    attacking_strength_home FLOAT,
    attacking_strength_away FLOAT,
    defensive_strength_home FLOAT,
    defensive_strength_away FLOAT,
    attacking_strength_home_change FLOAT,
    attacking_strength_away_change FLOAT,
    defensive_strength_home_change FLOAT,
    defensive_strength_away_change FLOAT,
    calculated_at VARCHAR(100)
);
```

## Usage Examples

### Basic Team Strength Analysis

```python
from airsenal.framework.team_strength import create_team_strength_analyzer

# Initialize analyzer
analyzer = create_team_strength_analyzer(dbsession)

# Calculate strengths for current gameweek
strengths = analyzer.calculate_team_strengths("2023-24", 10)

# Get specific team analysis
arsenal_strength = strengths["ARS"]
print(f"Arsenal attacking strength (home): {arsenal_strength['attacking_strength_home']:.2f}")
```

### API Usage

```bash
# Get team strength
curl "http://localhost:5002/team_strength/ARS?season=2023-24&gameweek=10"

# Compare teams
curl "http://localhost:5002/team_strength/compare/ARS/CHE"

# Get trends
curl "http://localhost:5002/team_strength/ARS/trends?num_gameweeks=5"

# Update all strengths
curl -X POST "http://localhost:5002/team_strength/update"
```

### Model Validation

```python
# Validate model performance
validation_results = analyzer.validate_strength_predictions("2023-24")

if validation_results["correlation_meets_target"]:
    print(f"Model meets correlation target: {validation_results['pearson_correlation']:.3f}")
else:
    print("Model requires retraining or parameter adjustment")
```

## Future Enhancements

### Advanced Modeling

- **Player-level contributions**: Decompose team strength by individual players
- **Tactical formations**: Account for formation-specific strengths
- **Weather conditions**: Environmental factors in strength estimation
- **Referee effects**: Official-specific match dynamics

### Real-time Updates

- **Live match tracking**: Continuous strength updates during games
- **Event-based adjustments**: Red cards, injuries during matches
- **Momentum shifts**: Within-game momentum modeling

### Multi-league Extension

- **Cross-league comparisons**: Strength estimation across different leagues
- **European competition**: Champions League and Europa League integration
- **International football**: National team strength modeling

## Conclusion

The Team Strength Analyzer provides a robust, theoretically grounded approach to estimating team capabilities in football. By combining Bayesian hierarchical modeling with modern computational techniques, it delivers reliable strength estimates with proper uncertainty quantification.

The system's modular design enables easy extension and customization, while the comprehensive validation framework ensures consistent performance meeting the required correlation standards.

For technical support or questions about the methodology, refer to the implementation in `airsenal/framework/team_strength.py` or consult the test suite in `airsenal/tests/test_team_strength.py`.