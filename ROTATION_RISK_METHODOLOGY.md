# Rotation Risk Calculator Methodology

## Overview

The Rotation Risk Calculator is a machine learning-based system designed to predict the likelihood that a Fantasy Premier League (FPL) player will be rotated (benched or rested) in upcoming matches. The system provides risk scores on a 0-1 scale where 0 indicates a player is very unlikely to be rotated (nailed starter) and 1 indicates very high rotation risk.

The system is designed to achieve >80% accuracy on historical validation data and integrates seamlessly with AIrsenal's existing player availability predictions and transfer optimization systems.

## Architecture

### Core Components

1. **RotationRiskCalculator**: Main prediction engine using ensemble ML models
2. **ManagerPatternAnalyzer**: Analyzes manager-specific rotation behaviors
3. **FixtureCongestionAnalyzer**: Evaluates fixture scheduling impact on rotation
4. **Database Models**: Store predictions, patterns, and congestion data
5. **API Endpoints**: Provide access to rotation risk data
6. **Validation Framework**: Ensures model accuracy meets requirements

### Model Architecture

The system uses an ensemble approach combining:

- **Random Forest Classifier**: Captures non-linear relationships and feature interactions
- **Logistic Regression**: Provides interpretable baseline predictions
- **Weighted Ensemble**: Combines predictions (60% Random Forest, 40% Logistic Regression)

## Prediction Factors

### Primary Risk Factors

1. **Fixture Congestion (Weight: 25%)**
   - Number of fixtures in 7-day and 14-day windows
   - Travel burden from away fixtures
   - Recovery time since last match
   - Competition overlap periods

2. **Player Fatigue (Weight: 20%)**
   - Minutes played in last 3 matches
   - Cumulative fatigue over recent fixtures
   - Age-adjusted fatigue calculations
   - Position-specific workload analysis

3. **Manager Rotation Patterns (Weight: 15%)**
   - Historical rotation frequency by manager
   - Response to fixture congestion
   - Position-specific preferences
   - Competition prioritization

4. **Player Importance (Weight: -20%, reduces risk)**
   - Recent performance metrics
   - Start frequency in recent matches
   - Point contributions to team
   - Irreplaceability in squad

5. **Competition Importance (Weight: -10%, reduces risk)**
   - Premier League vs Cup competitions
   - Round significance (finals, semifinals)
   - European competition context

6. **Other Factors (Weight: 20%)**
   - Team depth in player's position
   - Age-related rotation tendency
   - Recent injury history
   - International break impact

### Factor Calculation Details

#### Fixture Congestion Score
```
congestion_score = min(1.0, (
    fixtures_7_days * 0.4 +
    fixtures_14_days * 0.2 +
    travel_burden_normalized * 0.3 +
    recovery_time_factor * 0.1
))
```

#### Fatigue Risk Score
```
fatigue_risk = min(1.0, total_recent_minutes / 270.0)
if avg_minutes > 80:
    fatigue_risk *= 1.2  # Penalty for consistently playing full matches
```

#### Player Importance Score
```
importance = (start_rate * 0.7) + min(1.0, avg_points / 6.0) * 0.3
```

## Manager Pattern Analysis

### Pattern Detection

The system analyzes historical data to identify manager-specific rotation patterns:

1. **Overall Rotation Rate**: Frequency of lineup changes between matches
2. **Congestion Response**: Increased rotation during fixture congestion
3. **Position Preferences**: Rotation rates by position (GK, DEF, MID, FWD)
4. **Competition Priorities**: Team selection strength by competition type

### Pattern Storage

Manager patterns are stored in the `ManagerRotationPattern` database model with fields for:
- Team and season context
- Rotation rates by position
- Competition-specific priorities
- Behavioral indicators (age bias, youth integration)
- Analysis metadata and quality scores

## Integration Points

### 1. AIrsenal Player Model Integration

The rotation risk system integrates with existing player models:

```python
from airsenal.framework.rotation_risk import get_rotation_risk_for_player

# Get rotation risk for a player
risk_data = get_rotation_risk_for_player(player_id=123, gameweek=15)
rotation_risk = risk_data['rotation_risk']  # 0.0 - 1.0 scale
confidence = risk_data['confidence']        # Model confidence
```

### 2. Transfer Optimization Integration

Rotation risk can be incorporated into transfer decisions:

```python
from airsenal.framework.rotation_risk import get_team_rotation_risks

# Get risks for entire team
team_risks = get_team_rotation_risks("ARS", gameweek=15)
high_risk_players = team_risks['high_risk_players']  # Risk > 0.6

# Filter out high rotation risk players from transfer targets
safe_targets = [p for p in potential_transfers 
                if get_rotation_risk_for_player(p.id)['rotation_risk'] < 0.4]
```

### 3. Squad Selection Integration

For optimal squad selection considering rotation risk:

```python
from airsenal.framework.rotation_risk import RotationRiskCalculator

calculator = RotationRiskCalculator()

# Calculate risks for multiple players
player_ids = [1, 2, 3, 4, 5]
risks = calculator.calculate_batch_rotation_risks(player_ids, gameweek=15)

# Weight expected points by rotation risk
for player_id, risk_data in risks.items():
    expected_points = get_predicted_points(player_id)
    rotation_risk = risk_data['rotation_risk']
    risk_adjusted_points = expected_points * (1 - rotation_risk)
```

## API Usage

### Individual Player Risk

```bash
GET /rotation_risk/player/123?gameweek=15&season=2023-24
```

Response:
```json
{
  "player_id": 123,
  "gameweek": 15,
  "season": "2023-24",
  "rotation_risk": 0.35,
  "confidence": 0.82,
  "risk_level": "Moderate",
  "factors": {
    "fixture_congestion": 0.4,
    "fatigue_risk": 0.3,
    "player_importance": 0.7
  },
  "model_version": "1.0.0"
}
```

### Team Rotation Risks

```bash
GET /rotation_risk/team/ARS?gameweek=15
```

Response:
```json
{
  "team": "ARS",
  "gameweek": 15,
  "player_count": 15,
  "avg_rotation_risk": 0.42,
  "high_risk_players": [
    {
      "player_id": 456,
      "player_name": "Player Name",
      "rotation_risk": 0.78,
      "risk_level": "High"
    }
  ],
  "players": [...]
}
```

### High Risk Players Across League

```bash
GET /rotation_risk/high_risk?threshold=0.6&limit=20
```

### Manager Patterns

```bash
GET /rotation_risk/manager_patterns/ARS?season=2023-24
```

### Fixture Congestion

```bash
GET /rotation_risk/fixture_congestion/ARS?gameweek=15
```

## Database Schema

### RotationRiskPrediction

Stores individual rotation risk predictions:

```sql
CREATE TABLE rotation_risk_prediction (
    id INTEGER PRIMARY KEY,
    player_id INTEGER REFERENCES player(player_id),
    season VARCHAR(100),
    gameweek INTEGER,
    rotation_risk FLOAT,           -- 0-1 scale
    confidence FLOAT,              -- 0-1 scale
    model_version VARCHAR(100),
    -- Risk factor breakdown
    fixture_congestion FLOAT,
    fatigue_risk FLOAT,
    age_factor FLOAT,
    manager_rotation_tendency FLOAT,
    position_rotation_rate FLOAT,
    player_importance FLOAT,
    competition_importance FLOAT,
    team_depth FLOAT,
    recovery_time FLOAT,
    -- Metadata
    predicted_at VARCHAR(100),     -- ISO datetime
    data_quality_score FLOAT,
    calculation_method VARCHAR(100),
    -- Validation
    actual_outcome BOOLEAN,        -- True=rotated, False=started
    prediction_accuracy FLOAT,
    validated_at VARCHAR(100)
);
```

### ManagerRotationPattern

Stores manager-specific rotation behaviors:

```sql
CREATE TABLE manager_rotation_pattern (
    id INTEGER PRIMARY KEY,
    team VARCHAR(100),
    season VARCHAR(100),
    manager_name VARCHAR(100),
    -- Overall metrics
    avg_rotation_rate FLOAT,
    congestion_response FLOAT,
    sample_size INTEGER,
    -- Position-specific rates
    gk_rotation_rate FLOAT,
    def_rotation_rate FLOAT,
    mid_rotation_rate FLOAT,
    fwd_rotation_rate FLOAT,
    -- Competition priorities
    premier_league_priority FLOAT DEFAULT 1.0,
    champions_league_priority FLOAT DEFAULT 0.9,
    europa_league_priority FLOAT DEFAULT 0.8,
    fa_cup_priority FLOAT DEFAULT 0.6,
    carabao_cup_priority FLOAT DEFAULT 0.4,
    -- Behavioral indicators
    age_bias FLOAT DEFAULT 0.0,
    youth_integration FLOAT DEFAULT 0.0,
    injury_caution FLOAT DEFAULT 0.5,
    -- Metadata
    analysis_version VARCHAR(100),
    analyzed_at VARCHAR(100),
    data_quality_score FLOAT
);
```

### FixtureCongestion

Stores fixture congestion analysis:

```sql
CREATE TABLE fixture_congestion (
    id INTEGER PRIMARY KEY,
    team VARCHAR(100),
    season VARCHAR(100),
    gameweek INTEGER,
    -- Core metrics
    congestion_score FLOAT,
    fixtures_7_days INTEGER,
    fixtures_14_days INTEGER,
    travel_burden FLOAT,
    recovery_time FLOAT,
    -- Context
    european_competition BOOLEAN DEFAULT FALSE,
    international_break_impact FLOAT DEFAULT 0.0,
    injury_crisis_multiplier FLOAT DEFAULT 1.0,
    -- Metadata
    calculated_at VARCHAR(100),
    calculation_version VARCHAR(100),
    data_quality_score FLOAT
);
```

## Model Training and Validation

### Training Process

1. **Historical Data Extraction**: Extract rotation events from player minutes data
2. **Feature Engineering**: Calculate risk factors for each historical event
3. **Data Splitting**: Temporal split (70% training, 30% testing)
4. **Model Training**: Train Random Forest and Logistic Regression models
5. **Hyperparameter Tuning**: Optimize model parameters using cross-validation
6. **Ensemble Weighting**: Determine optimal weights for model combination

### Validation Requirements

The system must achieve >80% accuracy on historical validation data:

```bash
# Run validation script
python airsenal/scripts/validate_rotation_system.py --min-accuracy 0.8 --verbose
```

### Validation Metrics

- **Primary**: Classification accuracy (>80% required)
- **Secondary**: Precision, recall, F1-score by risk level
- **Robustness**: Cross-validation consistency
- **Segments**: Performance by player type, congestion level, importance

### Continuous Monitoring

The system includes monitoring for:

1. **Prediction Accuracy**: Track actual vs predicted rotation events
2. **Model Drift**: Monitor feature distributions and model performance over time
3. **Data Quality**: Validate input data completeness and consistency
4. **Performance Metrics**: API response times and error rates

## Usage Guidelines

### Best Practices

1. **Use in Combination**: Rotation risk should complement, not replace, other injury/availability data
2. **Consider Confidence**: Lower confidence predictions should be weighted accordingly
3. **Update Frequency**: Recalculate risks after team news, fixture changes, or injuries
4. **Threshold Guidance**:
   - **0.0-0.2**: Very safe (nailed starters)
   - **0.2-0.4**: Low risk (occasional rotation possible)
   - **0.4-0.6**: Moderate risk (regular rotation candidate)
   - **0.6-0.8**: High risk (likely to be rotated)
   - **0.8-1.0**: Very high risk (almost certain rotation)

### Integration Examples

#### Transfer Optimization

```python
def get_transfer_targets(budget, position, max_rotation_risk=0.4):
    """Get transfer targets with acceptable rotation risk."""
    candidates = get_players_by_position_and_budget(position, budget)
    
    safe_candidates = []
    for player in candidates:
        risk_data = get_rotation_risk_for_player(player.id)
        if risk_data['rotation_risk'] <= max_rotation_risk:
            # Weight expected points by rotation risk
            adjusted_value = player.expected_points * (1 - risk_data['rotation_risk'])
            safe_candidates.append({
                'player': player,
                'rotation_risk': risk_data['rotation_risk'],
                'adjusted_value': adjusted_value
            })
    
    return sorted(safe_candidates, key=lambda x: x['adjusted_value'], reverse=True)
```

#### Squad Selection

```python
def select_optimal_squad(players, formation):
    """Select optimal squad considering rotation risk."""
    squad = []
    
    for position in formation:
        position_players = [p for p in players if p.position == position]
        
        # Calculate risk-adjusted scores
        for player in position_players:
            risk_data = get_rotation_risk_for_player(player.id)
            player.risk_adjusted_score = (
                player.expected_points * 
                (1 - risk_data['rotation_risk']) * 
                risk_data['confidence']
            )
        
        # Select best risk-adjusted player for position
        best_player = max(position_players, key=lambda x: x.risk_adjusted_score)
        squad.append(best_player)
    
    return squad
```

#### Captain Selection

```python
def get_captaincy_options(squad, min_confidence=0.7):
    """Get captaincy options with low rotation risk."""
    captain_candidates = []
    
    for player in squad:
        risk_data = get_rotation_risk_for_player(player.id)
        
        if (risk_data['rotation_risk'] < 0.3 and 
            risk_data['confidence'] >= min_confidence):
            captain_candidates.append({
                'player': player,
                'expected_points': player.expected_points,
                'rotation_risk': risk_data['rotation_risk'],
                'safety_score': player.expected_points * (1 - risk_data['rotation_risk'])
            })
    
    return sorted(captain_candidates, key=lambda x: x['safety_score'], reverse=True)
```

## Troubleshooting

### Common Issues

1. **Low Accuracy (<80%)**
   - Check data quality and completeness
   - Verify feature engineering logic
   - Consider expanding training dataset
   - Review manager pattern analysis

2. **High Variance in Predictions**
   - Increase training data size
   - Add regularization to models
   - Review feature selection

3. **Missing Manager Patterns**
   - Ensure sufficient historical data (minimum 10 fixtures)
   - Check fixture data completeness
   - Verify team name consistency

4. **API Performance Issues**
   - Implement caching for repeated requests
   - Use batch endpoints for multiple players
   - Consider pre-computing risks for popular players

### Debug Commands

```bash
# Test individual player risk calculation
python -c "
from airsenal.framework.rotation_risk import get_rotation_risk_for_player
print(get_rotation_risk_for_player(123, 15))
"

# Validate system accuracy
python airsenal/scripts/validate_rotation_system.py --verbose

# Test API endpoints
curl "http://localhost:5002/rotation_risk/player/123?gameweek=15"
```

## Performance Considerations

### Computational Complexity

- **Individual Prediction**: O(1) - Fast prediction after model training
- **Batch Prediction**: O(n) - Linear scaling with number of players
- **Manager Analysis**: O(m) - Linear with number of fixtures analyzed
- **Congestion Analysis**: O(f) - Linear with number of nearby fixtures

### Optimization Strategies

1. **Caching**: Cache frequent calculations (manager patterns, congestion scores)
2. **Batch Processing**: Use batch APIs for multiple players
3. **Lazy Loading**: Only calculate detailed factors when needed
4. **Database Indexing**: Proper indexes on player_id, season, gameweek

### Memory Usage

- **Model Storage**: ~50MB for trained ML models
- **Pattern Cache**: ~10MB for manager patterns (all teams)
- **Prediction Cache**: ~1MB per gameweek of predictions

## Future Enhancements

### Planned Improvements

1. **Enhanced Features**
   - Player age and injury history integration
   - Weather conditions impact
   - International break fatigue modeling
   - Social media sentiment analysis

2. **Model Improvements**
   - Deep learning models (LSTM for sequence prediction)
   - Transfer learning between seasons
   - Ensemble of specialized models by position

3. **Real-time Updates**
   - Live team news integration
   - Automated fixture congestion updates
   - Dynamic threshold adjustment

4. **Advanced Analytics**
   - Uncertainty quantification
   - Counterfactual analysis ("what if" scenarios)
   - Multi-gameweek rotation prediction

### Contributing

To contribute to the rotation risk system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Ensure validation accuracy remains >80%
4. Update documentation for API changes
5. Run the full test suite before submitting changes

## Conclusion

The Rotation Risk Calculator provides a sophisticated, ML-based approach to predicting player rotation in Fantasy Premier League. By considering multiple factors including fixture congestion, player fatigue, manager patterns, and player importance, the system delivers accurate predictions that enhance transfer and squad selection decisions.

The system's integration with AIrsenal's existing infrastructure makes it a powerful tool for optimizing FPL performance while managing the inherent uncertainty of player rotation decisions.