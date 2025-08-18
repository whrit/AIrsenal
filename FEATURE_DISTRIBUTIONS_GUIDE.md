# AIrsenal Feature Distributions and Expected Ranges Guide

## Overview

This guide documents all feature distributions and expected ranges for the AIrsenal Fantasy Premier League optimization system, as implemented in TASK-113: Feature Validation and Testing Framework. This documentation covers all Sprint 01 features and their statistical properties, validation rules, and quality thresholds.

## Sprint 01 Features Documented

### Form Metrics (TASK-101)
- `form_3_games`: 3-game rolling average
- `form_5_games`: 5-game rolling average  
- `form_10_games`: 10-game rolling average
- `momentum`: Trend-based momentum indicator

### Weighted Performance (TASK-102)
- Position-specific weighted scoring algorithms
- Performance adjustment factors

### Trend Detection (TASK-103)
- Statistical trend analysis indicators
- Time-series pattern recognition

### xG/xA Data (TASK-104)
- `xg_per_90`: Expected goals per 90 minutes
- `xa_per_90`: Expected assists per 90 minutes
- `xgi_per_90`: Expected goal involvement per 90 minutes

### Fixture Difficulty (TASK-107)
- `next_3_fixture_difficulty`: Upcoming 3-game difficulty
- `next_5_fixture_difficulty`: Upcoming 5-game difficulty

### Player Roles (TASK-110)
- `is_penalty_taker`: Primary penalty taker flag
- `is_free_kick_taker`: Free kick specialist flag
- `is_corner_taker`: Corner specialist flag
- `role_confidence`: Confidence in role assignment

### Additional Performance Metrics
- `shots_per_90`: Shots per 90 minutes
- `key_passes_per_90`: Key passes per 90 minutes

---

## Feature Distributions and Expected Ranges

### Form Metrics

#### `form_3_games`
- **Data Type**: Float
- **Range**: 0.0 - 25.0 points
- **Expected Mean**: 2.0 - 8.0 points
- **Expected Standard Deviation**: 1.0 - 5.0 points
- **Distribution**: Normal
- **Null Tolerance**: ≤15%
- **Outlier Threshold**: ≤5%

**Description**: 3-game rolling average of FPL points. Higher variance expected due to smaller sample size.

**Validation Rules**:
- Minimum value ≥ 0.0
- Maximum value ≤ 25.0
- Mean within expected range indicates healthy scoring distribution
- Standard deviation indicates appropriate variability

#### `form_5_games`
- **Data Type**: Float
- **Range**: 0.0 - 25.0 points
- **Expected Mean**: 2.0 - 8.0 points
- **Expected Standard Deviation**: 1.0 - 4.5 points
- **Distribution**: Normal
- **Null Tolerance**: ≤15%
- **Outlier Threshold**: ≤5%

**Description**: 5-game rolling average of FPL points. More stable than 3-game form with reduced variance.

**Validation Rules**:
- Should be strongly correlated with `form_3_games` (r > 0.7)
- Lower standard deviation than `form_3_games`
- Smoother trend progression expected

#### `form_10_games`
- **Data Type**: Float  
- **Range**: 0.0 - 25.0 points
- **Expected Mean**: 2.0 - 8.0 points
- **Expected Standard Deviation**: 0.8 - 4.0 points
- **Distribution**: Normal
- **Null Tolerance**: ≤20%
- **Outlier Threshold**: ≤5%

**Description**: 10-game rolling average of FPL points. Most stable form metric with lowest variance.

**Validation Rules**:
- Lowest standard deviation of all form metrics
- Strong correlation with shorter form metrics
- Higher null tolerance due to new players/season start

#### `momentum`
- **Data Type**: Float
- **Range**: -1.0 - 1.0
- **Expected Mean**: -0.1 - 0.1
- **Expected Standard Deviation**: 0.2 - 0.6
- **Distribution**: Normal (centered around 0)
- **Null Tolerance**: ≤20%
- **Outlier Threshold**: ≤5%

**Description**: Trend-based momentum indicator. Positive values indicate improving form, negative indicate declining form.

**Validation Rules**:
- Bounded between -1 and 1
- Should be approximately centered around 0 across all players
- Low correlation with absolute form values expected

### xG/xA Metrics

#### `xg_per_90`
- **Data Type**: Float
- **Range**: 0.0 - 1.5
- **Expected Mean**: 0.1 - 0.6
- **Expected Standard Deviation**: 0.1 - 0.4
- **Distribution**: Exponential (right-skewed)
- **Null Tolerance**: ≤10%
- **Outlier Threshold**: ≤5%

**Description**: Expected goals per 90 minutes played. Follows exponential distribution with many low values and few high values.

**Validation Rules**:
- Strong correlation with position (forwards > midfielders > defenders)
- Should correlate with shots_per_90
- Exponential distribution expected (positive skew)

#### `xa_per_90`
- **Data Type**: Float
- **Range**: 0.0 - 1.0
- **Expected Mean**: 0.05 - 0.4
- **Expected Standard Deviation**: 0.05 - 0.3
- **Distribution**: Exponential (right-skewed)
- **Null Tolerance**: ≤10%
- **Outlier Threshold**: ≤5%

**Description**: Expected assists per 90 minutes played. Generally lower values than xG, concentrated in creative players.

**Validation Rules**:
- Strong correlation with position (midfielders > forwards > defenders)
- Should correlate with key_passes_per_90
- Lower values than xg_per_90 on average

#### `xgi_per_90`
- **Data Type**: Float
- **Range**: 0.0 - 2.0
- **Expected Mean**: 0.15 - 0.8
- **Expected Standard Deviation**: 0.1 - 0.5
- **Distribution**: Exponential (right-skewed)
- **Null Tolerance**: ≤10%
- **Outlier Threshold**: ≤5%

**Description**: Expected goal involvement (xG + xA) per 90 minutes. Highest values among xG metrics.

**Validation Rules**:
- Should equal approximately xg_per_90 + xa_per_90
- Strong correlation with attacking output
- Position-dependent distribution patterns

### Fixture Difficulty

#### `next_3_fixture_difficulty`
- **Data Type**: Float
- **Range**: 1.0 - 5.0 (FPL difficulty scale)
- **Expected Mean**: 2.5 - 3.5
- **Expected Standard Deviation**: 0.5 - 1.2
- **Distribution**: Normal
- **Null Tolerance**: ≤5%
- **Outlier Threshold**: ≤5%

**Description**: Average difficulty rating for next 3 fixtures on FPL 1-5 scale.

**Validation Rules**:
- Should be normally distributed around league average (3.0)
- Strong correlation with next_5_fixture_difficulty
- Low null tolerance as fixtures are always known

#### `next_5_fixture_difficulty`
- **Data Type**: Float
- **Range**: 1.0 - 5.0 (FPL difficulty scale)
- **Expected Mean**: 2.5 - 3.5
- **Expected Standard Deviation**: 0.4 - 1.0
- **Distribution**: Normal
- **Null Tolerance**: ≤5%
- **Outlier Threshold**: ≤5%

**Description**: Average difficulty rating for next 5 fixtures. More stable than 3-fixture version.

**Validation Rules**:
- Lower standard deviation than next_3_fixture_difficulty
- Converges toward league mean (3.0) with larger sample
- Should correlate with short-term difficulty but be more stable

### Player Roles

#### `is_penalty_taker`
- **Data Type**: Boolean (0/1)
- **Range**: 0.0 - 1.0
- **Expected Mean**: 0.02 - 0.08 (2-8% of players)
- **Distribution**: Beta (highly skewed toward 0)
- **Null Tolerance**: 0%
- **Outlier Threshold**: N/A (binary)

**Description**: Binary flag indicating primary penalty taker status.

**Validation Rules**:
- Only ~3-5% of players should be penalty takers
- Should be mutually exclusive per team (mostly)
- No null values allowed

#### `is_free_kick_taker`
- **Data Type**: Boolean (0/1)
- **Range**: 0.0 - 1.0
- **Expected Mean**: 0.05 - 0.15 (5-15% of players)
- **Distribution**: Beta (skewed toward 0)
- **Null Tolerance**: 0%
- **Outlier Threshold**: N/A (binary)

**Description**: Binary flag indicating free kick specialist status.

**Validation Rules**:
- ~8-12% of players should be free kick takers
- Can have multiple per team
- Should correlate with attacking midfield positions

#### `is_corner_taker`
- **Data Type**: Boolean (0/1)
- **Range**: 0.0 - 1.0
- **Expected Mean**: 0.08 - 0.20 (8-20% of players)
- **Distribution**: Beta (skewed toward 0)
- **Null Tolerance**: 0%
- **Outlier Threshold**: N/A (binary)

**Description**: Binary flag indicating corner specialist status.

**Validation Rules**:
- ~12-18% of players should be corner takers
- Usually 1-3 players per team
- Often correlates with midfield positions and crossing ability

#### `role_confidence`
- **Data Type**: Float
- **Range**: 0.0 - 1.0
- **Expected Mean**: 0.3 - 0.8
- **Expected Standard Deviation**: 0.2 - 0.4
- **Distribution**: Beta
- **Null Tolerance**: ≤30%
- **Outlier Threshold**: ≤5%

**Description**: Confidence score for role assignments based on analysis strength.

**Validation Rules**:
- Higher values for well-established role assignments
- Lower values for uncertain or recently changed roles
- Should correlate with data availability and consistency

### Performance Metrics

#### `shots_per_90`
- **Data Type**: Float
- **Range**: 0.0 - 8.0
- **Expected Mean**: 0.5 - 3.0
- **Expected Standard Deviation**: 0.5 - 2.0
- **Distribution**: Exponential (right-skewed)
- **Null Tolerance**: ≤20%
- **Outlier Threshold**: ≤5%

**Description**: Shots taken per 90 minutes played.

**Validation Rules**:
- Strong correlation with xG metrics
- Position-dependent (forwards > midfielders > defenders)
- Should follow exponential distribution

#### `key_passes_per_90`
- **Data Type**: Float
- **Range**: 0.0 - 8.0
- **Expected Mean**: 0.3 - 2.5
- **Expected Standard Deviation**: 0.3 - 1.5
- **Distribution**: Exponential (right-skewed)
- **Null Tolerance**: ≤20%
- **Outlier Threshold**: ≤5%

**Description**: Key passes (passes leading to shots) per 90 minutes played.

**Validation Rules**:
- Strong correlation with xA metrics
- Creative midfielders should have highest values
- Should follow exponential distribution

---

## Data Quality Thresholds

### Overall Quality Score Calculation

The overall quality score is calculated as a weighted combination of:
- **Null Percentage** (25% weight): Heavy penalty for missing data
- **Outlier Percentage** (35% weight): Penalty for data quality issues
- **Distribution Conformity** (25% weight): Bonus for expected statistical properties
- **Range Compliance** (15% weight): Penalty for out-of-range values

**Quality Score Interpretation**:
- **0.9 - 1.0**: Excellent - Production ready
- **0.8 - 0.9**: Good - Minor issues to address
- **0.7 - 0.8**: Acceptable - Some quality concerns
- **0.6 - 0.7**: Poor - Significant issues present
- **< 0.6**: Critical - Requires immediate attention

### Performance Benchmarks

#### Speed Requirements (Sprint 01 Targets)
- **Form calculation**: <50ms per player
- **Weighted metrics**: <50ms per player
- **xG integration**: <500ms API response
- **Batch operations**: <2s for 650 players
- **Trend detection**: <100ms per player
- **Fixture difficulty**: <200ms calculation
- **Rotation risk**: <75ms per player

#### Accuracy Requirements
- **Validation pass rate**: ≥95%
- **Feature correlation**: Expected relationships maintained
- **Distribution tests**: Statistical significance ≥95%
- **Outlier detection**: ≤5% false positives
- **Null handling**: Graceful degradation

---

## Correlation Matrix Documentation

### Expected Strong Correlations (r > 0.7)
- `form_3_games` ↔ `form_5_games`
- `form_5_games` ↔ `form_10_games`
- `xg_per_90` ↔ `shots_per_90`
- `xa_per_90` ↔ `key_passes_per_90`
- `xg_per_90` ↔ `xgi_per_90`
- `xa_per_90` ↔ `xgi_per_90`

### Expected Moderate Correlations (0.4 < r < 0.7)
- `form_3_games` ↔ `form_10_games`
- `next_3_fixture_difficulty` ↔ `next_5_fixture_difficulty`
- `xg_per_90` ↔ `form_5_games`
- `shots_per_90` ↔ `form_metrics`

### Expected Low/No Correlations (r < 0.4)
- `momentum` ↔ `form_levels` (absolute values)
- `role_flags` ↔ `performance_metrics` (position-dependent)
- `fixture_difficulty` ↔ `player_performance`

### Multicollinearity Warnings
- **High Risk**: Form metrics (3/5/10 games) - consider principal components
- **Medium Risk**: xG metrics (xG, xA, xGI) - monitor condition number
- **Low Risk**: Role flags - binary nature limits multicollinearity

---

## Seasonal Drift Monitoring

### Population Stability Index (PSI) Thresholds
- **PSI < 0.1**: No drift - stable feature
- **0.1 ≤ PSI < 0.2**: Moderate drift - monitor closely
- **PSI ≥ 0.2**: Significant drift - requires recalibration

### Expected Seasonal Changes
- **Form metrics**: Natural variation with player transfers/form changes
- **xG metrics**: Relatively stable, minor tactical evolution
- **Fixture difficulty**: Stable by design (opponent strength based)
- **Role assignments**: Moderate changes with transfers/injuries

### Drift Detection Schedule
- **Real-time**: Monitor during active season
- **End-of-season**: Full recalibration assessment
- **Pre-season**: Baseline establishment for new season
- **Transfer windows**: Increased monitoring due to squad changes

---

## Validation Testing Procedures

### Automated Tests (Run Daily)
1. **Range validation**: All features within expected bounds
2. **Null percentage**: Below tolerance thresholds
3. **Distribution tests**: Statistical conformity checks
4. **Correlation monitoring**: Relationship stability
5. **Performance benchmarks**: Speed requirement compliance

### Weekly Analysis
1. **Quality score trends**: Track data quality over time
2. **Outlier analysis**: Investigate unusual patterns
3. **Correlation drift**: Monitor relationship changes
4. **Feature importance**: Assess predictive value changes

### Monthly Reviews
1. **Full validation report**: Comprehensive analysis
2. **Threshold adjustment**: Update based on seasonal patterns
3. **Model performance**: Validate against actual outcomes
4. **Documentation updates**: Maintain accuracy of expectations

---

## Troubleshooting Guide

### Common Issues and Solutions

#### High Null Percentage
- **Cause**: Data collection failures, new players, missing API data
- **Solution**: Implement graceful fallbacks, historical averages
- **Threshold**: Investigate if >15% for most features

#### Outlier Detection Alerts
- **Cause**: Exceptional performances, data errors, calculation bugs
- **Solution**: Manual verification, outlier analysis, data validation
- **Action**: Review top 1% of values for each feature

#### Distribution Drift
- **Cause**: Tactical changes, rule changes, seasonal variation
- **Solution**: Gradual threshold adjustment, model retraining
- **Monitoring**: Weekly PSI calculation and trend analysis

#### Performance Degradation
- **Cause**: Database growth, complex calculations, inefficient queries
- **Solution**: Query optimization, caching, batch processing
- **Target**: Maintain <50ms per player for core features

#### Correlation Breakdown
- **Cause**: Tactical evolution, data quality issues, model drift
- **Solution**: Feature engineering review, relationship reanalysis
- **Alert**: Investigate if expected correlations drop >20%

---

## Integration with AIrsenal System

### Database Schema Integration
- All features stored in `PlayerAttributes` table
- Validation results in `FeatureValidation` tables
- Historical tracking in time-series tables
- Quality scores in `FeatureQuality` table

### Pipeline Integration
- **Data ingestion**: Automatic validation on new data
- **Feature calculation**: Real-time quality monitoring
- **Model training**: Pre-training validation checks
- **Prediction serving**: Runtime validation guards

### Alerting and Monitoring
- **Critical alerts**: >5% feature failures, performance >2x targets
- **Warning alerts**: Quality score <0.8, drift PSI >0.15
- **Info alerts**: Weekly quality reports, monthly trend analysis

---

## Version History and Changes

### Version 1.0.0 (Current)
- Initial implementation for Sprint 01 features
- Baseline expectations established from historical analysis
- Complete validation framework implementation
- Performance benchmarks defined and tested

### Future Enhancements
- **v1.1.0**: Enhanced drift detection algorithms
- **v1.2.0**: Automated threshold adjustment
- **v1.3.0**: Machine learning-based quality scoring
- **v2.0.0**: Real-time validation streaming

---

This documentation serves as the definitive guide for understanding, validating, and maintaining feature quality in the AIrsenal system. Regular updates ensure accuracy and relevance as the system evolves.