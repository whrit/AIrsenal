# AIrsenal Architecture

## System Overview

AIrsenal is a machine learning-powered system for optimizing Fantasy Premier League (FPL) team selection and transfers. It combines statistical modeling, genetic algorithms, and real-time FPL API data to make data-driven decisions.

## Core Components

### 1. Data Layer (`airsenal/framework/`)

#### Database Schema (`schema.py`)
- **ORM**: SQLAlchemy 2.0+
- **Models**: Player, Team, Match, PlayerScore, PredictedScore, Squad, Transaction
- **Storage**: SQLite (default) or PostgreSQL
- **Location**: `$AIRSENAL_HOME/data.db`

#### Data Fetching (`data_fetcher.py`)
- **FPLDataFetcher**: Primary interface to FPL API
  - Handles authentication for private leagues
  - Rate limiting and session management
  - Caches API responses
- **Historical Data**: JSON/CSV files in `airsenal/data/`
  - Player histories, fixtures, team ratings
  - Absence/injury records

### 2. Prediction Engine

#### Player Models (`player_model.py`, `bpl_interface.py`)
- **Technology**: JAX/NumPyro Bayesian models
- **Approach**: Hierarchical Bayesian modeling
- **Features**:
  - Individual player performance prediction
  - Team-level effects via BPL package
  - Injury/absence probability modeling
  - Form and momentum tracking

#### Team Models
- **BPL Integration**: Bayesian Premier League models
- **Match Predictors**:
  - ExtendedDixonColesMatchPredictor
  - NeutralDixonColesMatchPredictor
  - RandomMatchPredictor (baseline)

### 3. Optimization Engine

#### Squad Building (`optimization_squad.py`)
- **Algorithm**: Genetic algorithm (DEAP library)
- **Constraints**:
  - Budget: £100M total
  - Formation: 2 GK, 5 DEF, 5 MID, 3 FWD
  - Max 3 players per team
- **Objective**: Maximize expected points over N gameweeks

#### Transfer Optimization (`optimization_transfers.py`)
- **Strategy**: Dynamic programming with constraints
- **Considerations**:
  - Transfer costs (4 points per extra transfer)
  - Budget constraints
  - Future gameweek planning
  - Wildcard/chips usage

#### Optimization Utilities (`optimization_utils.py`)
- Shared scoring functions
- Constraint validation
- Solution evaluation

### 4. Models & Entities

#### Player (`player.py`)
- **CandidatePlayer**: Wrapper with predictions
- **Attributes**: Position, team, price, ownership
- **Methods**: Expected points calculation

#### Squad (`squad.py`)
- **Composition**: 15 players with formation
- **Validation**: Budget, position, team constraints
- **Operations**: Transfers, captain selection, bench ordering

### 5. CLI Interface (`airsenal/scripts/`)

#### Main Commands
- `airsenal_run_pipeline`: Complete automation pipeline
- `airsenal_setup_initial_db`: Initialize with historical data
- `airsenal_update_db`: Fetch latest FPL data
- `airsenal_run_prediction`: Generate predictions
- `airsenal_run_optimization`: Find optimal transfers
- `airsenal_make_squad`: Build initial squad

#### Supporting Scripts
- Database population scripts (`fill_*.py`)
- Transfer execution (`make_transfers.py`)
- Lineup setting (`set_lineup.py`)
- Data validation (`data_sanity_checks.py`)

### 6. API Layer (`airsenal/api/`)

- **Framework**: Flask
- **Purpose**: RESTful API for web/mobile clients
- **Sessions**: Squad optimization session management

## Data Flow

```
1. FPL API → FPLDataFetcher → Database
   ↓
2. Database → Prediction Models → PredictedScore table
   ↓
3. PredictedScores → Optimization Engine → TransferSuggestion
   ↓
4. Suggestions → make_transfers/set_lineup → FPL Account
```

## Configuration

### Environment Variables
- `FPL_TEAM_ID`: Your FPL team ID (required)
- `FPL_LOGIN`: FPL email (optional, for transfers)
- `FPL_PASSWORD`: FPL password (optional)
- `AIRSENAL_HOME`: Data directory location
- `AIRSENAL_DB_FILE`: Override database location

### Settings Storage
- Platform-specific via `platformdirs`
- Config in `$AIRSENAL_HOME/` directory

## Testing Architecture

- **Framework**: pytest
- **Coverage**: pytest-cov
- **Test Data**: `airsenal/tests/testdata/`
- **CI/CD**: GitHub Actions (Python 3.10, 3.12)

## Development Workflow

1. **Branches**:
   - `main`: Stable releases
   - `develop`: Active development
   - `feature/*`: New features
   - `bugfix/*`: Bug fixes

2. **Quality Assurance**:
   - Pre-commit hooks (ruff)
   - Type checking (mypy)
   - Unit tests (pytest)
   - Integration tests

## Performance Considerations

- **Multiprocessing**: Configurable thread count for predictions
- **Caching**: API responses and predictions
- **Database Indexing**: Optimized queries for large datasets
- **Batch Processing**: Efficient bulk operations

## Security

- **API Credentials**: Never stored in code
- **Environment Variables**: Sensitive data isolation
- **Session Management**: Secure token handling
- **Input Validation**: SQL injection prevention via ORM