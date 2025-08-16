# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIrsenal is a Fantasy Premier League (FPL) optimization tool that uses machine learning to pick optimal teams and transfers. It combines statistical modeling, genetic algorithms, and FPL API data to make data-driven decisions.

## Development Setup

### Dependencies and Installation
- **Package manager**: `uv` (recommended) or `pip`
- **Python versions**: 3.10-3.12 (requires <4.0 due to bpl dependency)
- **Install with uv**: `uv sync` or `uv sync --extra dev` (for development)
- **Install with pip**: `pip install -e .` or `pip install -e .[dev]` (for development)

### Code Quality Commands
- **Lint and format**: `ruff check --fix .` and `ruff format .`
- **Type checking**: `mypy airsenal/framework airsenal/scripts`
- **Testing**: `pytest` (with coverage: `pytest --cov=airsenal`)
- **Pre-commit setup**: `pre-commit install` (runs ruff and other checks automatically)

## Key Commands

All AIrsenal commands are prefixed with `airsenal_` and can be run with `uv run` prefix if using uv:

### Main Pipeline
- `airsenal_run_pipeline`: Complete pipeline (update DB → predictions → optimization)
- `airsenal_setup_initial_db`: Initialize database with historical data (required first step)
- `airsenal_update_db`: Update database with latest FPL data

### Core Operations
- `airsenal_run_prediction --weeks_ahead 3`: Generate player point predictions
- `airsenal_run_optimization --weeks_ahead 3`: Optimize transfer strategy
- `airsenal_make_squad --num_gameweeks 3`: Build initial squad (pre-season)

### Data and Environment
- `airsenal_env get`: View environment variables and AIRSENAL_HOME location
- `airsenal_env set -k FPL_TEAM_ID -v 123456`: Set environment variables
- `airsenal_check_data`: Run data sanity checks

## Architecture

### Directory Structure
- **`airsenal/framework/`**: Core business logic and models
- **`airsenal/scripts/`**: CLI entry points (all `airsenal_*` commands)
- **`airsenal/tests/`**: Test suite
- **`airsenal/data/`**: Historical FPL data (JSON/CSV files)
- **`notebooks/`**: Jupyter notebooks for analysis and experimentation

### Key Components

#### Database Layer (`schema.py`)
- SQLAlchemy ORM models for Players, Teams, Matches, Predictions, etc.
- Supports SQLite (default) and PostgreSQL
- Database file location: `AIRSENAL_HOME/data.db` or `AIRSENAL_DB_FILE`

#### Data Sources (`data_fetcher.py`)
- `FPLDataFetcher`: Interfaces with FPL API for live data
- Handles authentication for private league data and transfers
- Rate limiting and session management built-in

#### Player and Squad Models (`player.py`, `squad.py`)
- `CandidatePlayer`: Individual player with predictions and attributes
- `Squad`: 15-player FPL squad with formation and transfer logic
- Position constraints: GK(2), DEF(5), MID(5), FWD(3)

#### Optimization Engines
- **`optimization_squad.py`**: Genetic algorithm (DEAP) for full squad building
- **`optimization_transfers.py`**: Transfer optimization with budget constraints
- **`optimization_utils.py`**: Shared optimization utilities and scoring

#### Prediction Models (`player_model.py`, `bpl_interface.py`)
- JAX/NumPyro-based Bayesian models for player performance
- Team-level models using BPL (Bayesian Premier League) package
- Prediction horizon typically 3 gameweeks ahead

### Environment Configuration
Required: `FPL_TEAM_ID` (your team ID from FPL website URL)
Optional: `FPL_LOGIN`, `FPL_PASSWORD` (for private leagues and auto-transfers)
Config stored in `AIRSENAL_HOME` directory (platform-specific)

## Testing

- **Framework**: pytest with coverage reporting
- **CI**: Runs on Python 3.10 & 3.12, Ubuntu & macOS
- **Test database**: Uses `airsenal/tests/testdata/` for fixtures
- **Environment**: Set `FPL_TEAM_ID=742663` for tests

## Development Workflow

1. **Branches**: `main` (stable) and `develop` (latest features)
2. **Feature branches**: `feature/<issue>-description` or `bugfix/<issue>-description`
3. **Code style**: PEP-8, numpydoc docstrings, type hints encouraged
4. **Pre-commit hooks**: Automatically run ruff formatting and linting

## Data Flow

1. **Data Collection**: `airsenal_update_db` fetches latest FPL data via API
2. **Prediction**: `airsenal_run_prediction` generates expected points using ML models
3. **Optimization**: `airsenal_run_optimization` finds optimal transfers using genetic algorithms
4. **Execution**: `airsenal_make_transfers` and `airsenal_set_lineup` apply changes to FPL account

## Activity Logging

You have access to the `log_activity` tool. Use it to record your activities after every activity that is relevant for the project. This helps track development progress and understand what has been done.