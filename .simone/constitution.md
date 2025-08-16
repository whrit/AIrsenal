# AIrsenal Constitution

## Project
**AIrsenal** - AI-powered Fantasy Premier League optimization tool

## Tech Stack
- **Language**: Python 3.10-3.12
- **Framework**: SQLAlchemy, JAX/NumPyro, Flask
- **Database**: SQLite/PostgreSQL
- **Package Manager**: uv

## Structure
- `airsenal/framework/` - Core business logic and models
- `airsenal/scripts/` - CLI entry points
- `airsenal/tests/` - Test suite
- `notebooks/` - Analysis and experimentation

## Essential Commands
```bash
uv sync                      # Install dependencies
uv run airsenal_run_pipeline # Run complete pipeline
uv run pytest                # Run tests
ruff check --fix .          # Lint and fix
ruff format .               # Format code
```

## Critical Rules
1. **ALWAYS use uv for package management** - Never use pip directly
2. Python version must be <4.0 (bpl dependency constraint)
3. Never commit secrets or API keys
4. Set FPL_TEAM_ID environment variable before running