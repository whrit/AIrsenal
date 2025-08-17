# Historical Data Backfilling for AIrsenal

This document provides comprehensive documentation for the historical xG/xA data backfilling system implemented for AIrsenal.

## Overview

The historical data backfilling system allows AIrsenal to retroactively populate xG (Expected Goals) and xA (Expected Assists) data for past seasons, enabling more accurate historical analysis and improved prediction models.

### Key Features

- **Season-wide backfilling** for complete historical data coverage
- **Incremental backfilling** for recent data updates
- **Parallel processing** with configurable batch sizes for speed optimization
- **Resume capability** for interrupted operations
- **Comprehensive data validation** and quality assessment
- **Progress tracking** with real-time monitoring
- **Reconciliation reporting** to identify data gaps
- **Rate limiting** and API throttling for external providers

## Architecture

### Core Components

1. **`HistoricalDataBackfiller`** - Main orchestrator for backfill operations
2. **`BackfillValidator`** - Comprehensive data quality validation
3. **`BackfillProgressTracker`** - Progress monitoring and persistence
4. **`BackfillResult`** - Standardized result reporting
5. **Integration with existing `XGDataManager`** - Leverages existing data provider infrastructure

### Data Flow

```
External Provider (Sportmonks) → XGDataManager → BackfillValidator → Database
                                      ↓
                           HistoricalDataBackfiller
                                      ↓
                           Progress Tracking & Reporting
```

## Installation and Setup

### Prerequisites

1. **Valid API Key** - Sportmonks API key required for data access
2. **Database Schema** - Ensure database has the required xG/xA columns
3. **Python Dependencies** - All required packages installed via `uv sync`

### Configuration

Set your Sportmonks API key:
```bash
# Using environment variable
export SPORTMONKS_API_KEY="your_api_key_here"

# Or using AIrsenal's env system
airsenal_env set -k SPORTMONKS_API_KEY -v "your_api_key_here"
```

## Usage

### Command Line Interface

The backfill system is accessed through the `backfill_xg_data.py` script:

#### Complete Season Backfill

Backfill entire seasons (recommended approach for initial setup):

```bash
# Backfill specific seasons
python airsenal/scripts/backfill_xg_data.py --seasons 2122 2223 2324

# Backfill with custom configuration
python airsenal/scripts/backfill_xg_data.py --seasons 2122 2223 \
  --batch-size 25 --max-workers 2 --verbose
```

#### Specific Gameweek Backfill

Backfill targeted gameweeks:

```bash
# Backfill specific gameweeks
python airsenal/scripts/backfill_xg_data.py --season 2324 --gameweeks 1 2 3

# Backfill single gameweek
python airsenal/scripts/backfill_xg_data.py --season 2324 --gameweeks 15
```

#### Incremental Backfill

Update recent data (recommended for ongoing maintenance):

```bash
# Backfill last 7 days (default)
python airsenal/scripts/backfill_xg_data.py --season 2324 --incremental

# Backfill last 14 days
python airsenal/scripts/backfill_xg_data.py --season 2324 --incremental --incremental-days 14
```

#### Data Analysis and Validation

```bash
# Generate missing data report
python airsenal/scripts/backfill_xg_data.py --season 2324 --missing-data-report

# Run validation only
python airsenal/scripts/backfill_xg_data.py --season 2324 --validate-only

# Validate specific gameweeks
python airsenal/scripts/backfill_xg_data.py --season 2324 --gameweeks 1 5 10 --validate-only
```

#### Resume Interrupted Operations

```bash
# Resume from session ID (displayed when operation starts)
python airsenal/scripts/backfill_xg_data.py --resume season_2324_1641234567
```

#### Dry Run Mode

Test operations without making changes:

```bash
# See what would be done
python airsenal/scripts/backfill_xg_data.py --seasons 2122 2223 --dry-run
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Number of gameweeks processed in parallel | 50 |
| `--max-workers` | Maximum parallel worker threads | 4 |
| `--retry-attempts` | Number of retry attempts for failures | 3 |
| `--incremental-days` | Days back for incremental backfill | 7 |
| `--output-dir` | Directory for reports and logs | `$AIRSENAL_HOME/backfill_reports` |
| `--verbose` | Enable detailed logging | False |
| `--force` | Force backfill even if data exists | False |

## Programmatic Usage

### Basic Example

```python
from airsenal.framework.data_backfill import HistoricalDataBackfiller
from airsenal.framework.xg_data_provider import create_xg_manager

# Initialize components
xg_manager = create_xg_manager()
backfiller = HistoricalDataBackfiller(xg_manager)

# Configure for your needs
backfiller.configure(
    batch_size=25,
    max_workers=2,
    retry_attempts=5
)

# Backfill a season
result = backfiller.backfill_season("2324")

if result.status == BackfillStatus.COMPLETED:
    print(f"✅ Season backfill completed!")
    print(f"Processed: {result.progress.completed_items}/{result.progress.total_items}")
else:
    print(f"❌ Season backfill failed with {len(result.validation_issues)} issues")
```

### Advanced Example with Validation

```python
from airsenal.framework.data_backfill import HistoricalDataBackfiller, ValidationSeverity

# Initialize backfiller
backfiller = HistoricalDataBackfiller()

# Run validation first
issues = backfiller.validator.validate_backfilled_data("2324")

# Check for critical issues
critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
if critical_issues:
    print(f"⚠️ Found {len(critical_issues)} critical issues - resolve before backfilling")
    for issue in critical_issues:
        print(f"  - {issue.description}")
else:
    # Proceed with backfill
    result = backfiller.backfill_gameweeks("2324", [1, 2, 3])
```

### Progress Monitoring

```python
from airsenal.framework.data_backfill import BackfillProgressTracker

def progress_callback(progress):
    print(f"Progress: {progress.progress_percentage:.1f}% "
          f"({progress.completed_items}/{progress.total_items})")
    if progress.estimated_completion:
        print(f"Estimated completion: {progress.estimated_completion}")

# Create tracker
tracker = BackfillProgressTracker("my_session", total_items=100)
tracker.add_callback(progress_callback)

# Use tracker in backfill operations
# (automatically handled by HistoricalDataBackfiller)
```

## Data Quality and Validation

### Validation Rules

The system implements comprehensive validation rules:

1. **xG Data Consistency**
   - Validates xG + xA = xGI calculations
   - Checks for reasonable xG values (0 ≤ xG ≤ 5 per match)
   - Identifies unrealistic outliers

2. **Missing Data Detection**
   - Calculates coverage percentages by season/gameweek
   - Identifies gaps requiring priority backfill
   - Tracks data completeness metrics

3. **Statistical Outliers**
   - Flags extremely high xG values for verification
   - Compares against historical distributions
   - Identifies potential data quality issues

4. **Cross-Reference Validation**
   - Validates foreign key relationships
   - Checks for orphaned records
   - Ensures data integrity

5. **Temporal Consistency**
   - Validates fixture dates align with seasons
   - Checks for chronological ordering
   - Identifies date format issues

### Data Quality Metrics

The system tracks several quality metrics:

- **Coverage Percentage** - Proportion of records with xG data
- **Success Rate** - Percentage of successful backfill operations
- **Data Completeness** - Missing data identification by gameweek
- **Validation Scores** - Overall data quality assessment

### Reconciliation Reports

Generated reports include:

```json
{
  "season": "2324",
  "data_coverage": {
    "total_player_scores": 1500,
    "scores_with_xg": 1275,
    "xg_coverage_percentage": 85.0
  },
  "quality_metrics": {
    "average_xg_per_player_per_match": 0.45,
    "maximum_xg_in_match": 2.8
  },
  "gaps_identified": [
    {
      "gameweek": 5,
      "coverage_percentage": 45.0,
      "severity": "high"
    }
  ],
  "recommendations": [
    {
      "priority": "high",
      "action": "targeted_backfill",
      "gameweeks": [5, 12]
    }
  ]
}
```

## Performance Optimization

### Parallel Processing

The system uses ThreadPoolExecutor for parallel processing:

```python
# Optimize for your system
backfiller.configure(
    batch_size=100,    # Larger batches for better API efficiency
    max_workers=8,     # More workers for faster processing
    retry_attempts=5   # More retries for reliability
)
```

### Rate Limiting

Built-in rate limiting prevents API overload:

- Respects provider rate limits (100 requests/minute for Sportmonks Basic)
- Implements exponential backoff for retries
- Circuit breaker pattern for API failures
- Cost tracking for budget management

### Memory Management

- Processes data in configurable batches
- Lazy loading for large datasets
- Automatic cleanup of temporary files
- Progress persistence for resume capability

## Known Issues and Limitations

### API Provider Limitations

1. **Sportmonks Rate Limits**
   - Basic tier: 100 requests/minute
   - Historical data may have delays (12+ hours)
   - Some advanced metrics require higher tiers

2. **Data Availability**
   - xG data generally available from 2019-20 season onwards
   - Earlier seasons may have limited or no xG data
   - Some matches may be missing data due to provider limitations

### Database Schema Requirements

3. **New Column Requirements**
   - Database must be migrated to include new xG/xA columns
   - Existing installations need schema updates
   - Some validation rules require penalty tracking columns

### Performance Considerations

4. **Large Dataset Processing**
   - Full season backfills can take 30-60 minutes
   - API costs can accumulate for large operations
   - Network issues can interrupt long-running operations

### Data Quality Issues

5. **Historical Data Gaps**
   - Postponed matches may have delayed data availability
   - Player transfers can complicate data mapping
   - Some historical seasons have lower data quality

## Troubleshooting

### Common Issues

#### "No xG data providers available"
```bash
# Check API key configuration
airsenal_env get | grep SPORTMONKS

# Set API key if missing
airsenal_env set -k SPORTMONKS_API_KEY -v "your_key"
```

#### "Database column doesn't exist"
```bash
# Run database migrations (if available)
python -c "from airsenal.framework.schema import Base, get_session; Base.metadata.create_all(get_session().bind)"

# Or check if you're using the latest schema
```

#### "Rate limit exceeded"
```bash
# Use lower worker count and smaller batches
python airsenal/scripts/backfill_xg_data.py --seasons 2324 \
  --max-workers 1 --batch-size 10
```

#### "Circuit breaker OPEN"
```bash
# Wait for circuit breaker to reset (60 seconds default)
# Or check API key validity and network connectivity
```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
python airsenal/scripts/backfill_xg_data.py --verbose --seasons 2324
```

### Log Locations

- **Application logs**: Structured JSON logs to stdout
- **Progress files**: `$AIRSENAL_HOME/backfill_progress/`
- **Reports**: `$AIRSENAL_HOME/backfill_reports/` (default)

## Best Practices

### Initial Setup

1. **Start with Recent Data**
   ```bash
   # Test with current season first
   python airsenal/scripts/backfill_xg_data.py --season 2425 --gameweeks 1 --dry-run
   ```

2. **Validate Before Full Backfill**
   ```bash
   # Check existing data quality
   python airsenal/scripts/backfill_xg_data.py --season 2324 --missing-data-report
   ```

3. **Use Progressive Backfill**
   ```bash
   # Start with recent seasons and work backwards
   python airsenal/scripts/backfill_xg_data.py --seasons 2324 2223 2122
   ```

### Ongoing Maintenance

1. **Regular Incremental Updates**
   ```bash
   # Run weekly to catch recent matches
   python airsenal/scripts/backfill_xg_data.py --season 2425 --incremental
   ```

2. **Monitor Data Quality**
   ```bash
   # Monthly data quality reports
   python airsenal/scripts/backfill_xg_data.py --season 2425 --validate-only
   ```

3. **Budget Monitoring**
   ```bash
   # Check API costs regularly
   python airsenal/scripts/sync_xg_data.py --cost-summary
   ```

### Production Deployment

1. **Use Conservative Settings**
   ```python
   backfiller.configure(
       batch_size=25,      # Smaller batches
       max_workers=2,      # Fewer workers
       retry_attempts=5    # More retries
   )
   ```

2. **Implement Monitoring**
   - Set up alerts for failed backfills
   - Monitor API costs and rate limits
   - Track data quality metrics over time

3. **Schedule Regular Operations**
   ```bash
   # Cron job for daily incremental updates
   0 2 * * * cd /path/to/airsenal && python airsenal/scripts/backfill_xg_data.py --incremental
   ```

## API Integration

### Supported Providers

Currently supports:
- **Sportmonks** - Primary provider with comprehensive xG/xA data

Future providers could include:
- FBRef
- Opta
- StatsBomb (if API becomes available)

### Adding New Providers

To add a new provider:

1. Extend `XGDataProvider` abstract base class
2. Implement required methods: `get_player_xg_data`, `get_match_xg_data`, `get_bulk_xg_data`
3. Create provider-specific `DataTransformer` methods
4. Register with `XGDataManager`

See `SportmonksProvider` as a reference implementation.

## Contributing

### Code Organization

```
airsenal/framework/
├── data_backfill.py          # Main backfill classes
├── xg_data_provider.py       # Data provider integration
└── schema.py                 # Database models

airsenal/scripts/
├── backfill_xg_data.py       # CLI interface
└── sync_xg_data.py           # Real-time sync

airsenal/tests/
└── test_data_backfill.py     # Comprehensive tests

scripts/
└── test_backfill_integration.py  # Integration tests
```

### Testing

Run the test suite:

```bash
# Unit tests
python -m pytest airsenal/tests/test_data_backfill.py -v

# Integration tests
python scripts/test_backfill_integration.py

# Full test suite
python -m pytest airsenal/tests/ -k "backfill"
```

### Development Setup

```bash
# Install development dependencies
uv sync --extra dev

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Type checking
mypy airsenal/framework/data_backfill.py
```

## Change Log

### Version 1.0.0 (Initial Implementation)

**Features:**
- Complete season backfilling capability
- Incremental backfill for recent data
- Parallel processing with configurable workers
- Resume capability for interrupted operations
- Comprehensive data validation framework
- Progress tracking and persistence
- Reconciliation reporting
- CLI interface with extensive options

**Supported Seasons:**
- 2021-22 (2122)
- 2022-23 (2223)  
- 2023-24 (2324)
- Current season (auto-detected)

**Database Schema:**
- Extended PlayerScore with xG/xA fields
- Extended PlayerAttributes with form metrics
- Added comprehensive indexing for performance

## Support

For issues and questions:

1. **Check the troubleshooting section** in this document
2. **Run integration tests** to verify setup
3. **Enable verbose logging** for detailed error information
4. **Check API provider status** and rate limits
5. **Review generated reports** for data quality insights

The backfill system is designed to be robust and self-diagnosing, with comprehensive error reporting and suggested actions for most common issues.