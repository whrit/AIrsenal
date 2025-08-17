# xG/xA Data Provider Integration Guide

This guide covers the integration of external xG (Expected Goals) and xA (Expected Assists) data providers into AIrsenal for enhanced prediction accuracy.

## Overview

The xG/xA data provider integration provides:
- **Flexible Provider Interface**: Abstract base class supporting multiple data sources
- **Sportmonks Integration**: Complete implementation for Sportmonks API
- **Circuit Breaker Pattern**: Automatic failure handling and recovery
- **Rate Limiting**: Respect provider API limits
- **Cost Tracking**: Monitor API usage and costs
- **Health Monitoring**: Real-time provider status monitoring
- **Graceful Degradation**: Handle missing credentials without breaking

## Architecture

### Core Components

```
xg_data_provider.py
├── XGDataProvider (Abstract Base Class)
├── SportmonksProvider (Sportmonks Implementation)  
├── DataTransformer (API Response Conversion)
├── APIHealthMonitor (Status & Cost Tracking)
├── CircuitBreaker (Failure Handling)
└── XGDataManager (High-level Operations)
```

### Database Integration

The integration populates existing PlayerScore table fields:
- `expected_goals`: Player's expected goals for the match
- `expected_assists`: Player's expected assists for the match  
- `expected_goal_involvements`: Sum of xG + xA
- `expected_goals_conceded`: For goalkeepers

## Quick Start

### 1. Install Dependencies

The xG provider integration uses only standard AIrsenal dependencies. For the web dashboard:

```bash
pip install flask  # Optional, for web monitoring dashboard
```

### 2. Configure API Key

Set your Sportmonks API key:

```bash
# Using airsenal_env command
airsenal_env set -k SPORTMONKS_API_KEY -v "your_api_key_here"

# Or set environment variable
export SPORTMONKS_API_KEY="your_api_key_here"
```

### 3. Basic Usage

```python
from airsenal.framework.xg_data_provider import create_xg_manager

# Initialize xG data manager
manager = create_xg_manager()

# Sync xG data for a specific gameweek
updated_count = manager.sync_xg_data_for_gameweek("2023-24", 15)
print(f"Updated {updated_count} player scores with xG data")

# Check provider health
dashboard = manager.get_health_dashboard()
print(dashboard["health_status"])
```

### 4. Command Line Usage

```bash
# Sync specific gameweek
python -m airsenal.scripts.sync_xg_data --season 2023-24 --gameweek 15

# Check provider health
python -m airsenal.scripts.sync_xg_data --health-check

# View cost summary
python -m airsenal.scripts.sync_xg_data --cost-summary

# Dry run (show what would be synced)
python -m airsenal.scripts.sync_xg_data --gameweek 15 --dry-run
```

## Provider Configuration

### Sportmonks Provider

The Sportmonks provider supports different subscription tiers:

```python
from airsenal.framework.xg_data_provider import ProviderConfig, ProviderTier, SportmonksProvider

# Configure for different tiers
config = ProviderConfig(
    api_key="your_key",
    tier=ProviderTier.STANDARD,  # BASIC, STANDARD, or ADVANCED
    rate_limit_per_minute=100,   # Adjust based on your plan
    cost_per_request=0.01,       # For cost tracking
    timeout_seconds=30,
    max_retries=3
)

provider = SportmonksProvider(config)
```

#### Tier Differences

| Tier | Data Delay | Live Updates | Rate Limit | Cost |
|------|------------|-------------|------------|------|
| Basic | 12 hours | No | Lower | Lower |
| Standard | Post-match | No | Medium | Medium |
| Advanced | Real-time | Yes | Higher | Higher |

### Custom Provider Implementation

To add a new provider (e.g., FBRef, Opta):

```python
from airsenal.framework.xg_data_provider import XGDataProvider, XGDataPoint

class CustomProvider(XGDataProvider):
    def get_player_xg_data(self, player_id, fixture_id=None, date_range=None):
        # Implement API call logic
        # Transform response to XGDataPoint objects
        return [XGDataPoint(...)]
    
    def get_match_xg_data(self, fixture_id):
        # Implement match data fetching
        return [XGDataPoint(...)]
    
    def get_bulk_xg_data(self, date_range, gameweeks=None):
        # Implement bulk data fetching
        return [XGDataPoint(...)]
    
    def _get_auth_headers(self):
        return {"Authorization": f"Bearer {self.config.api_key}"}
    
    def _is_credentials_optional(self):
        return False  # True if provider works without credentials
```

## Monitoring and Alerting

### Web Dashboard

Launch the monitoring dashboard:

```bash
# Console dashboard
python -m airsenal.scripts.xg_monitoring_dashboard --mode console

# Web dashboard (requires Flask)
python -m airsenal.scripts.xg_monitoring_dashboard --mode web --port 5000

# Save metrics snapshot
python -m airsenal.scripts.xg_monitoring_dashboard --mode snapshot

# Generate alert configuration
python -m airsenal.scripts.xg_monitoring_dashboard --mode alerts
```

### Health Metrics

The monitoring system tracks:

- **Provider Health**: Overall status, circuit breaker state
- **Success Rate**: Percentage of successful API calls
- **Response Times**: Average API response times
- **Rate Limiting**: Frequency of rate limit hits
- **Costs**: Total spend and cost per request
- **Circuit Breaker**: State transitions and failure counts

### Alerting

Example alert conditions:

```json
{
  "xg_provider_unhealthy": {
    "condition": "health_status.providers.*.is_healthy == false",
    "severity": "high"
  },
  "xg_circuit_breaker_open": {
    "condition": "health_status.providers.*.circuit_breaker_state == 'open'",
    "severity": "critical"
  },
  "xg_high_api_costs": {
    "condition": "cost_summary.total_cost > 10.0",
    "severity": "medium"
  }
}
```

## Error Handling

### Circuit Breaker Pattern

The circuit breaker automatically handles API failures:

1. **Closed**: Normal operation, requests pass through
2. **Open**: After 5 failures, all requests rejected for 60 seconds
3. **Half-Open**: After timeout, allows one test request

```python
# Circuit breaker configuration
breaker = CircuitBreaker(
    failure_threshold=5,    # Open after 5 failures
    recovery_timeout=60     # Wait 60s before testing recovery
)
```

### Retry Logic

Automatic retries with exponential backoff:

```python
retry_strategy = Retry(
    total=3,                          # Maximum retries
    backoff_factor=1,                 # Exponential backoff
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
```

### Graceful Degradation

The system handles missing credentials gracefully:

- Continues operation without xG data if no API key provided
- Logs warnings but doesn't break the main AIrsenal pipeline
- Returns empty results rather than raising exceptions

## Cost Management

### Tracking Costs

```python
# Get cost summary
manager = create_xg_manager()
dashboard = manager.get_health_dashboard()
costs = dashboard["cost_summary"]

print(f"Total cost: ${costs['total_cost']:.4f}")
print(f"Average per request: ${costs['average_cost_per_request']:.4f}")
```

### Cost Optimization

1. **Use Appropriate Tier**: Don't pay for real-time data if post-match is sufficient
2. **Batch Requests**: Use bulk endpoints when available
3. **Cache Results**: Store xG data locally to avoid repeated requests
4. **Rate Limiting**: Respect provider limits to avoid overage charges
5. **Monitor Usage**: Set up alerts for high costs

### Rate Limiting

```python
config = ProviderConfig(
    rate_limit_per_minute=100,  # Adjust based on your plan
    # ... other config
)

# Rate limiting is automatically enforced
# Requests are rejected if limit exceeded
```

## Integration with AIrsenal Pipeline

### Automatic Integration

Add xG data sync to your existing pipeline:

```python
# In airsenal_run_pipeline.py or custom script
from airsenal.framework.xg_data_provider import create_xg_manager

def update_xg_data(season, gameweek):
    manager = create_xg_manager()
    if manager.providers:
        updated = manager.sync_xg_data_for_gameweek(season, gameweek)
        print(f"Updated {updated} player scores with xG data")
    else:
        print("No xG providers configured - skipping xG data sync")

# Call after updating player scores but before running predictions
update_xg_data(CURRENT_SEASON, current_gameweek)
```

### Player Model Integration

The xG data automatically becomes available to prediction models:

```python
# In player models, xG data is available as:
player_score.expected_goals
player_score.expected_assists  
player_score.expected_goal_involvements
player_score.expected_goals_conceded
```

## Troubleshooting

### Common Issues

#### 1. "No API key configured"
```bash
# Solution: Set your API key
airsenal_env set -k SPORTMONKS_API_KEY -v "your_key"
```

#### 2. "Rate limit exceeded"
```bash
# Check your usage
python -m airsenal.scripts.sync_xg_data --health-check

# Solution: Upgrade your API plan or reduce request frequency
```

#### 3. "Circuit breaker OPEN"
```bash
# Check provider health
python -m airsenal.scripts.xg_monitoring_dashboard --mode console

# Solution: Wait for automatic recovery or investigate API issues
```

#### 4. "No PlayerScore found"
```bash
# Ensure player scores exist before syncing xG data
# Run airsenal_update_db first to populate PlayerScore table
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('airsenal.framework.xg_data_provider').setLevel(logging.DEBUG)
```

### Testing Provider Connection

```python
from airsenal.framework.xg_data_provider import SportmonksProvider

provider = SportmonksProvider()
try:
    # Test with a known player/fixture
    xg_data = provider.get_player_xg_data(player_id=123, fixture_id=456)
    print(f"Successfully retrieved {len(xg_data)} xG data points")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Optimization Tips

1. **Batch Operations**: Use bulk endpoints for multiple gameweeks
2. **Async Operations**: Consider async implementation for large datasets
3. **Caching**: Implement local caching to reduce API calls
4. **Database Indexing**: Ensure proper indexes on player_id and fixture_id
5. **Connection Pooling**: Reuse HTTP connections for better performance

### Benchmarking

```python
from airsenal.framework.logging_utils import timed

@timed
def sync_gameweek_xg_data(season, gameweek):
    manager = create_xg_manager()
    return manager.sync_xg_data_for_gameweek(season, gameweek)

# Function execution time will be logged
updated = sync_gameweek_xg_data("2023-24", 15)
```

## Security Considerations

### API Key Security

1. **Environment Variables**: Store API keys in environment, not code
2. **File Permissions**: Secure API key files with proper permissions
3. **Rotation**: Regularly rotate API keys
4. **Logging**: Don't log API keys in debug output

### Network Security

1. **HTTPS**: All API calls use HTTPS
2. **Timeout**: Configure appropriate timeouts
3. **Retry Limits**: Limit retry attempts to prevent abuse

## Future Enhancements

### Planned Features

1. **Historical Data Sync**: Backfill historical xG data
2. **Real-time Updates**: Live xG data during matches
3. **Multiple Providers**: Support for FBRef, Opta, etc.
4. **Advanced Analytics**: xG trend analysis and predictions
5. **Export Features**: Export xG data for external analysis

### Contributing

To add new providers or enhance existing functionality:

1. Implement the `XGDataProvider` interface
2. Add comprehensive tests following existing patterns  
3. Update documentation with new provider details
4. Submit pull request with test coverage

## API Reference

### Classes

#### XGDataProvider
Abstract base class for all xG data providers.

**Methods:**
- `get_player_xg_data(player_id, fixture_id=None, date_range=None)`: Get xG data for specific player
- `get_match_xg_data(fixture_id)`: Get xG data for all players in a match
- `get_bulk_xg_data(date_range, gameweeks=None)`: Bulk xG data retrieval

#### SportmonksProvider
Sportmonks API implementation.

**Configuration:**
- Supports all Sportmonks subscription tiers
- Automatic rate limiting and cost tracking
- Circuit breaker protection

#### XGDataManager
High-level management interface.

**Methods:**
- `sync_xg_data_for_gameweek(season, gameweek)`: Sync data for specific gameweek
- `get_health_dashboard()`: Get comprehensive status information

#### APIHealthMonitor
Provider health and cost monitoring.

**Features:**
- Real-time health status
- Cost tracking and projections
- Automated recommendations

### Configuration Options

#### ProviderConfig
```python
ProviderConfig(
    api_key=None,                    # API key
    base_url="",                     # Provider base URL
    rate_limit_per_minute=100,       # Rate limit
    timeout_seconds=30,              # Request timeout
    max_retries=3,                   # Retry attempts
    circuit_breaker_failure_threshold=5,  # Failure threshold
    circuit_breaker_recovery_timeout=60,  # Recovery timeout
    tier=ProviderTier.BASIC,         # Subscription tier
    cost_per_request=0.0             # Cost tracking
)
```

## Support

For issues, questions, or contributions:

1. Check existing GitHub issues
2. Review troubleshooting section
3. Check provider documentation
4. Submit detailed bug reports with logs

## License

This xG data provider integration follows the same license as AIrsenal core.