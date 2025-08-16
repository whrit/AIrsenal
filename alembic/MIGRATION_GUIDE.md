# PlayerAttributes Migration Guide

This guide provides comprehensive instructions for safely migrating the AIrsenal database to include enhanced PlayerAttributes features for machine learning predictions.

## Overview

**Migration ID:** `a55a2bfb8b17_extend_player_attributes_with_ml_features`

This migration extends the `player_attributes` table with 18 new columns and 8 performance indexes to support advanced FPL predictions:

### New Features Added

#### Expected Goals (xG) Metrics
- `xg_per_90`: Expected goals per 90 minutes
- `xa_per_90`: Expected assists per 90 minutes  
- `xgi_per_90`: Expected goal involvements per 90 minutes

#### Form Metrics
- `form_3_games`: Average FPL points over last 3 games
- `form_5_games`: Average FPL points over last 5 games
- `form_10_games`: Average FPL points over last 10 games
- `momentum`: Trend indicator for recent performance (-1 to 1)

#### Fixture Difficulty
- `next_3_fixture_difficulty`: Average difficulty rating for next 3 fixtures
- `next_5_fixture_difficulty`: Average difficulty rating for next 5 fixtures

#### Player Roles
- `is_penalty_taker`: Boolean indicator for penalty takers
- `is_free_kick_taker`: Boolean indicator for free kick takers
- `is_corner_taker`: Boolean indicator for corner takers
- `role_confidence`: Confidence score for set piece roles (0-1)

#### Advanced Statistics (per 90 minutes)
- `shots_per_90`: Total shots per 90 minutes
- `key_passes_per_90`: Key passes leading to shots
- `tackles_per_90`: Successful tackles
- `interceptions_per_90`: Interceptions
- `clearances_per_90`: Clearances

### Performance Indexes
- `ix_form_3_games`, `ix_form_5_games`: For form-based queries
- `ix_xg_per_90`, `ix_xgi_per_90`: For xG-based analysis
- `ix_next_3_fixture_difficulty`: For fixture difficulty filtering
- `ix_position_penalty_taker`: For role-based position queries
- `ix_momentum`: For momentum-based player selection

## Pre-Migration Requirements

### 1. Environment Setup
Ensure Alembic is installed:
```bash
uv sync  # or pip install alembic>=1.13.0
```

### 2. Database Backup
**CRITICAL**: Always backup your database before migration:

#### For SQLite:
```bash
cp $AIRSENAL_HOME/data.db $AIRSENAL_HOME/data.db.backup.$(date +%Y%m%d_%H%M%S)
```

#### For PostgreSQL:
```bash
pg_dump airsenal > airsenal_backup_$(date +%Y%m%d_%H%M%S).sql
```

### 3. Pre-Migration Validation
Run validation checks:
```bash
cd /path/to/AIrsenal
python alembic/migration_validation.py pre
```

Expected output:
```
🔍 Running pre-migration validation checks...
✅ Database connectivity: OK
✅ player_attributes table: EXISTS
✅ Schema consistency: OK (X columns)
📊 Existing records: XXX,XXX
📊 Unique seasons: X
📊 Unique players: X,XXX
✅ No null constraint violations detected
✅ All pre-migration checks passed!
```

## Migration Execution

### 1. Check Current Migration Status
```bash
alembic current
```

### 2. Run the Migration
```bash
alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> a55a2bfb8b17, extend_player_attributes_with_ml_features
```

### 3. Post-Migration Validation
```bash
python alembic/migration_validation.py post
```

Expected output:
```
🔍 Running post-migration validation checks...
✅ Database connectivity: OK
✅ New columns created: ALL 18 columns
✅ New indexes created: ALL 7 indexes
✅ xg_per_90: FLOAT
✅ is_penalty_taker: BOOLEAN
✅ form_3_games: FLOAT
✅ is_penalty_taker: No null values (defaults working)
✅ is_free_kick_taker: No null values (defaults working)
✅ is_corner_taker: No null values (defaults working)
✅ Data integrity: XXX,XXX records intact
✅ Index performance: Query optimizer using new indexes
✅ All post-migration checks passed!
```

## Rollback Procedures

### 1. Check Rollback Safety
```bash
python alembic/migration_validation.py rollback-check
```

### 2. Execute Rollback (if needed)
```bash
alembic downgrade -1
```

**Warning**: Rollback will permanently delete all data in the new columns!

## Database Compatibility

### SQLite
- ✅ Supported
- ✅ Column additions work seamlessly  
- ✅ Index creation supported
- ⚠️  Boolean columns stored as INTEGER (0/1)

### PostgreSQL
- ✅ Supported
- ✅ Full boolean type support
- ✅ Advanced index types available
- ✅ Column comments preserved

## Zero-Downtime Migration Strategy

For production systems requiring minimal downtime:

### 1. Staging Environment Testing
```bash
# Test on staging first
export AIRSENAL_DB_FILE=/path/to/staging.db
python alembic/migration_validation.py pre
alembic upgrade head
python alembic/migration_validation.py post
```

### 2. Production Migration
```bash
# During low-traffic period
# 1. Stop AIrsenal services
# 2. Create backup
# 3. Run migration
# 4. Validate
# 5. Restart services
```

### 3. Blue-Green Deployment (Advanced)
For PostgreSQL environments:
1. Create replica database
2. Apply migration to replica
3. Switch application to replica
4. Apply migration to primary
5. Switch back (if needed)

## Data Population

After migration, new columns will be `NULL` for existing records. To populate them:

### 1. Run Feature Computation
```bash
# Use AIrsenal's feature computation pipelines
airsenal_update_db  # Will populate some basic metrics
```

### 2. Manual Population (if needed)
```sql
-- Example: Set default values for role indicators
UPDATE player_attributes 
SET is_penalty_taker = false, 
    is_free_kick_taker = false, 
    is_corner_taker = false
WHERE is_penalty_taker IS NULL;

-- Example: Compute basic form metrics from existing data
-- (Complex queries would go here based on historical PlayerScore data)
```

## Troubleshooting

### Common Issues

#### Migration Fails: "Column already exists"
```bash
# Check current state
alembic current
alembic history

# If partially applied, fix manually:
# 1. Check which columns exist
# 2. Modify migration script to skip existing columns
# 3. Or rollback and re-run
```

#### Index Creation Fails
```bash
# Check existing indexes
sqlite3 $AIRSENAL_HOME/data.db ".indexes player_attributes"

# Drop conflicting indexes manually if needed
```

#### Performance Issues After Migration
```bash
# Check query plans
EXPLAIN QUERY PLAN SELECT * FROM player_attributes WHERE form_3_games > 5;

# Re-analyze statistics (PostgreSQL)
ANALYZE player_attributes;

# Vacuum and reindex (SQLite)
sqlite3 $AIRSENAL_HOME/data.db "VACUUM; REINDEX;"
```

### Recovery Procedures

#### Partial Migration Failure
1. Check migration state: `alembic current`
2. Check what was applied: `python alembic/migration_validation.py post`
3. Either complete manually or rollback
4. Fix issues and re-run

#### Data Corruption
1. Stop all AIrsenal processes
2. Restore from backup
3. Investigate root cause
4. Test migration on copy before re-applying

## Performance Impact

### Disk Space
- **Estimated increase**: ~20-30% for new columns
- **Index overhead**: ~10-15% additional space
- **Total impact**: ~30-45% database size increase

### Query Performance
- **Existing queries**: Minimal impact
- **New feature queries**: Significant improvement with indexes
- **Bulk inserts**: Slight slowdown due to index maintenance

### Memory Usage
- **SQLite**: Minimal increase
- **PostgreSQL**: Plan cache may need refresh

## Monitoring Post-Migration

### Key Metrics to Monitor
1. **Query performance** on player_attributes table
2. **Database size growth** over time
3. **Index usage statistics**
4. **Application performance** for player selection queries

### Recommended Monitoring Queries

```sql
-- Check new column population rates
SELECT 
    COUNT(*) as total_records,
    COUNT(xg_per_90) as xg_populated,
    COUNT(form_3_games) as form_populated,
    COUNT(*) - COUNT(xg_per_90) as xg_missing
FROM player_attributes;

-- Monitor index usage (PostgreSQL)
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename = 'player_attributes';
```

## Integration with AIrsenal

### Code Changes Required
After migration, update code to utilize new features:

1. **Feature computation pipelines**: Populate new columns
2. **Player selection logic**: Use form metrics and xG data
3. **Transfer optimization**: Leverage fixture difficulty ratings
4. **Model training**: Include new features in ML models

### API Changes
Update any API endpoints that return player attributes to include new fields.

## Support and Maintenance

### Regular Maintenance
- **Weekly**: Monitor index usage and query performance
- **Monthly**: Check data population rates and quality
- **Quarterly**: Analyze feature importance and usage patterns

### Support Resources
- **Schema documentation**: `airsenal/framework/schema.py`
- **Migration validation**: `alembic/migration_validation.py`
- **Community support**: AIrsenal GitHub issues

---

**Last Updated**: 2025-08-16  
**Migration Version**: a55a2bfb8b17  
**Compatibility**: AIrsenal v1.11.0+