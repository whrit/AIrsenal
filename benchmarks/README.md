# AIrsenal Performance Benchmarking Framework

A comprehensive performance testing and monitoring system for AIrsenal's machine learning pipeline and Sprint 00 framework components.

## Overview

This benchmarking framework provides:

- **Framework Component Benchmarks**: Performance tests for individual components (Feature Store, Redis Cache, Model Versioning, Base Models, Database operations)
- **Integration Benchmarks**: End-to-end performance tests for prediction pipeline and optimization
- **Load Testing**: Realistic user load simulation using Locust
- **Performance Dashboard**: Real-time monitoring and historical analysis
- **CI/CD Integration**: Automated performance regression detection

## Quick Start

### Prerequisites

```bash
# Install dependencies
uv sync --extra dev

# Set up environment
export FPL_TEAM_ID=742663
export BENCHMARK_ENV=local
```

### Run Benchmarks

```bash
# Run all framework component benchmarks
airsenal_run_benchmarks --suite framework

# Run integration benchmarks
airsenal_run_benchmarks --suite integration

# Run with baseline comparison
airsenal_run_benchmarks --suite framework --compare-baseline

# Run with threshold checking
airsenal_run_benchmarks --suite framework --threshold-check
```

### View Performance Dashboard

```bash
# Generate dashboard
airsenal_performance_dashboard --generate-only

# Run continuous monitoring
airsenal_performance_dashboard --monitor
```

### Load Testing

```bash
# Start AIrsenal API (if available)
# Then run load tests
locust -f benchmarks/load_testing/prediction_load_test.py --host=http://localhost:5000
```

## Framework Components

### 1. Feature Store Benchmarks

Tests the performance of the FeatureStore component:

- **Single feature retrieval**: `< 1ms` per feature
- **Batch feature retrieval**: `< 10ms` for 100 players
- **Rolling window calculations**: `< 100ms` for 5-game windows
- **Cache hit rates**: Target `> 85%` hit rate
- **Memory usage**: `< 50MB` for typical workloads

```python
# Example usage
from benchmarks.framework.test_feature_store_benchmarks import FeatureStoreBenchmarks
from benchmarks.utils import BenchmarkRunner

runner = BenchmarkRunner()
fs_bench = FeatureStoreBenchmarks(runner)
fs_bench.run_all()
```

### 2. Redis Cache Benchmarks

Tests Redis caching layer performance:

- **Single operations**: `< 0.5ms` GET, `< 1ms` SET
- **Batch operations**: `< 5ms` for 100 keys
- **Serialization**: Efficient for NumPy arrays and DataFrames
- **Connection pooling**: Minimal overhead
- **TTL performance**: Fast expiration handling

### 3. Model Versioning Benchmarks

Tests model management system:

- **Model registration**: `< 100ms` per model
- **Serialization**: `< 1s` for medium models
- **Loading**: `< 2s` for large models
- **Artifact storage**: Compressed storage efficiency
- **Version operations**: Fast metadata queries

### 4. Base Models Benchmarks

Tests abstract model interfaces:

- **Adaptive model updates**: `< 100ms` per update
- **Form calculations**: `< 50ms` per player
- **Availability predictions**: `< 20ms` per prediction
- **Feature engineering**: Efficient pipeline execution

### 5. Database Benchmarks

Tests database query performance:

- **Simple queries**: `< 10ms`
- **Complex joins**: `< 100ms`
- **Bulk operations**: `< 1s` for 1000 records
- **Aggregations**: Optimized for FPL data patterns

## Integration Testing

### Prediction Pipeline

End-to-end tests covering:

```python
# Typical prediction workflow
features = feature_store.get_features(player_id, gameweek)
prediction = model.predict(features)
cache.set(cache_key, prediction)
```

**Performance targets:**
- **Single prediction**: `< 50ms` including feature retrieval
- **Batch predictions**: `< 2s` for 100 players
- **Memory usage**: `< 200MB` for typical batch sizes

### Optimization Benchmarks

Tests optimization algorithms:

- **Squad optimization**: Genetic algorithm performance
- **Transfer optimization**: Multi-transfer scenarios
- **Memory usage**: Large population handling

## Load Testing

### User Simulation

Realistic user patterns using Locust:

```python
class PredictionServiceUser(HttpUser):
    @task(3)
    def get_single_player_prediction(self):
        # Single prediction request
        
    @task(2)  
    def get_batch_player_predictions(self):
        # Batch prediction request
        
    @task(1)
    def get_optimization_suggestions(self):
        # Heavy optimization request
```

**Load scenarios:**
- **Normal load**: 10 users, 1 req/sec spawn rate
- **Peak load**: 100 users, 10 req/sec spawn rate
- **Stress test**: Breaking point identification

### Performance Targets

| Component | Latency (p95) | Throughput | Memory |
|-----------|---------------|------------|---------|
| Feature Store | 10ms | 1000 req/s | 50MB |
| Prediction | 100ms | 500 req/s | 200MB |
| Cache | 1ms | 5000 req/s | 100MB |
| Database | 20ms | 1000 req/s | 150MB |

## Performance Dashboard

### Real-time Monitoring

The dashboard provides:

- **System metrics**: CPU, memory, disk usage
- **Application metrics**: Request latency, throughput
- **Performance trends**: Historical analysis
- **Alert system**: Threshold violations
- **Regression detection**: Baseline comparison

### Metrics Collection

```python
# Prometheus metrics (if available)
prediction_latency = Histogram('airsenal_prediction_latency_seconds')
cache_hit_rate = Gauge('airsenal_cache_hit_rate')
memory_usage = Gauge('airsenal_memory_usage_mb')
```

### Visualization

Automated generation of:
- Performance trend plots
- Memory usage charts  
- Test execution summaries
- Regression analysis

## Configuration

### Environment Settings

```python
# benchmarks/configs/benchmark_config.py
@dataclass
class PerformanceThresholds:
    feature_get_single: float = 0.001  # 1ms
    redis_get_single: float = 0.0005   # 0.5ms
    prediction_single: float = 0.050   # 50ms
    # ... more thresholds
```

### Environment-specific Configs

- **CI**: Faster tests, fewer iterations
- **Local**: Full test suite, profiling enabled
- **Stress**: Large datasets, breaking point tests

## CI/CD Integration

### GitHub Actions Workflow

Automated performance testing on:
- **Push to main/develop**: Framework benchmarks
- **Pull requests**: Regression detection  
- **Daily schedule**: Full test suite + load testing
- **Manual trigger**: Custom benchmark suites

### Performance Gates

```yaml
- name: Run framework benchmarks
  run: |
    uv run airsenal_run_benchmarks \
      --suite framework \
      --env ci \
      --compare-baseline \
      --threshold-check
```

**Failure conditions:**
- Performance threshold violations
- > 10% regression vs baseline
- Failed benchmark tests

### Baseline Management

- **Automatic updates**: On main branch merges
- **Baseline comparison**: Against previous releases
- **Regression alerts**: Automatic PR comments

## Development Guidelines

### Adding New Benchmarks

1. **Create benchmark class**:
```python
class NewComponentBenchmarks:
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def run_all(self):
        self.benchmark_operation()
    
    @pytest.mark.benchmark(group="new_component")
    def benchmark_operation(self):
        with self.runner.benchmark_context("operation_name"):
            # Benchmark code here
```

2. **Add to main runner**:
```python
# In run_framework_benchmarks()
new_bench = NewComponentBenchmarks(runner)
new_bench.run_all()
```

3. **Set performance thresholds**:
```python
# In benchmark_config.py
class PerformanceThresholds:
    new_operation: float = 0.010  # 10ms threshold
```

### Best Practices

- **Realistic data**: Use representative test datasets
- **Warm-up**: Cache warming before measurements
- **Isolation**: Clear state between tests
- **Stability**: Multiple iterations for reliable results
- **Documentation**: Clear performance expectations

### Debugging Performance Issues

1. **Enable profiling**:
```bash
airsenal_run_benchmarks --suite framework --profile
```

2. **Check memory usage**:
```python
with runner.benchmark_context("memory_test"):
    # Memory-intensive operation
    # Results include memory_used_mb
```

3. **Analyze trends**:
```bash
airsenal_performance_dashboard --data-dir benchmarks/reports
```

## Troubleshooting

### Common Issues

**Redis connection failed**:
```bash
# Start Redis
redis-server
# Or use Docker
docker run -d -p 6379:6379 redis:alpine
```

**Database initialization failed**:
```bash
# Set up test database
export FPL_TEAM_ID=742663
airsenal_setup_initial_db --test-data-only
```

**Import errors**:
```bash
# Ensure project root in path
export PYTHONPATH=/path/to/AIrsenal:$PYTHONPATH
```

### Performance Analysis

**Slow benchmarks**:
- Check system load during tests
- Verify cache warming
- Review test data size
- Monitor memory usage

**Inconsistent results**:
- Increase benchmark iterations
- Check for background processes
- Verify test isolation
- Review environment configuration

## Future Enhancements

### Planned Features

- **Distributed testing**: Multi-node load testing
- **GPU benchmarks**: JAX/CUDA performance tests
- **Cost analysis**: Resource usage optimization
- **A/B testing**: Model performance comparison
- **Real-time alerts**: Slack/email notifications

### Performance Optimization Opportunities

- **Vectorization**: NumPy/JAX optimizations
- **Caching strategies**: Multi-level caching
- **Database indexing**: Query optimization
- **Parallel processing**: Multi-threading/async
- **Memory management**: Garbage collection tuning

## Contributing

When contributing performance improvements:

1. **Run benchmarks** before and after changes
2. **Update baselines** if performance improves
3. **Document changes** in threshold expectations
4. **Add new benchmarks** for new features
5. **Monitor regressions** in CI/CD pipeline

---

For questions or issues with the benchmarking framework, please create an issue or contact the development team.