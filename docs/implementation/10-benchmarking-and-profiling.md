# 10 — Benchmarking and Profiling

## 1. Benchmark Strategy

### 1.1 Two-Phase Approach

1. **Phase A (Community Bonding Week 1):** Establish baselines by profiling the existing WRT system. No estimates — actual measurements.
2. **Phase B (Phase 4, Weeks 17-18):** Full benchmark suite comparing before/after with the statistics module integrated.

### 1.2 Baseline Profiling Script (Week 1)

```python
"""benchmarks/baseline_profiling.py — Run BEFORE any code changes."""

import cProfile
import pstats
import tracemalloc
import time
import xarray as xr

def profile_read_dataset(filepath):
    """Profile existing read_dataset() performance."""
    tracemalloc.start()
    t0 = time.perf_counter()

    ds = xr.open_dataset(filepath)
    _ = ds['VHM0'].values  # Force full materialization

    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"read_dataset: {dt:.3f}s, peak RSS: {peak / 1024**2:.1f} MB")
    print(f"Dataset size: {sum(v.nbytes for v in ds.data_vars.values()) / 1024**2:.1f} MB")
    return {'time_s': dt, 'peak_mb': peak / 1024**2}

def profile_evaluate_weather(filepath, n_points=600):
    """Profile sequential .sel() lookup cost."""
    import numpy as np
    ds = xr.open_dataset(filepath)
    lats = np.random.uniform(float(ds.latitude.min()), float(ds.latitude.max()), n_points)
    lons = np.random.uniform(float(ds.longitude.min()), float(ds.longitude.max()), n_points)
    times = np.random.choice(ds.time.values, n_points)

    # Sequential (current)
    t0 = time.perf_counter()
    for i in range(n_points):
        _ = ds['VHM0'].sel(latitude=lats[i], longitude=lons[i],
                            time=times[i], method='nearest').values
    dt_seq = time.perf_counter() - t0

    # Vectorized (proposed)
    t0 = time.perf_counter()
    lat_da = xr.DataArray(lats, dims='points')
    lon_da = xr.DataArray(lons, dims='points')
    time_da = xr.DataArray(times, dims='points')
    _ = ds['VHM0'].sel(latitude=lat_da, longitude=lon_da,
                        time=time_da, method='nearest').values
    dt_vec = time.perf_counter() - t0

    print(f"Sequential ({n_points} pts): {dt_seq:.3f}s ({dt_seq/n_points*1000:.2f} ms/pt)")
    print(f"Vectorized ({n_points} pts): {dt_vec:.3f}s ({dt_vec/n_points*1000:.2f} ms/pt)")
    print(f"Speedup: {dt_seq/dt_vec:.1f}×")

if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'path/to/weather.nc'
    profile_read_dataset(filepath)
    profile_evaluate_weather(filepath)
```

## 2. Benchmark Suite (Phase B)

### 2.1 Dimensions

| Dimension | Values |
|-----------|--------|
| Dataset size | Synthetic: 100 MB, 500 MB, 1 GB |
| Chunk strategy | No chunks (eager), balanced, timeseries-optimized |
| Dask scheduler | synchronous, threads |
| Operation | mean, std, percentile, pairwise correlation, distribution fit |
| NaN percentage | 0%, 30%, 50% |

### 2.2 Benchmark Script

```python
"""benchmarks/run_benchmarks.py"""

import json, time, tracemalloc
import numpy as np, xarray as xr
from WeatherRoutingTool.statistics import WeatherStatisticsEngine

def generate_dataset(n_time, n_lat, n_lon, nan_pct=0.0):
    np.random.seed(42)
    data = np.random.weibull(2, (n_time, n_lat, n_lon)).astype(np.float32) * 2
    if nan_pct > 0:
        mask = np.random.random(data.shape) < nan_pct
        data[mask] = np.nan
    return xr.Dataset({
        'VHM0': (['time', 'latitude', 'longitude'], data),
    }, coords={
        'time': np.arange(n_time).astype('datetime64[h]') + np.datetime64('2024-01-01'),
        'latitude': np.linspace(35, 65, n_lat),
        'longitude': np.linspace(-60, -20, n_lon),
    })

def benchmark(engine, method_path, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()

    # Navigate composition: 'aggregation.compute_mean' → engine.aggregation.compute_mean
    parts = method_path.split('.')
    obj = engine
    for p in parts[:-1]:
        obj = getattr(obj, p)
    result = getattr(obj, parts[-1])(**kwargs)
    if hasattr(result, 'compute'):
        result = result.compute()

    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'time_s': round(dt, 3), 'peak_mb': round(peak / 1024**2, 1)}

CONFIGS = {
    '100MB': (40, 100, 120),
    '500MB': (40, 250, 250),
    '1GB':   (80, 300, 400),
}

if __name__ == '__main__':
    results = {}
    for label, (nt, nlat, nlon) in CONFIGS.items():
        for nan_pct in [0.0, 0.3]:
            ds = generate_dataset(nt, nlat, nlon, nan_pct)
            tag = f"{label}_nan{int(nan_pct*100)}"

            # Eager
            engine = WeatherStatisticsEngine(ds)
            results[f"{tag}_eager_mean"] = benchmark(
                engine, 'aggregation.compute_mean', variables=['VHM0'])

            # Dask
            ds_dask = ds.chunk({'time': 10, 'latitude': 100, 'longitude': 100})
            engine_dask = WeatherStatisticsEngine(ds_dask)
            results[f"{tag}_dask_mean"] = benchmark(
                engine_dask, 'aggregation.compute_mean', variables=['VHM0'])

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
```

## 3. Profiling Methods

| Method | What It Measures | When To Use |
|--------|-----------------|-------------|
| `cProfile` | CPU time per function call | Finding hotspot functions |
| `tracemalloc` | Peak memory allocation | Verifying chunk-based memory control |
| `memory_profiler` (`@profile`) | Line-by-line memory | Debugging memory leaks in specific functions |
| Dask dashboard (`:8787`) | Task execution, worker utilization | Optimizing Dask scheduler and chunk sizes |
| `time.perf_counter()` | Wall-clock time | Before/after comparison |

## 4. Success Criteria

> All targets are validated against actual Week 1 baselines, not estimates.

| Metric | Baseline Source | Target |
|--------|----------------|--------|
| Weather data load time (chunked) | Week 1 profiling | < baseline × 0.2 (metadata-only lazy open) |
| Weather data peak RSS (chunked) | Week 1 profiling | ≤ configured chunk size |
| `compute_mean(dim='time')` 500 MB | N/A (new capability) | < 2s with 4-core Dask |
| `compute_pairwise` correlation | N/A (new capability) | < 5s for 11 vars, 500 MB |
| No memory regression for routing | Week 1 baseline | ≤ baseline (± 5%) |
| Results match eager vs Dask | N/A | `xr.testing.assert_allclose()` passes |

## 5. CI Integration

> **Note:** WRT does not appear to have an existing CI pipeline. The following is proposed as part of this project's deliverables.

```yaml
# .github/workflows/test.yml (proposed)
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: pytest tests/ --cov=WeatherRoutingTool/statistics --cov-report=xml
      - uses: codecov/codecov-action@v4
```

Dedicated benchmark runs on PRs touching `statistics/`:
```yaml
# .github/workflows/benchmark.yml (proposed)
name: Benchmarks
on:
  pull_request:
    paths: ['WeatherRoutingTool/statistics/**']
jobs:
  bench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[test]"
      - run: python benchmarks/run_benchmarks.py
      - run: python benchmarks/baseline_profiling.py tests/testdata/weather_sample.nc
```
