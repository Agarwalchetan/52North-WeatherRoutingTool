# 05 — Performance Optimization

## 1. Identified Bottlenecks

> **All performance numbers in this document are estimates until profiled during Community Bonding Week 1.** Estimated values are labeled with "~est".

### 1.1 Eager Dataset Loading — `weather.py:619`

```python
self.ds = xr.open_dataset(filepath)  # Loads entire file into RAM
```

**Impact:** For a 5-day, 0.083° North Atlantic dataset:
- Dimensions: 40 time × 420 lat × 480 lon
- Variables: 11 (float32)
- Estimated size: ~350–500 MB
- Load time: To be profiled (Week 1)

**Fix:** Dask-chunked lazy loading (opt-in):
```python
self.ds = xr.open_dataset(filepath, chunks={'time': 10, 'latitude': 100, 'longitude': 100})
```
Expected: metadata-only read (~100ms est.), peak RSS = configured chunk size.

### 1.2 Sequential Weather Lookups — `ship.py:52-75`

```python
for i_coord in range(0, n_coords):
    wave_direction.append(self.approx_weather(weather_data['VMDR'], lats[i_coord], ...))
    # ... 10 more variables
```

**Impact:** Per routing step with 600 candidates: 6,600 sequential `.sel()` calls.

**Fix:** Vectorized batch lookup:
```python
lat_da = xr.DataArray(lats, dims='points')
lon_da = xr.DataArray(lons, dims='points')
time_da = xr.DataArray(time, dims='points')

ship_params.wave_height = weather_data['VHM0'].sel(
    latitude=lat_da, longitude=lon_da, time=time_da,
    method='nearest').values * u.meter
```

**Expected:** Python loop overhead eliminated (6,600 calls → 11 calls). Per-element xarray indexing cost remains — the true speedup depends on chunk alignment and will be measured.

### 1.3 Redundant Dataset Opens — `ship.py:37`

```python
def evaluate_weather(self, ship_params, lats, lons, time):
    weather_data = xr.open_dataset(self.weather_path)
```

**Fix:** Pass `WeatherCond.ds` as a reference:
```python
def evaluate_weather(self, ship_params, lats, lons, time, weather_data=None):
    if weather_data is None:
        weather_data = xr.open_dataset(self.weather_path)
```

## 2. Dask Scheduler Selection

| Dataset Size | Scheduler | Rationale |
|-------------|-----------|-----------|
| < threshold | `synchronous` | Avoid Dask overhead entirely; operate on eager numpy arrays |
| threshold – 10× | `threads` | NumPy releases GIL for most array ops; threaded scheduler works well |
| > 10× | `distributed` | Full parallelism with memory spilling to disk |

> **Threshold will be determined empirically during Week 1 profiling.** Initial estimate: ~500 MB based on typical Dask scheduler overhead of 50-100ms.

```python
def _configure_scheduler(self):
    size_mb = self._get_dataset_size_mb()
    if size_mb < self.config.DASK_THRESHOLD_MB:
        if self._is_dask_backed():
            self.ds = self.ds.compute()  # Materialize to eager numpy
    elif size_mb < self.config.DASK_THRESHOLD_MB * 10:
        dask.config.set(scheduler='threads')
    else:
        from dask.distributed import Client
        self._client = Client(n_workers=self.config.N_WORKERS)
```

## 3. Memory Optimization

### 3.1 Float64 Upcasting for Aggregation Only

```python
def compute_mean(self, variables, dim='time', skipna=True, min_count=1):
    return self.ds[variables].astype(np.float64).mean(
        dim=dim, skipna=skipna, min_count=min_count)
```

Upcasting is applied **only during reduction**, not during storage. Results can be optionally downcast back to float32 if memory is constrained.

### 3.2 NaN-Aware Computation

All aggregations use `skipna=True` (explicit) and `min_count=1` to prevent computing statistics from cells with insufficient valid data. This is critical for ocean datasets where 30-40% of grid cells may be NaN (land pixels).

### 3.3 Approximate Percentiles for Very Large Datasets

For datasets where even a single variable exceeds available RAM, use reservoir sampling:
```python
def compute_approx_percentile(self, variable, q=0.95, n_samples=100000):
    """Approximate percentile via reservoir sampling — avoids full materialization."""
    var = self.ds[variable]
    if self._is_dask_backed():
        # Sample from each chunk, then combine
        samples = []
        for block in var.data.blocks:
            block_data = block.compute().ravel()
            valid = block_data[~np.isnan(block_data)]
            if len(valid) > 0:
                n = min(n_samples // var.data.numblocks[0], len(valid))
                samples.append(np.random.choice(valid, size=n, replace=False))
        all_samples = np.concatenate(samples)
    else:
        data = var.values.ravel()
        all_samples = data[~np.isnan(data)]
    return float(np.percentile(all_samples, q * 100))
```

## 4. Expected Performance Gains

> **All "Expected" values are estimates. Actual values will be measured during Phase 4 benchmarking (Weeks 17-18) and compared against Week 1 baselines.**

| Operation | Current (est.) | Proposed (est.) | Expected Improvement |
|-----------|---------------|-----------------|---------------------|
| Weather data load (500 MB) | Full file read into RAM | Metadata-only lazy open | Significant (to be measured) |
| Per-step weather lookup (600 pts) | 6,600 sequential `.sel()` | 11 vectorized `.sel()` | Python loop overhead eliminated |
| Redundant file opens (60 steps) | 60 × metadata parse | 0 (shared reference) | Eliminated |
| `compute_mean(dim='time')` | N/A | ~sub-second est. (4-core, 500 MB) | New capability |
| `compute_correlation()` (11 vars) | N/A | ~seconds est. (Dask parallel via `xr.corr()`) | New capability |
