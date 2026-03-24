# Risks and Trade-offs

## 1. NaN Handling (Critical Risk — Not Addressed in Original)

### Risk: Pervasive NaN Values in Ocean Datasets

Ocean weather data contains systematic NaN patterns:
- **Land pixels:** All ocean-only variables (VHM0, VMDR, VTPK, thetao, so, utotal, vtotal) are NaN over land.
- **Below-depth cells:** Physics variables (thetao, so) are NaN below the ocean floor.
- **Missing observations:** Temporal gaps in satellite-derived products.

For a typical North Atlantic domain (35°–70°N, -60°–20°W), approximately **30-40% of spatial grid cells are land** and therefore NaN for ocean variables.

**Impact on statistical operations:**

| Operation | NaN Behavior (default xarray) | Correct Behavior |
|-----------|------------------------------|-------------------|
| `.mean()` | `skipna=None` → True for float → ignores NaN | Correct default, but must be explicit |
| `.std()` | `skipna=None` → True for float → ignores NaN, but uses N-NaN as denominator | Must verify ddof correction with NaN |
| `.quantile()` | `skipna=None` → True for float | Correct default |
| `.rolling().mean()` | `min_periods=None` → returns NaN if any window element is NaN | Must set `min_periods` explicitly |
| `xr.corr()` | Drops NaN pairwise | Correct but reduces effective sample size — must report N |
| `scipy.stats.weibull_min.fit()` | **Crashes on NaN input** | Must filter NaN before fitting |

**Mitigation:**
```python
class AggregationMixin:
    def compute_mean(self, variables=None, dim='time', skipna=True, min_count=1):
        subset = self._select_variables(variables)
        return subset.astype(np.float64).mean(dim=dim, skipna=skipna, min_count=min_count)
```

- All aggregation functions accept explicit `skipna` and `min_count` parameters.
- `min_count` prevents artificially confident statistics from cells with very few valid observations.
- Distribution fitting pre-filters NaN with `data[~np.isnan(data)]` and checks `len(valid) >= min_samples`.
- A `NaN diagnostic report` method returns per-variable NaN percentages across the dataset.

---

## 2. Memory Overflow Risks

### Risk: Dask Chunk Graph Explosion

When performing operations that combine many chunks (e.g., correlation matrix across 11 variables), Dask constructs a task graph proportional to `n_chunks × n_variables × n_operations`.

**Mitigation:**
```python
# Anti-pattern: fine-grained chunks → huge task graph
ds = xr.open_dataset(path, chunks={'time': 1, 'latitude': 10, 'longitude': 10})
# → ~100,000 chunks → scheduler OOM

# Correct: coarser chunks aligned to operation
ds = xr.open_dataset(path, chunks={'time': -1, 'latitude': 200, 'longitude': 200})
# → ~25 chunks → manageable
```

`ChunkOptimizer` validates that task graph size doesn't exceed a configurable maximum (default: 100,000 tasks).

### Risk: Intermediate Materialization

`scipy.stats` functions and `RegularGridInterpolator` require materialized numpy arrays. `xr.apply_ufunc(dask='parallelized')` handles this per-chunk, but each chunk must fit in memory.

**Mitigation:**
- Default chunk sizes calibrated to produce ~100 MB chunks.
- `max_chunk_memory` config parameter (default: 256 MB) validates chunk sizes at load time.

---

## 3. I/O Bottlenecks

### Risk: NetCDF Chunk Alignment Mismatch

NetCDF4 (HDF5) has poor performance when read access patterns don't align with storage chunk layout. If the NetCDF was written with time-contiguous chunks but we read spatially, performance degrades 10-100×.

**Mitigation:**
- Provide `rechunk_for_analysis()` utility that converts to Zarr with optimal chunking.
- Document recommended chunk layouts for common analysis patterns.
- Consider `kerchunk` (creates a virtual Zarr view of NetCDF files without copying data) for large existing NetCDF archives.

### Risk: File System Metadata Overhead with Zarr

Zarr stores each chunk as a separate file. For 10,000+ chunks, file system metadata operations can become a bottleneck on NFS or network storage.

**Mitigation:**
- Document `zarr.convenience.consolidate_metadata()` for reducing metadata reads.
- For network storage, recommend `ZipStore` or `SQLiteStore`.

---

## 4. Dask Overhead vs. Benefit

### Risk: Small Dataset Penalty

For datasets that fit in memory (< 500 MB), Dask introduces scheduler overhead without benefit.

> **Note:** The numbers below are **estimates**. Actual values will be measured during Community Bonding Week 1 and reported in the Dask compatibility matrix.

| Operation | Expected NumPy (in-memory) | Expected Dask (4 workers) | Expected Overhead |
|-----------|---------------------------|---------------------------|-------------------|
| Mean over time (100 × 500 × 500 grid) | ~15 ms | ~100 ms | ~6× slower |
| Std over time | ~20 ms | ~120 ms | ~6× slower |

**Mitigation:**
```python
def _should_use_dask(self) -> bool:
    """Use Dask only when dataset exceeds in-memory threshold."""
    return self._get_dataset_size_mb() > self.config.CHUNK_THRESHOLD_MB
```

For small datasets, the engine calls `.compute()` upfront and operates on eager numpy arrays.

### Risk: Scheduler Thread Contention

NumPy releases the GIL for most array operations, so the threaded scheduler works well for xarray reductions. However, Python-level operations (e.g., `scipy.stats.fit()` inside `apply_ufunc`) hold the GIL and contend with threads.

**Mitigation:**
- Use `dask='parallelized'` with `apply_ufunc()` which uses the default scheduler (thread-safe for GIL-releasing numpy).
- For scipy-heavy workloads, switch to `processes` scheduler via config.

---

## 5. Coordinate System Risks

### Risk: Longitude Convention Inconsistency

GFS uses 0°–360° longitude convention. CMEMS uses -180°–180°. After `interp_like()` merge (weather.py:298-310), the resulting dataset uses the CMEMS convention. However:

- If a user provides pre-processed NetCDF files with 0-360° convention, spatial statistics at the dateline (180°) may be incorrect.
- Rolling spatial windows that cross the dateline will produce discontinuities.

**Mitigation:**
- Add longitude normalization in `WeatherStatisticsEngine.__init__()`:
```python
def _normalize_coordinates(self):
    if self.ds.longitude.max() > 180:
        self.ds = self.ds.assign_coords(
            longitude=((self.ds.longitude + 180) % 360) - 180
        ).sortby('longitude')
```
- Document the longitude convention used by the statistics module.

### Risk: Time Calendar Incompatibility

Different data sources may use different calendar systems (standard, proleptic_gregorian, 360_day). `xr.Dataset.resample()` requires a standard calendar.

**Mitigation:**
- Check calendar attribute in `_validate_dataset()`.
- Convert non-standard calendars via `xr.coding.times.decode_cf_datetime()`.

---

## 6. Numerical Accuracy

### Risk: Floating Point Accumulation

WRT uses `float32` (ship.py:77). Aggregating large float32 arrays accumulates rounding errors.

**Mitigation:**
- Upcast to `float64` before reduction, downcast result if needed:
```python
result = self.ds[variables].astype(np.float64).mean(dim=dim)
```
- Configurable via `precision` parameter.

### Risk: Distribution Fitting Convergence Failures

`scipy.stats.weibull_min.fit()` can fail for small samples, zero-heavy data, or extreme outliers.

**Mitigation:**
- Pre-filter: `len(valid) >= min_samples` (default: 30).
- Provide method-of-moments initial estimates.
- Return `NaN` parameters + goodness-of-fit metrics for failed cells.
- Cap optimizer iterations (`max_iter=200`).

---

## 7. Backward Compatibility

### Risk: Breaking Existing `WeatherCond` API

Chunked loading could break code that assumes eager numpy arrays.

**Mitigation:**
- Opt-in via `ENABLE_DASK_CHUNKS: bool = False` (default off).
- All existing `.to_numpy()` calls (15+ locations) are Dask-compatible (trigger computation automatically).
- All existing tests must pass with both `ENABLE_DASK_CHUNKS=True` and `False`.

### Risk: Dask Compatibility Assumptions

The claim that all xarray operations in WRT are Dask-compatible is **unverified**.

**Mitigation:**
- **Week 2 of Community Bonding:** Run every xarray operation used in WRT against a chunked dataset. Create a compatibility matrix. Document which operations require `.compute()` barriers.
- Known concerns: `.interp_like()` may trigger full materialization. `xr.merge()` with different chunk structures may produce unexpected results.

---

## 8. Integration Risks

### Risk: Increased Import Time

The `statistics/` module must not increase import time for users who don't need statistics.

**Mitigation:**
- Lazy imports for optional dependencies (`zarr`, `dask.distributed`).
- The module is only imported when explicitly requested via config or CLI.

### Risk: Configuration Complexity

Adding statistical parameters to the 534-line `Config` class risks overwhelming users.

**Mitigation:**
- Separate `StatisticsConfig` class, only instantiated when needed.
- Sensible defaults require zero configuration for basic usage.
