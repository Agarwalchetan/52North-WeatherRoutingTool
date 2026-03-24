# Architecture Decisions

## ADR-1: xarray-Native Statistical Operations vs. Custom Data Structures

### Decision
All statistical computations will operate directly on `xr.Dataset` and `xr.DataArray` objects, using xarray's built-in aggregation methods (`.mean()`, `.std()`, `.quantile()`, `.groupby()`, `.resample()`).

### Context
The WRT weather module already stores all environmental data in `xr.Dataset` (see `WeatherCond.ds: xr.Dataset` at `weather.py:45`). Two approaches were considered:

1. **xarray-native**: Extend with statistical methods that accept and return xarray objects.
2. **Custom structures**: Extract numpy arrays, compute statistics in custom containers, wrap results.

### Rationale

| Criterion | xarray-Native | Custom Structures |
|-----------|---------------|-------------------|
| **Dimension awareness** | Automatic handling of `(time, latitude, longitude)` dims | Must manually track dimension semantics |
| **Coordinate preservation** | Results retain lat/lon/time coordinates for downstream use | Must manually attach coordinates |
| **Dask compatibility** | `.mean(dim='time')` is Dask-aware when chunked | Must implement parallelism from scratch |
| **NaN handling** | Built-in `skipna` parameter on all reductions | Must implement NaN filtering manually |
| **Metadata propagation** | Units, variable names, attributes preserved | Must manually copy metadata |
| **Interoperability** | Direct compatibility with WRT plotting and ship evaluation | Requires conversion layer |
| **Maintenance cost** | Zero—leverages upstream xarray releases | High—custom test/maintenance burden |

### Consequences
- All new code paths accept `xr.Dataset` as input and return `xr.Dataset` or `xr.DataArray`.
- Custom aggregation functions (e.g., distribution fitting) are wrapped via `xr.apply_ufunc()` to maintain Dask compatibility.
- The `WeatherStatisticsEngine` holds a reference to the same `xr.Dataset` stored in `WeatherCond.ds`, avoiding data duplication.

---

## ADR-2: Dask for Lazy Computation

### Decision
Integrate Dask as the parallel computation backend, using `xr.open_dataset(chunks={...})` for lazy loading. Dask integration is **opt-in** and requires empirical validation before production use.

### Context
Dask is already a dependency in `requirements.txt` (line 3: `dask`) but is **not used** anywhere in the WRT codebase for weather data processing. The only Dask usage is in `WaterDepth.load_data_automatic()` at `constraints.py:639`:

```python
depth_data_chunked = depth_data.chunk(chunks={"latitude": "100MB", "longitude": "100MB"})
```

### Rationale

**Why Dask over alternatives:**

| Alternative | Pros | Cons |
|-------------|------|------|
| **Pure NumPy** | No additional complexity | Cannot process datasets > RAM; no parallelism |
| **Dask** | Transparent parallelism via xarray; chunk-based memory control; already a dependency | Scheduler overhead for small datasets; lazy evaluation debugging is harder |
| **Vaex** | Fast out-of-core DataFrames | Not compatible with xarray's multi-dimensional model |
| **Ray** | Distributed computing | Not integrated with xarray; requires significant wrapping |

### Dask Compatibility: Known Limitations

> **CRITICAL:** Not all xarray operations in WRT are guaranteed Dask-compatible. The following must be empirically tested during Community Bonding (Week 2):

| WRT Operation | Expected Dask Behavior | Risk |
|---------------|----------------------|------|
| `.sel(method='nearest')` | **Works** — per-chunk selection, returns Dask array | Low |
| `.interp_like()` | **May trigger full materialization** — interpolation across chunk boundaries requires loading adjacent chunks | **High** |
| `xr.merge()` with mixed chunk shapes | **Works but may produce inefficient task graphs** — Dask unifies chunk sizes, potentially creating many small tasks | Medium |
| `.coarsen()` | **Works** — boundary handling via `boundary='trim'` | Low |
| `.where()` | **Works** — element-wise, no cross-chunk dependency | Low |
| `RegularGridInterpolator` (scipy) | **Requires `.compute()` first** — scipy does not accept Dask arrays | High (existing `.to_numpy()` barriers handle this) |

**Week 2 deliverable:** A Dask compatibility matrix based on running each operation against a chunked version of a real WRT weather file. Results will confirm or revise this table.

### Consequences
- Weather data loading becomes near-instantaneous (metadata read only) when `chunks` is specified.
- `.compute()` barriers must be placed before numpy-only code paths (already exists at 15+ `.to_numpy()` call sites).
- `Boat.evaluate_weather()` (which uses `.sel()`) works unchanged with Dask-backed datasets — `.sel(method='nearest')` returns a scalar, triggering per-element computation.

---

## ADR-3: Storage Format Strategy (Zarr, NetCDF, Kerchunk)

### Decision
Support **three** access patterns with two storage formats:
1. **NetCDF** — retained for backward compatibility with existing weather data and WRT ecosystem.
2. **Zarr** — recommended for statistical analysis results (chunk-aligned writes, appendable).
3. **Kerchunk** — for accessing existing large NetCDF archives with Zarr-like performance without data conversion.

### Context
WRT currently uses NetCDF exclusively:
- `WeatherCondEnvAutomatic.write_data()` → `self.ds.to_netcdf(filepath)` (weather.py:321)
- `WeatherCondODC.write_data()` → `self.ds.to_netcdf(filepath)` (weather.py:696)

### Rationale

| Criterion | NetCDF4 | Zarr | Kerchunk (virtual Zarr over NetCDF) |
|-----------|---------|------|-------------------------------------|
| **Chunk-aligned writes** | Requires writing full variable | Native per-chunk writes | Read-only (references existing NetCDF) |
| **Parallel writes** | Possible with MPI-enabled HDF5 (`h5py` + `mpi4py`), but not via default xarray API | Native (each chunk independent) | N/A (read-only) |
| **Cloud storage** | Requires full file download | Direct chunk access via `fsspec` | Direct chunk access via `fsspec` |
| **Existing archive access** | Native | Requires conversion | **No conversion required** — creates a JSON reference file pointing to NetCDF chunks |
| **xarray API** | `xr.open_dataset()` | `xr.open_zarr()` | `xr.open_dataset(engine='kerchunk')` |

**Kerchunk rationale:** When users have existing multi-GB NetCDF archives (e.g., ERA5 reanalysis), converting to Zarr is expensive and doubles storage. Kerchunk creates a lightweight JSON reference file that maps Zarr-style chunk requests to byte ranges in the original NetCDF. This provides Zarr-like random access with zero data duplication.

### Consequences
- Statistical results default to Zarr for new outputs.
- Existing NetCDF weather files continue to work unchanged.
- `kerchunk` is an optional dependency (not required for core functionality).
- The `DataStorageManager` auto-detects format from file path extension (`.zarr` directory → Zarr, `.nc` file → NetCDF, `.json` reference → Kerchunk).

---

## ADR-4: Precompute vs. On-Demand Statistics with Content-Hash Cache

### Decision
Support **both** on-demand and precomputed statistics, with a content-hash-based cache that works for both file-backed and in-memory datasets.

### Rationale

| Mode | Use Case | Pros | Cons |
|------|----------|------|------|
| **On-demand** | Interactive exploration, one-off analysis | No wasted computation; always current | Repeated queries re-compute |
| **Precompute + cache** | Dashboards, repeated comparisons | Instant after first run | Storage cost; stale if source changes |

### Implementation

```python
class StatisticsCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_source_hash(self, ds: xr.Dataset) -> str:
        """Content-aware hash that works for file-backed AND in-memory datasets."""
        import hashlib
        h = hashlib.sha256()
        # Include dimension sizes
        for dim, size in ds.dims.items():
            h.update(f"{dim}={size}".encode())
        # Include variable names and dtypes
        for var in sorted(ds.data_vars):
            h.update(f"{var}:{ds[var].dtype}".encode())
        # Include coordinate bounds (not full data — too expensive)
        for coord in ['time', 'latitude', 'longitude']:
            if coord in ds.coords:
                h.update(str(float(ds[coord].min())).encode())
                h.update(str(float(ds[coord].max())).encode())
        return h.hexdigest()[:16]

    def get(self, operation_key: str, source_hash: str) -> Optional[xr.Dataset]:
        path = self.cache_dir / f"{source_hash}_{operation_key}.zarr"
        if path.exists():
            return xr.open_zarr(path)
        return None

    def put(self, operation_key: str, source_hash: str, result: xr.Dataset):
        path = self.cache_dir / f"{source_hash}_{operation_key}.zarr"
        # .compute() to materialize Dask arrays before serialization
        result.compute().to_zarr(path, mode='w', consolidated=True)
```

**Key fix from review:** The cache now calls `.compute()` before serialization (Dask lazy arrays cannot be written to Zarr without materializing first). The cache key uses a content hash based on dataset structure and coordinate bounds, not file mtime — this works for in-memory and cloud-backed datasets.

### Consequences
- Cache is invalidated when dataset structure changes (dimensions, variables, coordinate ranges).
- .compute() is called explicitly before cache writes — no silent lazy-object serialization.
- Cache format is always Zarr (chunk-aligned, efficient for partial reads).

---

## ADR-5: Module Placement within WRT Architecture

### Decision
Create `WeatherRoutingTool/statistics/` (peer to `algorithms/`, `constraints/`, `ship/`, `utils/`), with `statistics/` depending on `weather.py` but **not** on `algorithms/` or `ship/`.

### Architecture

```
                 ┌─────────────┐
                 │ config.py   │  (configuration layer)
                 └──────┬──────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐   ┌──────────────┐  ┌────────────┐
   │ weather │◄──│ statistics/  │  │constraints/│  (data/domain layer)
   └────┬────┘   └──────┬───────┘  └─────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
               ┌────────────────┐
               │  algorithms/   │  (computation layer)
               │                │
               │  (optionally   │
               │   imports      │
               │  statistics/)  │
               └────────────────┘
```

### Consequences
- `statistics/` imports from `weather.py` (to access `WeatherCond.ds`).
- `algorithms/` can optionally import from `statistics/` for route analysis.
- Independently testable with mock `xr.Dataset` objects.
- No circular dependencies.

---

## ADR-6: Error Handling in Chunked Statistical Operations

### Decision
Use a **fail-per-chunk, report-and-continue** strategy for statistical operations on Dask arrays. Individual chunk failures are recorded as NaN in the output with warnings logged, rather than aborting the entire computation.

### Context
When running distribution fitting or anomaly detection across thousands of spatial chunks, individual chunks may fail due to:
- Insufficient valid (non-NaN) data points in a chunk (e.g., land-only chunks).
- Numerical convergence failures in `scipy.stats.fit()`.
- Corrupted data values (Inf, extremely large values).

### Implementation

```python
def _safe_fit_cell(data, dist_func, min_samples=30):
    """Per-cell fitting with error containment."""
    valid = data[~np.isnan(data)]
    if len(valid) < min_samples:
        return np.array([np.nan] * 5)  # NaN result for insufficient data
    try:
        params = dist_func.fit(valid, method='mle')
        ks_stat, p_val = scipy.stats.kstest(valid, dist_func.name, args=params)
        padded = list(params) + [np.nan] * (3 - len(params))
        return np.array(padded[:3] + [ks_stat, p_val])
    except (RuntimeError, ValueError, FloatingPointError) as e:
        logger.warning(f"Distribution fit failed: {e}")
        return np.array([np.nan] * 5)  # NaN result for failed fit
```

When used with `xr.apply_ufunc(dask='parallelized')`, each chunk's cells are processed independently. A failure in one cell/chunk does not abort the computation — the output contains NaN for failed cells, with a summary log reporting the failure rate.

### Consequences
- No silent data loss — failed cells are explicitly NaN.
- The engine reports a `fit_failure_rate` metric after distribution fitting.
- Users can inspect which spatial cells failed via `result.isnull().sum()`.
- For aggregation operations (mean, std), xarray's built-in error handling with `skipna=True` is sufficient.
