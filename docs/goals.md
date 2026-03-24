# Goals and Success Metrics

## 1. Feature Goals Mapped to System Gaps

### 1.1 Core Statistical Analysis (Must-Have)

| Feature | System Gap Addressed | Technical Scope |
|---------|---------------------|----------------|
| **Spatial/Temporal Aggregation** | No mean/median/std/percentile over weather data | xarray `.mean()`, `.std()`, `.quantile()` over `(time, latitude, longitude)` dims with `skipna=True` |
| **Distribution Fitting** | No characterization of variable distributions | `scipy.stats` Weibull/Rayleigh/Normal fitting per spatial cell via `xr.apply_ufunc(dask='parallelized')` |
| **Correlation Analysis** | No cross-variable correlation computation | `xr.corr()` pairwise across weather variables (Dask-compatible, avoids `.values.ravel()` materialization) |
| **Temporal Analytics** | Weather consumed at fixed 3h intervals only | `xr.Dataset.resample()`, `.rolling()` with configurable windows for temporal smoothing and resampling |
| **NaN-Aware Processing** | Only `fillna(0)` at ship.py:98 — statistically incorrect | Explicit `skipna=True` in all reductions; land-mask-aware spatial means using `xr.DataArray.where()` from VHM0 masking |

### 1.2 Performance Infrastructure (Must-Have)

| Feature | System Gap Addressed | Technical Scope |
|---------|---------------------|----------------|
| **Lazy Data Loading** | `xr.open_dataset()` eagerly loads all data (weather.py:619) | `xr.open_dataset(filepath, chunks={...})` via opt-in `ENABLE_DASK_CHUNKS` config |
| **Vectorized Weather Lookup** | Sequential `.sel()` loop in ship.py:52-75 | `xr.DataArray` vectorized indexing along `points` dimension |
| **Zarr Backend** | NetCDF-only storage | `xr.Dataset.to_zarr()` / `xr.open_zarr()` for statistical result persistence |

### 1.3 Stretch Goals (If Time Permits)

| Feature | Risk Level | Why Stretch |
|---------|-----------|-------------|
| **Ensemble Statistics** | High | Requires adding `ensemble` dimension to dataset schema, affects downstream `sel()`/`plot()` calls |
| **Anomaly Detection** | Medium | Z-score/IQR methods are straightforward, but defining "anomalous" in ocean weather context requires domain expertise |
| **Route Risk Scoring** | Medium | Integration with genetic algorithm's objective function changes optimization landscape |

---

## 2. Success Metrics

### 2.1 Memory Reduction

> **IMPORTANT:** All baseline numbers below are **estimates based on dataset size calculations**, not measured values. Actual profiling will be performed during Community Bonding (Week 1) to establish real baselines.

| Metric | Estimated Baseline | Target | Measurement Method |
|--------|-------------------|---------|--------------------| 
| Peak RSS during weather data load | ~Full dataset size (estimated 300-500 MB for typical 5-day North Atlantic at 0.083°, 11 float32 vars, ~40 time steps × 420 × 480 spatial grid) | ≤ 2× configured chunk size via lazy loading | `tracemalloc` peak tracking around `read_dataset()` |
| Memory during `compute_mean(dim='time')` | N/A (no stats module) | ≤ 3× single chunk size (chunk in + reduction intermediate + result) | `tracemalloc` peak during operation |
| Memory during pairwise correlation | N/A | ≤ 2 × single variable size × 2 (using `xr.corr()` which operates pair-wise, not full materialization) | `memory_profiler` line-by-line |

### 2.2 Runtime Targets

| Metric | Estimated Baseline | Target | How To Verify |
|--------|-------------------|--------|--------------|
| Weather data load time | Estimated proportional to file size (~1-3s for 500 MB NetCDF on SSD) | ≤ 100ms for lazy open (metadata-only read) | `time.perf_counter()` around `xr.open_dataset(chunks=...)` |
| Per-step weather lookup (600 candidates, 11 vars, IsoFuel) | Estimated 11 × 600 sequential `.sel()` calls | 11 vectorized `.sel()` calls (Python loop overhead eliminated; per-element xarray cost remains) | A/B benchmark: loop vs. vectorized on same dataset |
| `compute_mean(dim='time')` on 500 MB dataset with Dask | N/A | To be determined by benchmarking — expected sub-second for 4-core machine | Benchmark suite with `time.perf_counter()` |

### 2.3 Dataset Scalability

| Metric | Current Limit | Target | Validation |
|--------|---------------|--------|-----------|
| Maximum dataset size | Limited by RAM (~2-4 GB) | ≥ 10 GB via Dask chunked processing | Process a 10 GB synthetic dataset (generated via `xr.Dataset` with large arrays) without OOM |
| Maximum time range | ~5-7 days at 3h resolution | Months of reanalysis data | Process a 3-month ERA5-like synthetic dataset |
| Native resolution tracking | Not tracked (merged datasets lose resolution provenance) | Metadata attribute per variable recording native resolution | `result.attrs['native_resolution']` populated |

### 2.4 Code Quality Metrics

| Metric | Target |
|--------|--------|
| Test coverage for `statistics/` module | ≥ 85% line coverage (measured via `pytest-cov`) |
| Docstring coverage | 100% public API (enforced via `pydocstyle`) |
| Type annotation coverage | 100% public methods (verified via `mypy --strict` on `statistics/`) |
| Backward compatibility | All existing tests in `tests/` pass unchanged |
| NaN correctness | All statistical functions tested with datasets containing 10-50% NaN values |

---

## 3. Non-Goals

- **Real-time streaming ingestion** from live weather APIs (batch/file-based analysis only).
- **Machine learning model training** for weather prediction.
- **QGIS plugin UI development** (Python API and CLI only; QGIS integration is separate).
- **Modifying existing routing algorithm logic** (statistical module wraps routing, not replaces it).
- **GRIB2 format support** (WRT only supports NetCDF; ensemble data must be pre-converted).
- **Ensemble data ingestion from ECMWF** (WRT uses `maridatadownloader` which supports GFS/CMEMS only; ensemble support assumes pre-prepared multi-member NetCDF files).
