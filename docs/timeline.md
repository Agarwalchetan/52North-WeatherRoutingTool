# Timeline

## GSoC 2026 Schedule

> **Important:** The dates below are illustrative. Actual GSoC 2026 dates will be confirmed by Google. The structure (4 phases + buffer) is designed to be date-agnostic.

### Community Bonding: Weeks 1–4

| Week | Activities | Deliverables |
|------|-----------|-------------|
| **Week 1** | Profile existing weather data loading with `cProfile` and `tracemalloc`. Run `read_dataset()`, `evaluate_weather()`, full IsoFuel routing. Establish **measured** baseline metrics. | Baseline performance report with **real** timing/memory measurements. This replaces all estimated numbers in goals.md. |
| **Week 2** | Test Dask chunking strategies: open real WRT weather data with `chunks={}`. Test every xarray operation used in WRT (`.sel()`, `.interp_like()`, `.merge()`, `.coarsen()`, `.where()`) with chunked datasets. Document which operations work, which fail, which materialize. | **Dask compatibility matrix** — empirical evidence, not assumptions. |
| **Week 3** | Set up dev environment (mypy, pytest, sphinx). Create synthetic test datasets (small: 50 MB, medium: 500 MB). Create `FakeWeather`-based test fixtures for statistics module. | Test infrastructure, synthetic data generator, CI integration. |
| **Week 4** | Finalize API design with mentor. Review ADRs. Agree on scope: which stretch goals are in/out. | Approved API specification. Signed-off feature scope. |

---

### Phase 1 — Core Infrastructure + Aggregation: Weeks 5–8

**Objective:** Build `statistics/` module skeleton, implement Dask-chunked loading, and deliver core aggregation operations.

| Week | Focus | Deliverables | Validation |
|------|-------|-------------|-----------|
| **Week 5** | **Dask integration first (highest risk).** Modify `WeatherCondFromFile.read_dataset()` to accept optional `chunks` parameter. Implement `ChunkOptimizer`. Verify that existing routing pipeline works unchanged with chunked datasets by running all existing tests. | Modified `weather.py`, `chunk_optimizer.py`. All existing tests pass. | `pytest tests/` — 100% pass rate. Memory profiling on chunked vs. eager. |
| **Week 6** | Create `WeatherRoutingTool/statistics/` module structure. Implement `WeatherStatisticsEngine` with `__init__()`, dataset validation, NaN-aware `compute_mean()`, `compute_std()`, `compute_percentile()`. | `engine.py`, `aggregation.py`. Unit tests. | Numerical verification against `numpy.nanmean()` on small datasets. Dask-vs-eager result equality tests. |
| **Week 7** | Implement `compute_spatial_mean()` with cosine-latitude area weighting. Add `compute_rolling()` and `compute_resample()` for temporal analytics. | `temporal.py` with rolling/resampling. | Compare rolling output against `pandas.DataFrame.rolling()` via `.to_dataframe()`. |
| **Week 8** | Implement `DataStorageManager` with NetCDF/Zarr save/load. Implement basic `StatisticsCache`. | `storage.py` with round-trip tests. | Save → reload → compare for both formats. |

**Milestone 1 checkpoint (Week 8):** Core aggregation and temporal analytics functional. Dask integration tested and verified. Storage working.

---

### Phase 2 — Advanced Analytics: Weeks 9–12

**Objective:** Distribution fitting, correlation analysis, and route weather analysis.

| Week | Focus | Deliverables | Validation |
|------|-------|-------------|-----------|
| **Week 9** | Implement `compute_distribution()` for Weibull/Rayleigh/Normal fitting per spatial cell via `xr.apply_ufunc(dask='parallelized')`. Handle NaN cells, small samples (< 30 points). | `distributions.py` with fitting + goodness-of-fit (KS statistic). | Fit synthetic Weibull data with known parameters, verify recovery within 10% tolerance. |
| **Week 10** | Implement `compute_correlation()` using `xr.corr()` for Dask-compatible pairwise correlation. Add spatial correlation via per-pixel `xr.corr()` along time dimension. | `correlation.py` with full and spatial correlation. | Compare against `numpy.corrcoef()` on materialized data. Verify Dask compatibility (no `.values.ravel()`). |
| **Week 11** | Implement `StatisticalRouteAnalyzer`: extract weather conditions along computed routes, compute per-route statistics (mean wave height, max wind, risk score). Integrate with `execute_routing.py` (optional post-routing block). | `route_analysis.py`, modified `execute_routing.py`. | End-to-end test: run IsoFuel routing → extract route weather stats → verify values match manual `.sel()` lookups. |
| **Week 12** | Implement visualization: spatial heatmaps (reuse WRT's cartopy patterns), time-series plots, distribution histograms, correlation matrices. Use xarray's `.plot()` where possible. | `visualization.py` with 4 plot types. | Visual inspection. Smoke tests (plot generation without error). |

**Midterm evaluation (Week 12):** Core statistical methods, route analysis, and visualization functional. All Dask-compatible.

---

### Phase 3 — Integration, CLI, and Stretch Goals: Weeks 13–16

**Objective:** CLI interface, configuration integration, and stretch goals if on schedule.

| Week | Focus | Deliverables | Validation |
|------|-------|-------------|-----------|
| **Week 13** | Implement CLI interface integrated with existing `cli.py`. Add `StatisticsConfig` pydantic model. Wire into `Config` system. | CLI tool, config model, updated docs. | `python -m WeatherRoutingTool.statistics --help` works. Config validation tests. |
| **Week 14** | Comprehensive integration testing: run full WRT routing pipeline with statistics module active. Test with multiple data modes (`from_file`, `automatic` if test data available). | Integration test suite. | No regressions. Memory/runtime within targets from Week 1 baselines. |
| **Week 15** | **Stretch:** If on schedule, implement anomaly detection (Z-score, IQR). If behind, use this week for debugging and polish. | `anomaly.py` OR bug fixes and edge case handling. | Inject known anomalies into synthetic data, verify detection. |
| **Week 16** | **Stretch:** If on schedule, implement basic ensemble statistics (`compute_ensemble_mean()`, `compute_ensemble_spread()`) for pre-prepared multi-member NetCDF files. If behind, focus on documentation and testing. | `ensemble.py` OR comprehensive documentation. | Test with synthetic 10-member ensemble. |

**Milestone 2 checkpoint (Week 16):** Full system integrated with CLI. Stretch goals delivered if timeline permitted.

---

### Phase 4 — Benchmarking, Documentation, and Final: Weeks 17–19

| Week | Focus | Deliverables | Validation |
|------|-------|-------------|-----------|
| **Week 17** | Comprehensive benchmark suite. Compare all metrics against Week 1 baselines. Generate before/after tables with real data. | `benchmarks/` directory with reproducible scripts and measured results. | All core success metrics from goals.md met. |
| **Week 18** | Performance tuning. Write comprehensive API documentation (Sphinx). Create tutorial notebook demonstrating all features. | Sphinx docs, tutorial notebook, usage examples. | Mentor review of documentation. |
| **Week 19** | Final code review, clean git history, passing CI, edge case testing. Prepare final submission. | Clean PR, passing tests, mentor sign-off. | All tests green. Documentation complete. |

**Final evaluation (Week 19).**

---

## Risk Buffer

- Weeks 15-16 serve as **buffer weeks** if core features run over schedule.
- Stretch goals (anomaly detection, ensemble support) are explicitly optional — core value is delivered by Week 14.
- Week 19 is dedicated to polish, not new features.

## Scope Control Rules

| Situation | Action |
|-----------|--------|
| 2+ weeks behind at midterm | Drop all stretch goals. Focus on core + testing + docs. |
| 1 week behind at midterm | Drop ensemble support. Keep anomaly detection if time. |
| On schedule at midterm | Proceed with stretch goals. |
| Ahead of schedule | Add property-based testing (hypothesis). Add `kerchunk` support for NetCDF access optimization. |

## Communication Plan

| Frequency | Channel | Content |
|-----------|---------|---------|
| Daily | Async on project board | Brief progress notes, blockers |
| Weekly | 30-min video call | Demo progress, next week plan |
| Bi-weekly | Written report | Technical progress with benchmarks |
| At milestones | Pull request | Code review of completed phase |
