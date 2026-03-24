# Contribution issue ideas (code-backed) — WeatherRoutingTool

Generated: 2026-02-20

This document lists **PR-sized issues** based on **actual code patterns found in this repository**, optimized for **high likelihood of acceptance** (small surface area, clear bug/robustness wins, and/or tests).

## Best “fast-merge” PR candidates (pick one)

1. **Fix `Path(path).exists` misuse in config loaders** (Issue 1)
2. **Fix `ShipFactory.get_ship` error message (`SHIP_TYPE` → `BOAT_TYPE`)** (Issue 4)
3. **Fail fast for unknown `WeatherFactory.get_weather(data_mode=...)`** (Issue 12)
4. **Fix float index bug in `WeatherCondFromFile.get_wind_vector`** (Issue 10)
5. **Fix `ConstraintsList.safe_endpoint` list/ndarray accumulation bug** (Issue 5)

---

## Issue 1 — Fix `Path(path).exists` misuse in config loaders

- **Type**: bugfix
- **Why it matters**: `Path(path).exists` is a *method object*; not calling it makes the check always truthy. This can silently accept invalid paths and later fail in confusing ways.
- **Evidence**
  - `WeatherRoutingTool/config.py` L191–L200 (`if Path(path).exists:` then `with path.open(...)`)
  - `WeatherRoutingTool/ship/ship_config.py` L68–L71 (`if ... and Path(path).exists:` then `with path.open(...)`)
- **Proposal**
  - Normalize once: `path = Path(path)` (for `str`/`Path` callers).
  - Use `path.exists()` (call) and `path.open(...)`.
  - In `ShipConfig.assign_config`, keep the existing control flow but fix the `exists()` call and path normalization.
- **Acceptance criteria**
  - Missing config path raises a clear `ValueError` (not a later `AttributeError` / `FileNotFoundError`).
  - Works whether caller passes `str` or `Path`.
- **Estimated effort**: S
- **Merge likelihood**: **High**
- **Notes/risks**: Also consider making `CONFIG_PATH` consistently a `Path` (or consistently a `str`) across the codebase.

---

## Issue 2 — Make `ARRIVAL_TIME` validation consistent (remove invalid sentinel string)

- **Type**: bugfix
- **Why it matters**: `ARRIVAL_TIME` is typed as `datetime` but defaults to the string `'9999-99-99T99:99Z'` (not parseable). The validator also emits an incorrect error message mentioning only `DEPARTURE_TIME`.
- **Evidence**
  - `WeatherRoutingTool/config.py` L59 (string sentinel default)
  - `WeatherRoutingTool/config.py` L210–L220 (datetime validator error message)
  - `WeatherRoutingTool/config.py` L477–L488 (logic compares `ARRIVAL_TIME` to the sentinel string)
- **Proposal**
  - Change `ARRIVAL_TIME` to `Optional[datetime] = None`.
  - Update `parse_and_validate_datetime` to accept `None` and to raise a correct message for both fields.
  - Update `check_speed_determination` to use `self.ARRIVAL_TIME is None` instead of string comparisons.
- **Acceptance criteria**
  - Config validation succeeds with `ARRIVAL_TIME` omitted.
  - Config validation rejects “both `ARRIVAL_TIME` and `BOAT_SPEED` specified”.
  - Config validation rejects “neither specified”.
- **Estimated effort**: S–M
- **Merge likelihood**: **High**
- **Notes/risks**: This is a config schema change; if backward compatibility is needed, accept the old sentinel string as an input alias (and convert it to `None`).

---

## Issue 3 — Fix boat speed sentinel handling (`Quantity` vs scalar comparison)

- **Type**: bugfix
- **Why it matters**: `RoutingAlg` stores `boat_speed` as an astropy `Quantity`, but compares it to the scalar `-99`. This can behave unexpectedly and let “unset” speeds propagate as `-99 m/s`.
- **Evidence**
  - `WeatherRoutingTool/algorithms/routingalg.py` L53–L58 (`self.boat_speed = ... * u.meter/u.second`; then `if self.boat_speed == -99`)
- **Proposal**
  - Compare `self.boat_speed.value == -99.0` (or store `boat_speed_raw` separately).
  - Consider returning `None` consistently when speed is unset.
- **Acceptance criteria**
  - `get_boat_speed()` returns `None` when config speed is the sentinel.
  - No unit warnings/errors from comparing `Quantity` to scalar.
- **Estimated effort**: S
- **Merge likelihood**: **High**
- **Notes/risks**: Genetic algorithm already does `self.boat_speed.value == -99.`; aligning behavior is a good consistency win.

---

## Issue 4 — Fix error path in `ShipFactory.get_ship` (`SHIP_TYPE` → `BOAT_TYPE`)

- **Type**: bugfix
- **Why it matters**: Unknown boat types should raise a clear `NotImplementedError`, but the current code references `config.SHIP_TYPE` (not in `Config`), causing an `AttributeError`.
- **Evidence**
  - `WeatherRoutingTool/ship/ship_factory.py` L32–L34
- **Proposal**
  - Replace `config.SHIP_TYPE` with `config.BOAT_TYPE`.
  - Optional: change the chain of `if` to `elif` for clarity (only one ship type can match).
- **Acceptance criteria**
  - Unknown `BOAT_TYPE` triggers `NotImplementedError` with the correct type included.
- **Estimated effort**: S
- **Merge likelihood**: **High**

---

## Issue 5 — Fix `ConstraintsList.safe_endpoint` list/ndarray accumulation bug

- **Type**: bugfix
- **Why it matters**: `safe_endpoint()` starts with a Python list, then does `is_constrained += is_constrained_temp` where `is_constrained_temp` is array-like. With lists, `+=` extends the list rather than doing elementwise boolean accumulation.
- **Evidence**
  - `WeatherRoutingTool/constraints/constraints.py` L305–L340 (list init + `is_constrained += is_constrained_temp` at L339)
- **Proposal**
  - Standardize `is_constrained` to a `np.ndarray[bool]` inside `safe_endpoint`.
  - Accumulate via elementwise OR: `is_constrained = is_constrained | is_constrained_temp`.
  - Return type should be consistent with callers (`list` or `np.ndarray`)—but be consistent across `safe_*` methods.
- **Acceptance criteria**
  - Multi-constraint endpoint checks produce a boolean vector of the expected length.
  - Type doesn’t unexpectedly change across `shall_I_pass` → `safe_endpoint` → `safe_crossing`.
- **Estimated effort**: S–M
- **Merge likelihood**: **High**
- **Notes/risks**: Touches types across constraint APIs; update tests in `tests/test_constraints.py` if needed.

---

## Issue 6 — Replace runtime `assert` checks with explicit exceptions

- **Type**: refactor / bugfix
- **Why it matters**: `assert` can be disabled with Python `-O`, removing safety checks in production. Several asserts validate critical invariants (route endpoints, route validity, input shape).
- **Evidence**
  - `WeatherRoutingTool/algorithms/genetic/population.py` L50–L53 and L79–L81
  - `WeatherRoutingTool/algorithms/genetic/mutation.py` L269–L272 (shape checks)
- **Proposal**
  - Replace asserts with `ValueError`/`RuntimeError` with clear messages.
  - For numeric comparisons, consider tolerances (`np.allclose`) where appropriate.
- **Acceptance criteria**
  - Invalid inputs raise exceptions regardless of Python optimization flags.
- **Estimated effort**: M
- **Merge likelihood**: **Medium–High**
- **Notes/risks**: Choose exception types carefully so downstream callers can handle them.

---

## Issue 7 — Fix `RouteParams.__eq__` (references non-existent attributes)

- **Type**: bugfix
- **Why it matters**: `RouteParams.__eq__` references `self.fuel` and `self.rpm`, but these aren’t defined on the class (likely moved into `ship_params_per_step`). This makes equality comparisons broken and can also hide real regressions in tests.
- **Evidence**
  - `WeatherRoutingTool/routeparams.py` L85–L108 (`self.fuel`, `self.rpm`)
- **Proposal**
  - Either remove `__eq__` entirely, or implement it against real/stable fields (and for floats consider tolerance).
  - If deep equality is needed, compare `ship_params_per_step` content explicitly (or a stable subset).
- **Acceptance criteria**
  - `RouteParams` comparisons don’t crash and behave deterministically.
- **Estimated effort**: S–M
- **Merge likelihood**: **High**
- **Notes/risks**: Avoid very strict float equality if results are expected to vary slightly.

---

## Issue 8 — Reduce repeated netCDF open + per-point `sel` loops in `Boat.evaluate_weather`

- **Type**: performance
- **Why it matters**: `evaluate_weather()` opens the dataset on every call and loops coordinate-by-coordinate, doing many `.sel(..., method='nearest')` calls. Routing will call this frequently; this is a likely performance bottleneck.
- **Evidence**
  - `WeatherRoutingTool/ship/ship.py` L42–L96 (`xr.open_dataset` at L43 + per-point loop at L58–L82)
- **Proposal**
  - Keep the dataset open (cache it on the `Boat` instance), and close it when done (context manager or explicit close).
  - Use vectorized xarray selection/interp for arrays of `latitude`, `longitude`, and `time` rather than per-point loops.
- **Acceptance criteria**
  - Same outputs for a fixed input (within tolerance).
  - Reduced runtime for many points (can add a lightweight perf test/benchmark).
- **Estimated effort**: M–L
- **Merge likelihood**: **Medium**
- **Notes/risks**: xarray indexing with time arrays needs care to avoid loading entire datasets.

---

## Issue 9 — Make `WaterDepth.load_data_from_file` / `check_depth` more memory/perf friendly

- **Type**: performance
- **Why it matters**: The code has explicit FIXME notes about potentially loading the whole file. `check_depth()` interpolates on every call; routing can call this frequently.
- **Evidence**
  - `WeatherRoutingTool/constraints/constraints.py` L655–L663 (FIXME + `xr.open_dataset(depth_path)` without subsetting)
  - `WeatherRoutingTool/constraints/constraints.py` L674–L679 (`interp` each call)
- **Proposal**
  - Subset by map bbox at load time (if `map_size` is known) and/or use chunking.
  - Vectorize depth checks for arrays of points instead of per-call scalar interpolation.
- **Acceptance criteria**
  - Identical results for in-bounds points.
  - Reduced memory footprint on large depth datasets.
- **Estimated effort**: M
- **Merge likelihood**: **Medium**

---

## Issue 10 — Fix `WeatherCondFromFile.get_wind_vector` dict indexing with float key

- **Type**: bugfix
- **Why it matters**: `get_time_step_index()` returns `idx` as a float; `wind_vectors` is keyed by integers. That can raise `KeyError` even when time aligns, or behave inconsistently due to float representation.
- **Evidence**
  - `WeatherRoutingTool/weather.py` L548–L552 (`idx = (...)` float)
  - `WeatherRoutingTool/weather.py` L593–L610 (`self.wind_vectors[idx]`)
- **Proposal**
  - Convert to `int` explicitly (and define rounding policy: floor vs nearest).
  - Validate bounds and improve error message (include allowed index range).
- **Acceptance criteria**
  - Exact-step times return wind vectors reliably.
  - Off-step times follow documented rounding policy.
- **Estimated effort**: S–M
- **Merge likelihood**: **High**

---

## Issue 11 — Make route postprocessing angle math robust (vertical segments / division by zero)

- **Type**: bugfix
- **Why it matters**: `calculate_slope()` divides by `(x2-x1)` and will crash for vertical lines; `calculate_angle_from_slope()` also has singularities. Real routes can contain vertical/near-vertical segments in lon/lat.
- **Evidence**
  - `WeatherRoutingTool/constraints/route_postprocessing.py` L643–L659 (`slope = (y2-y1)/(x2-x1)`)
  - `WeatherRoutingTool/constraints/route_postprocessing.py` L661–L676 (angle from slopes)
- **Proposal**
  - Replace slope-based logic with vector-based angle via `atan2` using direction vectors.
  - Add tests for vertical, horizontal, and near-degenerate segments.
- **Acceptance criteria**
  - No division-by-zero for vertical segments.
  - Angle results remain correct for typical cases.
- **Estimated effort**: M
- **Merge likelihood**: **Medium**
- **Notes/risks**: Keep CRS assumptions clear (EPSG:4326 is angular coordinates).

---

## Issue 12 — Fail fast for unknown `WeatherFactory.get_weather(data_mode=...)`

- **Type**: bugfix / DX
- **Why it matters**: `get_weather()` uses multiple `if` blocks and then unconditionally calls `wt.check_units()`. For an unknown mode, `wt` stays `None` and the code fails later with an attribute error instead of a clear “unsupported mode”.
- **Evidence**
  - `WeatherRoutingTool/weather_factory.py` L15–L65 (`wt = None` then `wt.check_units()` at L62)
- **Proposal**
  - Use `elif` chain + `else: raise ValueError(...)` listing supported modes.
  - Add a unit test for unknown `data_mode`.
- **Acceptance criteria**
  - Unknown mode raises `ValueError` with supported values.
- **Estimated effort**: S
- **Merge likelihood**: **High**

---

## Issue 13 — Decide what to do with `IsoChrone` (API mismatch / likely dead code)

- **Type**: refactor / docs
- **Why it matters**: `IsoChrone` constructor signature doesn’t match the current `IsoBased(config)` pattern and the file itself questions whether it’s in use. It’s also not selectable in `Config.ALGORITHM_TYPE`.
- **Evidence**
  - `WeatherRoutingTool/algorithms/isochrone.py` L12–L22 (constructor uses `start, finish, time, delta_time`)
  - `WeatherRoutingTool/config.py` L56–L58 (`ALGORITHM_TYPE` doesn’t include `isochrone`)
- **Proposal**
  - Either remove it (and any references), or modernize to `config`-based API and add config option + a minimal test.
  - If removal is risky for external users, deprecate first (document + warning).
- **Acceptance criteria**
  - No confusing/unused algorithm remains without documentation and a selection path.
- **Estimated effort**: M
- **Merge likelihood**: **Medium**

