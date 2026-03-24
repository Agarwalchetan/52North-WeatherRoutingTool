# 01 — System Analysis

## 1. Weather Module Files

| File | Lines | Purpose |
|------|-------|---------|
| `WeatherRoutingTool/weather.py` | 900 | Core weather data classes: `WeatherCond` (base), `WeatherCondFromFile`, `WeatherCondEnvAutomatic`, `WeatherCondODC`, `FakeWeather` |
| `WeatherRoutingTool/weather_factory.py` | 69 | Factory pattern for weather initialization based on `data_mode` |
| `WeatherRoutingTool/config.py` | 534 | Pydantic-based configuration with weather data validation |
| `WeatherRoutingTool/utils/unit_conversion.py` | 178 | Time rounding, angle conversion, dataset consistency checks |

## 2. Routing Module Files

| File | Lines | Purpose |
|------|-------|---------|
| `WeatherRoutingTool/algorithms/routingalg.py` | 113 | Base `RoutingAlg` class: start/finish, GCR computation, boat speed |
| `WeatherRoutingTool/algorithms/isobased.py` | 1,893 | `IsoBased` class: iterative routing with pruning, course expansion, constraint checking |
| `WeatherRoutingTool/algorithms/isofuel.py` | 169 | `IsoFuel`: constant-fuel routing steps |
| `WeatherRoutingTool/algorithms/genetic/problem.py` | 182 | `RoutingProblem` (pymoo): multi-objective fuel + arrival time optimization |
| `WeatherRoutingTool/algorithms/data_utils.py` | 140 | `GridMixin` for grid sampling, speed calculation from arrival time |
| `WeatherRoutingTool/algorithms/routingalg_factory.py` | ~50 | Algorithm selection factory |

### 2.1 GridMixin Analysis (Previously Missing)

The `GridMixin` class in `data_utils.py` provides spatial grid operations used by the genetic algorithm:

```python
class GridMixin:
    def get_grid_coords(self, ds, n_grid_points):
        """Extracts evenly-spaced lat/lon grid from dataset for route sampling."""
```

**Relevance to statistics module:** `GridMixin` demonstrates an existing pattern for spatial subsampling from `xr.Dataset`. The statistics module's `subset()` method follows a similar approach but uses `xr.Dataset.sel(slice(...))` for coordinate-based selection rather than index-based sampling.

## 3. Ship Module Files

| File | Lines | Purpose |
|------|-------|---------|
| `WeatherRoutingTool/ship/ship.py` | 163 | `Boat` base class with `evaluate_weather()` and `approx_weather()` |
| `WeatherRoutingTool/ship/direct_power_boat.py` | 452 | `DirectPowerBoat`: Fujiwara wind resistance, power estimation |
| `WeatherRoutingTool/ship/shipparams.py` | ~300 | `ShipParams` container for fuel rate, resistances, weather conditions |

### 3.1 Weather Coupling by Ship Model

| Ship Model | Weather Coupling | Impact on Statistics |
|------------|-----------------|---------------------|
| `DirectPowerBoat` | **Full**: wind resistance (Fujiwara) + wave resistance + current correction | Statistical analysis of wind/wave fields directly affects fuel predictions |
| `ConstantFuelBoat` | **None**: fixed fuel rate ignoring weather | Statistical module provides route weather characterization without affecting fuel calculation |

## 4. Constraint Module Files

| File | Lines | Purpose |
|------|-------|---------|
| `WeatherRoutingTool/constraints/constraints.py` | 1,144 | `ConstraintsList`, `LandCrossing`, `WaterDepth`, `StayOnMap`, `SeamarkCrossing` |

## 5. Support Files

| File | Lines | Purpose |
|------|-------|---------|
| `WeatherRoutingTool/routeparams.py` | 705 | `RouteParams` container with GeoJSON I/O |
| `WeatherRoutingTool/execute_routing.py` | 78 | Top-level orchestrator |
| `cli.py` | ~70 | CLI entry point |
| `WeatherRoutingTool/utils/graphics.py` | ~200 | Plot generation, basemap creation |
| `WeatherRoutingTool/utils/maps.py` | ~50 | `Map` class for bounding box management |

---

## 6. Data Flow: Input → Weather → Routing → Output

### 6.1 All Data Source Paths

| Data Mode | Class | Source | Dataset Schema |
|-----------|-------|--------|---------------|
| `from_file` | `WeatherCondFromFile` | Local `.nc` | `(time, latitude, longitude)` — 11 vars |
| `automatic` | `WeatherCondEnvAutomatic` | GFS + CMEMS downloads | Same schema, merged via `interp_like()` |
| `odc` | `WeatherCondODC` | Open Data Cube | Same `xr.Dataset` schema via `Datacube().load()` |
| `fake` | `FakeWeather` | Synthetic `np.full()` | Same schema, constant values |

**ODC Implications:** `WeatherCondODC` uses `datacube.Datacube().load()` which returns `xr.Dataset` objects that may already have internal tiling (ODC storage chunking). The statistics module must handle datasets from all data modes uniformly — the `xr.Dataset` interface abstracts away the storage backend. `FakeWeather` can serve as a test fixture for the statistics module since its constant fields have known statistical properties (mean=value, std=0).

### 6.2 Per-Routing-Step Weather Data Access

```
IsoBased.execute_routing()                                # isobased.py:601
  │
  ├──── Loop: while count < ncount ──────────────────────┐
  │                                                       │
  │  define_courses_per_step()                            │
  │    └─→ expands candidate routes (course_segments+1)   │
  │                                                       │
  │  estimate_fuel_consumption(boat)                      │
  │    └─→ boat.get_ship_parameters(                      │
  │          courses, lats, lons, time, speed)             │
  │        └─→ DirectPowerBoat.get_ship_parameters()      │
  │            ├─→ evaluate_weather(ship_params,           │
  │            │     lats, lons, time)                     │
  │            │   └─→ for i_coord in range(n_coords):    │ ← SEQUENTIAL LOOP
  │            │        approx_weather(ds['VHM0'],         │
  │            │          lats[i], lons[i], time[i])       │
  │            │        └─→ var.sel(latitude=lat,          │
  │            │              longitude=lon, time=time,    │
  │            │              method='nearest')            │
  │            │                                           │
  │            ├─→ evaluate_resistance(ship_params,        │
  │            │     courses)                              │
  │            │   └─→ get_wind_resistance(u, v, courses)  │
  │            │       └─→ Fujiwara approximation          │
  │            │                                           │
  │            └─→ P = get_power(deltaR)                   │
  │                fuel_rate = self.fuel_rate * P           │
  │                                                       │
  │  move_boat(bs, ship_params)                           │
  │    └─→ get_delta_variables_netCDF(ship_params, bs)    │
  │        └─→ delta_time = delta_fuel / fuel_rate        │
  │                                                       │
  │  check_constraints(constraint_list)                   │
  │    └─→ safe_crossing(lat_s, lon_s, lat_e, lon_e)     │
  │                                                       │
  │  pruning_per_step()                                   │
  │    └─→ select best routes per azimuthal bin           │
  └───────────────────────────────────────────────────────┘
```

### 6.3 Weather Variables and Their Usage

| Variable | Source | Used By | How |
|----------|--------|---------|-----|
| `u-component_of_wind_height_above_ground` | GFS | `DirectPowerBoat.get_wind_resistance()` | True wind speed/direction → apparent wind → wind resistance |
| `v-component_of_wind_height_above_ground` | GFS | `DirectPowerBoat.get_wind_resistance()` | Same |
| `VHM0` (wave height) | CMEMS wave | `Boat.evaluate_weather()`, `FakeWeather` | Wave resistance estimation, ocean pixel identification |
| `VMDR` (wave direction) | CMEMS wave | `Boat.evaluate_weather()` | Wave resistance (currently returns 0, see `direct_power_boat.py:387`) |
| `VTPK` (wave period) | CMEMS wave | `Boat.evaluate_weather()` | Wave resistance (not yet implemented) |
| `Temperature_surface` | GFS | `Boat.evaluate_weather()` | Air temperature |
| `Pressure_reduced_to_MSL_msl` | GFS | `Boat.evaluate_weather()` | Atmospheric pressure |
| `thetao` (ocean temperature) | CMEMS phys | `Boat.evaluate_weather()` | Water temperature |
| `so` (salinity) | CMEMS phys | `Boat.evaluate_weather()` | Water density |
| `utotal` (u current) | CMEMS curr | `Boat.evaluate_weather()` | Ocean current correction |
| `vtotal` (v current) | CMEMS curr | `Boat.evaluate_weather()` | Ocean current correction |

> **Note:** GFS wind variables include a `height_above_ground` dimension (10m, 80m). The vectorized lookup optimization must handle this extra dimension via `.sel(height_above_ground=10)` before batch selection.

---

## 7. Current Performance Bottleneck Locations

### 7.1 Eager Dataset Loading — `weather.py:619`
```python
self.ds = xr.open_dataset(filepath)
```
Loads the entire weather NetCDF into memory. For a 5-day, 0.083° North Atlantic dataset, estimated size is 350–500 MB based on: 40 time steps × 420 lat × 480 lon × 11 vars × 4 bytes (float32).

> **Note:** Load time and memory will be profiled during Community Bonding Week 1.

### 7.2 Sequential Weather Lookups — `ship.py:52-75`
```python
for i_coord in range(0, n_coords):
    wave_direction.append(
        self.approx_weather(weather_data['VMDR'], lats[i_coord], lons[i_coord], time[i_coord]))
    # ... 10 more .sel() calls per coordinate
```
11 `.sel()` calls × N coordinates, executed sequentially.

### 7.3 Redundant Dataset Opens — `ship.py:37`
```python
def evaluate_weather(self, ship_params, lats, lons, time):
    weather_data = xr.open_dataset(self.weather_path)
```
The `Boat` class opens the weather file on every call to `evaluate_weather()`. While `xarray`/`h5py` may internally cache file descriptors (reducing actual I/O), the Python-level `xr.open_dataset()` call still parses metadata and constructs the `xr.Dataset` object each time. The fix is to pass the pre-loaded `WeatherCond.ds` dataset reference to the ship model, eliminating the redundant open entirely.
