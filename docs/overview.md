# Weather Routing Tool – Extension of Weather Module for Statistical Analysis

## GSoC 2026 Proposal | Organization: 52°North

---

## 1. System Purpose

The **Weather Routing Tool (WRT)** is a Python-based geospatial system that computes **fuel-optimal ship routes** by integrating multi-source meteorological and oceanographic data with physics-based vessel performance models. The system solves a constrained optimization problem: given a departure point, destination, vessel characteristics, and forecast weather conditions, determine the route that minimizes total fuel consumption while respecting navigational constraints (land avoidance, minimum water depth, staying within the weather data extent, and intermediate waypoints).

WRT supports multiple routing algorithms with varying levels of weather coupling:

| Algorithm | Strategy | Weather Coupling Level |
|-----------|----------|----------------------|
| **IsoFuel** | Iterative wavefront expansion with equal-fuel steps | **Full**: fuel rate depends on wind/wave resistance via `DirectPowerBoat` at each candidate position |
| **Genetic** (pymoo) | Multi-objective evolutionary optimization (NSGA-II) | **Full**: fitness evaluates fuel consumption + arrival time accuracy per route segment |
| **ConstantFuelBoat** | Ships with fixed fuel consumption rate | **None**: fuel rate is constant regardless of weather — `get_ship_parameters()` ignores environmental data |
| **Dijkstra** | Graph-based shortest path on a spatial grid | **Indirect**: edge weights may incorporate weather data |
| **GCR Slider** | Great Circle Route perturbation with land avoidance | **None**: weather data not consumed |

> **Note:** Weather coupling depends on the **ship model**, not just the algorithm. `ConstantFuelBoat` (ship.py:142-163) ignores weather entirely, using a fixed fuel rate. Only `DirectPowerBoat` integrates weather into fuel estimation via the Fujiwara wind resistance model, wave-added resistance, and current corrections.

The system operates within a **QGIS plugin context** and as a standalone CLI tool via `cli.py`.

---

## 2. How Weather Data Drives Routing Decisions

### 2.1 Fuel Consumption Estimation (DirectPowerBoat Only)

In `DirectPowerBoat.get_ship_parameters()` (direct_power_boat.py:84-166):

1. **Reads weather conditions** at each candidate waypoint via `Boat.evaluate_weather()` (ship.py:37-75), which performs `xr.DataArray.sel(latitude=..., longitude=..., time=..., method='nearest')` lookups for 11 environmental variables.

2. **Calculates wind resistance** using the Fujiwara approximation (direct_power_boat.py:350-398), converting u/v wind components to relative wind speed/direction against the ship's course.

3. **Derives brake power** via the added-resistance power equation, accounting for propulsion efficiency (`eta`), overload factor (`OF`), and service power (`P_sp`).

4. **Fuel rate** is computed as `fuel_rate = SFOC × P_brake` (specific fuel oil consumption × brake power).

### 2.2 Data Source Merging and Its Statistical Implications

A critical consideration for statistical analysis: the unified `xr.Dataset` is constructed by **interpolating heterogeneous sources onto a common grid** (weather.py:298-310):

```python
# CMEMS physics (0.083°, 1h) interpolated to wave timestamps (3h)
phys_interpolated = ds_CMEMS_phys.interp_like(ds_CMEMS_wave)
# GFS (0.25°) interpolated to CMEMS resolution (0.083°)
GFS_interpolated = ds_GFS.interp_like(full_CMEMS_data)
```

**Statistical implication:** The merged dataset contains variables at different native resolutions. GFS wind fields at 0.083° are interpolated from 0.25° — they have artificially smooth spatial structure. Any spatial statistics (correlation length, spatial variance) computed on interpolated GFS variables will **underestimate true variability**. The statistics module must track and report the native resolution of each variable.

### 2.3 Data Source Paths

WRT supports 4 data ingestion paths, each producing `xr.Dataset`:

| Data Mode | Class | Source | Special Considerations |
|-----------|-------|--------|----------------------|
| `from_file` | `WeatherCondFromFile` | Local `.nc` file | Eager load via `xr.open_dataset()` (line 619). Main target for Dask integration. |
| `automatic` | `WeatherCondEnvAutomatic` | GFS + CMEMS downloads via `maridatadownloader` | Downloads and merges 4 sub-datasets. Post-merge interpolation affects statistical properties. |
| `odc` | `WeatherCondODC` | Open Data Cube | Uses `Datacube().load()` which returns pre-tiled xarray data. Already has internal chunking semantics via ODC's storage model. |
| `fake` | `FakeWeather` | Synthetic `np.full()` arrays | Useful as statistics module test fixture — constant fields with known statistical properties. |

### 2.4 Current Data Flow

```
Config.json → Config (pydantic) → execute_routing()
                                        │
                     ┌──────────────────┘
                     ▼
            WeatherFactory.get_weather(data_mode)
                     │
      ┌──────────┬───┴────┬──────────┐
      ▼          ▼        ▼          ▼
  FromFile   Automatic   ODC     FakeWeather
      │          │        │          │
      └──────────┴────┬───┴──────────┘
                      ▼
               xr.Dataset (self.ds)
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                 ▼
RoutingAlg      ShipModel         Constraints
(IsoFuel/       (DirectPower       (WaterDepth,
 Genetic)        or Constant)       StayOnMap)
                      │
                      ▼           ← NEW
              WeatherStatisticsEngine
```

---

## 3. Missing Statistical Capabilities

### 3.1 No Aggregation or Summary Statistics
No facility to compute spatial or temporal means, medians, standard deviations, or percentiles over the weather dataset.

### 3.2 No Time-Series Analytics
Weather data is consumed point-by-point via `.sel(method='nearest')` — nearest timestep snapping with **no temporal interpolation**, no smoothing, no trend detection.

### 3.3 No Distributional Analysis
No mechanism to characterize statistical distributions (Weibull for wind speed, Rayleigh for wave height). No probabilistic routing.

### 3.4 No Correlation Analysis
Cross-variable correlations (wind speed vs. wave height, pressure gradients vs. storm intensity) are never computed.

### 3.5 No Multi-Ensemble Support
Single forecast realization only. No infrastructure for ensemble-based uncertainty quantification.

### 3.6 No NaN-Aware Processing
Ocean weather datasets contain pervasive NaN values — land pixels, below-depth cells, missing observations. The current system handles NaN only via `fillna(0)` in `ship.py:98`, which is statistically incorrect for aggregation operations.

### 3.7 Performance and Memory Limitations
- `WeatherCondFromFile.read_dataset()` at line 619: eager `xr.open_dataset(filepath)`, full materialization into RAM.
- `Boat.evaluate_weather()` at ship.py:52-75: Python `for` loop performing 11 sequential `.sel()` calls per coordinate.
- `Boat.evaluate_weather()` at ship.py:37: re-opens the weather NetCDF file from disk on **every call** (60-100 times per routing run).

---

## 4. Problem Statement

The WRT weather module provides only raw data access — point lookups and spatial interpolation — with no capacity for:
1. Statistical summarization (means, percentiles, distributions along routes or across regions)
2. NaN-aware aggregation over ocean datasets with land masking
3. Performance-optimized processing via chunked/lazy evaluation
4. Awareness of native resolution differences between merged data sources

This proposal introduces a **WeatherStatisticsEngine** module providing xarray-native, NaN-aware, Dask-compatible statistical computation with clean integration into the existing routing pipeline.
