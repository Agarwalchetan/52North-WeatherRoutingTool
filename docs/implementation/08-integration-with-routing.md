# 08 — Integration with Routing

## 1. Integration Architecture

The statistics module integrates at **two points** in the routing pipeline:

```
Config → WeatherFactory → xr.Dataset
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              RoutingAlg    Ship    Constraints
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                       RouteResult
                              │
          ┌───────────────────┤
          ▼                   ▼
  [PRE-ROUTING]         [POST-ROUTING]        ← Both optional
  StatisticsEngine      StatisticalRoute
  .aggregation.*        Analyzer
  .temporal.*           .analyze_route()
  .distributions.*      .compute_risk_score()
  .correlation.*
```

### 1.1 Pre-Routing Analysis (Dataset-Level)

Before routing, compute summary statistics to characterize weather conditions:

```python
if config.ENABLE_STATISTICS:
    engine = WeatherStatisticsEngine(wt.ds, config.STATISTICS_CONFIG)

    # Dataset summary — useful for decision-makers
    summary = {
        'nan_report': engine.get_nan_report(),
        'temporal_mean': engine.aggregation.compute_mean(dim='time'),
        'wind_distribution': engine.distributions.fit('VHM0', distribution='weibull'),
    }
```

### 1.2 Post-Routing Analysis (Route-Level)

After routing, extract weather conditions along the computed route:

```python
if config.ENABLE_STATISTICS:
    analyzer = StatisticalRouteAnalyzer(engine)
    route_stats = analyzer.analyze_route_weather(
        lats=route.lats_per_step,
        lons=route.lons_per_step,
        times=route.starttime_per_step,
    )
```

## 2. StatisticalRouteAnalyzer

```python
class StatisticalRouteAnalyzer:
    """Extracts and analyzes weather conditions along computed routes."""

    def __init__(self, engine: WeatherStatisticsEngine):
        self.engine = engine

    def analyze_route_weather(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        variables: list = None,
    ) -> xr.Dataset:
        """
        Extract weather values along route waypoints.

        Returns xr.Dataset with dim='waypoint' containing weather at each route point.
        """
        if variables is None:
            variables = list(self.engine.ds.data_vars)

        lat_da = xr.DataArray(lats, dims='waypoint')
        lon_da = xr.DataArray(lons, dims='waypoint')
        time_da = xr.DataArray(times, dims='waypoint')

        route_data = {}
        for var in variables:
            v = self.engine.ds[var]
            # Handle extra dimensions (height_above_ground)
            if 'height_above_ground' in v.dims:
                v = v.sel(height_above_ground=10)
            route_data[var] = v.sel(
                latitude=lat_da, longitude=lon_da, time=time_da,
                method='nearest',
            )

        ds = xr.Dataset(route_data)
        ds.attrs['n_waypoints'] = len(lats)
        return ds

    def compute_route_summary(
        self,
        route_weather: xr.Dataset,
    ) -> dict:
        """Compute summary statistics for a route."""
        summary = {}
        for var in route_weather.data_vars:
            values = route_weather[var].values
            valid = values[~np.isnan(values)]
            summary[var] = {
                'mean': float(np.nanmean(valid)) if len(valid) > 0 else None,
                'max': float(np.nanmax(valid)) if len(valid) > 0 else None,
                'min': float(np.nanmin(valid)) if len(valid) > 0 else None,
                'std': float(np.nanstd(valid)) if len(valid) > 0 else None,
            }
        return summary

    def compute_risk_index(
        self,
        route_weather: xr.Dataset,
        thresholds: dict = None,
    ) -> float:
        """
        Simple risk index: fraction of waypoints exceeding thresholds.

        Default thresholds based on maritime weather classifications:
        - VHM0 > 4.0m (rough sea, BF 6+)
        - wind speed > 17 m/s (near gale, BF 7+)
        """
        if thresholds is None:
            thresholds = {
                'VHM0': 4.0,  # meters — rough sea
            }

        n_waypoints = route_weather.dims['waypoint']
        violations = 0
        for var, threshold in thresholds.items():
            if var in route_weather:
                violations += int((route_weather[var] > threshold).sum())

        return violations / max(n_waypoints * len(thresholds), 1)
```

## 3. GeoJSON Route Output Extension

WRT outputs routes as GeoJSON via `RouteParams.to_geojson()`. The statistics module extends this with weather properties per waypoint:

```python
def extend_geojson_with_stats(geojson: dict, route_weather: xr.Dataset) -> dict:
    """Add weather statistics to GeoJSON route features."""
    for i, feature in enumerate(geojson.get('features', [])):
        if i < route_weather.dims.get('waypoint', 0):
            props = feature.setdefault('properties', {})
            for var in route_weather.data_vars:
                val = route_weather[var].isel(waypoint=i)
                props[f'weather_{var}'] = float(val) if not np.isnan(float(val)) else None
    return geojson
```

## 4. Genetic Algorithm Integration (Stretch Goal)

> **This is explicitly a stretch goal.** Adding weather risk as an optimization objective fundamentally changes the genetic algorithm's fitness landscape and requires careful design.

### 4.1 Current Genetic Algorithm Objectives

In `algorithms/genetic/problem.py`, `RoutingProblem` uses NSGA-II with:
1. **Objective 1:** Total fuel consumption
2. **Objective 2:** Arrival time accuracy

### 4.2 Proposed Addition (If Implemented)

Adding a third objective (weather risk) creates a 3-objective optimization requiring NSGA-III:

```python
class RoutingProblem(Problem):
    def __init__(self, ..., use_weather_risk=False):
        n_obj = 3 if use_weather_risk else 2
        super().__init__(n_var=..., n_obj=n_obj, ...)

    def _evaluate(self, x, out, *args, **kwargs):
        fuel = self._compute_fuel(x)
        time_error = self._compute_time_error(x)
        out["F"] = [fuel, time_error]

        if self.use_weather_risk:
            risk = self._compute_weather_risk(x)
            out["F"].append(risk)
```

### 4.3 Why This Is a Stretch Goal

| Concern | Impact |
|---------|--------|
| NSGA-II → NSGA-III transition | Changes pymoo Algorithm class, may break existing behavior |
| 3-objective Pareto front visualization | Cannot use simple 2D Pareto plots |
| Risk function definition | What constitutes "risk"? BF scale? Probability of damage? Domain expertise required. |
| Performance impact | Additional weather lookups per candidate evaluation |

**If implemented:** This will have its own ADR, its own test suite, and mentor approval before merge.
