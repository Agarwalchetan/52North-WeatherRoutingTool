# 06 — API and Interface Design

## 1. Public API Surface

### 1.1 WeatherStatisticsEngine (Composition-Based)

The primary entry point. Uses composition (not mixins) to organize capabilities:

```python
from WeatherRoutingTool.statistics import WeatherStatisticsEngine

# From existing weather module
wt = WeatherCondFromFile(departure_time, hours, time_res)
wt.read_dataset(filepath, chunks={'time': 10, 'latitude': 100, 'longitude': 100})

engine = WeatherStatisticsEngine(wt.ds)

# Access capabilities via composition
mean = engine.aggregation.compute_mean(['VHM0'], dim='time')
corr = engine.correlation.compute_pairwise(['VHM0', 'VTPK'])
dist = engine.distributions.fit('VHM0', distribution='weibull')
```

### 1.2 Core Method Signatures

```python
class WeatherStatisticsEngine:
    """Central orchestrator using composition pattern."""

    # Composition attributes (initialized in __init__)
    aggregation: Aggregation
    temporal: Temporal
    distributions: Distributions
    correlation: Correlation

    # Engine-level methods
    def get_nan_report(self) -> dict
    def subset(self, lat_range=None, lon_range=None, time_range=None,
               variables=None) -> 'WeatherStatisticsEngine'
    def get_variable_summary(self) -> dict


class Aggregation:
    def compute_mean(self, variables=None, dim='time', skipna=True, min_count=1) -> xr.Dataset
    def compute_std(self, variables=None, dim='time', ddof=1, skipna=True) -> xr.Dataset
    def compute_percentile(self, variables=None, q=0.95, dim='time', skipna=True) -> xr.Dataset
    def compute_spatial_mean(self, variables=None, weights=None) -> xr.Dataset


class Temporal:
    def compute_rolling(self, variables=None, window=6, operation='mean',
                        min_periods=1) -> xr.Dataset
    def compute_resample(self, variables=None, freq='6H', operation='mean') -> xr.Dataset
    def compute_rate_of_change(self, variable: str) -> xr.DataArray


class Distributions:
    def fit(self, variable: str, distribution='weibull', dim='time',
            min_samples=30) -> xr.Dataset


class Correlation:
    def compute_pairwise(self, variables=None, dim='time') -> xr.DataArray
    def compute_spatial_correlation(self, var1: str, var2: str) -> xr.DataArray
```

## 2. Configuration Model

```python
from pydantic import BaseModel, Field
from typing import Optional

class StatisticsConfig(BaseModel):
    """Configuration for statistical analysis. Separated from main Config."""

    # Chunking
    ENABLE_DASK_CHUNKS: bool = False
    CHUNK_TIME: int = Field(default=24, ge=1)
    CHUNK_LATITUDE: int = Field(default=200, ge=1)
    CHUNK_LONGITUDE: int = Field(default=200, ge=1)
    AUTO_CHUNK: bool = Field(default=True, description="Auto-select optimal chunks")
    DASK_THRESHOLD_MB: int = Field(default=500, ge=100,
        description="Datasets below this size use eager numpy, above use Dask")

    # Computation
    PRECISION: str = Field(default='float64', pattern='^float(32|64)$')
    MAX_CHUNK_MEMORY_MB: int = Field(default=256, ge=64)
    N_WORKERS: int = Field(default=4, ge=1, le=32)

    # NaN handling
    SKIP_NAN: bool = Field(default=True,
        description="Skip NaN values in aggregations")
    MIN_VALID_COUNT: int = Field(default=1,
        description="Minimum valid values for non-NaN result")

    # Caching
    ENABLE_CACHE: bool = False
    CACHE_DIR: Optional[str] = None

    # Output
    OUTPUT_FORMAT: str = Field(default='zarr', pattern='^(zarr|netcdf)$')
    OUTPUT_DIR: Optional[str] = None
```

### 2.1 Integration with Existing Config

```python
# In config.py — add as optional field to existing Config class
class Config(BaseModel):
    # ... existing 50+ fields unchanged ...

    # NEW (optional, no impact if not specified)
    ENABLE_STATISTICS: bool = Field(default=False,
        description="Enable post-routing statistical analysis")
    STATISTICS_CONFIG: Optional[StatisticsConfig] = Field(
        default=None,
        description="Configuration for statistical module (only used if ENABLE_STATISTICS=True)")
    STATISTICS_OUTPUT_PATH: Optional[str] = None
```

## 3. CLI Interface

Integrated as a subcommand of the existing `cli.py` entry point:

```python
# In cli.py — add statistics subcommand
@click.group()
def main():
    pass

@main.command()
@click.option('--config', required=True, help='Path to WRT config.json')
def route(config):
    """Existing routing command (unchanged)."""
    ...

@main.command()
@click.option('--input', required=True, help='Weather NetCDF file')
@click.option('--output', required=True, help='Output path (.zarr or .nc)')
@click.option('--operations', default='mean,std',
              help='Comma-separated operations')
@click.option('--variables', default=None, help='Variables to analyze')
@click.option('--dim', default='time', help='Reduction dimension')
@click.option('--chunks', default=None, help='Dask chunks, e.g. time=10,latitude=100')
def stats(input, output, operations, variables, dim, chunks):
    """Statistical analysis of weather data."""
    from WeatherRoutingTool.statistics import WeatherStatisticsEngine

    chunk_dict = _parse_chunks(chunks) if chunks else None
    ds = xr.open_dataset(input, chunks=chunk_dict)
    engine = WeatherStatisticsEngine(ds)

    results = {}
    for op in operations.split(','):
        vars_list = variables.split(',') if variables else None
        results[op] = getattr(engine.aggregation, f'compute_{op}')(
            variables=vars_list, dim=dim)

    merged = xr.merge(results.values())
    engine.storage.save(merged, output)
```

```bash
# Usage examples
python -m cli stats --input weather.nc --output stats.zarr --operations mean,std,percentile
python -m cli stats --input weather.nc --output stats.zarr --variables VHM0,VTPK --dim time
python -m cli route --config config.json  # Existing command unchanged
```

## 4. Integration with execute_routing.py

```python
# Modified execute_routing.py (optional stats block)
def execute_routing(config, ship_config):
    wt = WeatherFactory.get_weather(config, ship_config)
    boat = ShipFactory.get_ship(config, ship_config)

    # ... existing routing logic ...
    route = routing_alg.execute_routing()

    # NEW: Optional statistical analysis (only if configured)
    if config.ENABLE_STATISTICS:
        from WeatherRoutingTool.statistics import WeatherStatisticsEngine
        from WeatherRoutingTool.statistics.route_analysis import StatisticalRouteAnalyzer

        engine = WeatherStatisticsEngine(wt.ds, config.STATISTICS_CONFIG)
        analyzer = StatisticalRouteAnalyzer(engine)

        route_stats = analyzer.analyze_route_weather(
            lats=route.lats_per_step,
            lons=route.lons_per_step,
            times=route.starttime_per_step,
        )
        if config.STATISTICS_OUTPUT_PATH:
            engine.storage.save(route_stats, config.STATISTICS_OUTPUT_PATH)
```

## 5. Integration Points

| Integration Point | Current Code | Modification | Risk |
|-------------------|-------------|--------------|------|
| `weather.py:619` | `xr.open_dataset(filepath)` | Add optional `chunks` parameter | Low — opt-in, default unchanged |
| `execute_routing.py:43` | Post-routing output | Add optional stats block | Low — guarded by `ENABLE_STATISTICS` |
| `config.py` | No statistics config | Add `StatisticsConfig` as optional field | Low — optional, no validation impact |
| `ship.py:37` | Re-opens dataset from file | Accept pre-loaded dataset reference | Medium — changes function signature |
| `cli.py` | Single `route` command | Add `stats` subcommand | Low — additive |
| `weather_factory.py:15` | Returns `WeatherCond` | Unchanged | None |

## 6. Backward Compatibility Guarantee

- No existing function signatures change (new parameters have defaults).
- No existing imports break.
- No existing test assertions fail.
- Dask chunking is disabled by default (`ENABLE_DASK_CHUNKS=False`).
- Statistics module is only imported when explicitly requested.
- CLI subcommand `stats` is additive — `route` command unchanged.
