# 09 — Testing Strategy

## 1. Test Organization

```
tests/
├── test_statistics/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures (synthetic datasets)
│   ├── test_engine.py                 # Engine init, validation, NaN report
│   ├── test_aggregation.py            # mean, std, percentile, spatial mean
│   ├── test_temporal.py               # rolling, resample, rate-of-change
│   ├── test_distributions.py          # Distribution fitting, convergence failures
│   ├── test_correlation.py            # Pearson pairwise, spatial correlation
│   ├── test_storage.py               # NetCDF/Zarr save/load, caching, format detection
│   ├── test_visualization.py          # Smoke tests (plot generation without error)
│   ├── test_route_analysis.py         # Route weather extraction, risk scoring
│   ├── test_chunk_optimizer.py        # Chunk size selection, validation
│   ├── test_nan_handling.py           # NaN propagation, all-NaN slices, mixed NaN
│   ├── test_edge_cases.py            # Empty datasets, single-timestep, single-gridcell
│   ├── test_dask_compatibility.py     # Dask-vs-eager result equality
│   └── test_backward_compatibility.py # Existing WRT tests unchanged
```

## 2. Shared Fixtures (`conftest.py`)

```python
import numpy as np
import pytest
import xarray as xr

@pytest.fixture
def synthetic_weather_dataset():
    """Create synthetic weather dataset matching WRT variable conventions."""
    np.random.seed(42)
    n_time, n_lat, n_lon = 40, 50, 60

    time = np.arange(n_time).astype('datetime64[h]') + np.datetime64('2024-01-01')
    lat = np.linspace(35, 65, n_lat)
    lon = np.linspace(-60, -20, n_lon)

    ds = xr.Dataset({
        'VHM0': (['time', 'latitude', 'longitude'],
                 np.random.weibull(2, (n_time, n_lat, n_lon)).astype(np.float32) * 2),
        'VTPK': (['time', 'latitude', 'longitude'],
                 np.random.uniform(3, 12, (n_time, n_lat, n_lon)).astype(np.float32)),
        'VMDR': (['time', 'latitude', 'longitude'],
                 np.random.uniform(0, 360, (n_time, n_lat, n_lon)).astype(np.float32)),
    }, coords={'time': time, 'latitude': lat, 'longitude': lon})

    ds['VHM0'].attrs = {'units': 'm', 'native_resolution': 'native'}
    ds['VTPK'].attrs = {'units': 's', 'native_resolution': 'native'}
    ds['VMDR'].attrs = {'units': 'degree', 'native_resolution': 'native'}
    return ds

@pytest.fixture
def nan_heavy_dataset(synthetic_weather_dataset):
    """Dataset with ~40% NaN values (simulating land pixels)."""
    ds = synthetic_weather_dataset.copy(deep=True)
    # Simulate land mask: NaN for lat > 55 and lon < -40
    land_mask = (ds.latitude > 55) & (ds.longitude < -40)
    for var in ds.data_vars:
        ds[var] = ds[var].where(~land_mask)
    return ds

@pytest.fixture
def dask_weather_dataset(synthetic_weather_dataset):
    """Dask-backed version of synthetic dataset."""
    return synthetic_weather_dataset.chunk({'time': 10, 'latitude': 25, 'longitude': 30})

@pytest.fixture
def single_point_dataset():
    """Edge case: single spatial point."""
    return xr.Dataset({
        'VHM0': (['time', 'latitude', 'longitude'],
                 np.random.rand(20, 1, 1).astype(np.float32)),
    }, coords={
        'time': np.arange(20).astype('datetime64[h]') + np.datetime64('2024-01-01'),
        'latitude': [50.0], 'longitude': [-30.0],
    })

@pytest.fixture
def single_timestep_dataset():
    """Edge case: single time step."""
    return xr.Dataset({
        'VHM0': (['time', 'latitude', 'longitude'],
                 np.random.rand(1, 50, 60).astype(np.float32)),
    }, coords={
        'time': [np.datetime64('2024-01-01')],
        'latitude': np.linspace(35, 65, 50),
        'longitude': np.linspace(-60, -20, 60),
    })
```

## 3. Unit Tests

### 3.1 Aggregation with NaN

```python
def test_compute_mean_with_nan(nan_heavy_dataset):
    engine = WeatherStatisticsEngine(nan_heavy_dataset)
    result = engine.aggregation.compute_mean(variables=['VHM0'], dim='time')

    # NaN regions remain NaN
    land_mask = (nan_heavy_dataset.latitude > 55) & (nan_heavy_dataset.longitude < -40)
    assert result['VHM0'].where(land_mask).isnull().all()

    # Non-NaN regions have valid mean
    ocean = result['VHM0'].where(~land_mask, drop=True)
    assert not ocean.isnull().any()

def test_compute_mean_min_count(nan_heavy_dataset):
    engine = WeatherStatisticsEngine(nan_heavy_dataset)
    # Require many valid values — some cells should become NaN
    result = engine.aggregation.compute_mean(['VHM0'], dim='time', min_count=35)
    # Cells with < 35 valid timestamps should be NaN
    assert result['VHM0'].isnull().sum() > 0
```

### 3.2 Dask vs. Eager Equality

```python
def test_all_operations_dask_equals_eager(synthetic_weather_dataset, dask_weather_dataset):
    engine_eager = WeatherStatisticsEngine(synthetic_weather_dataset)
    engine_dask = WeatherStatisticsEngine(dask_weather_dataset)

    # Aggregation
    xr.testing.assert_allclose(
        engine_eager.aggregation.compute_mean(['VHM0']),
        engine_dask.aggregation.compute_mean(['VHM0']).compute()
    )
    xr.testing.assert_allclose(
        engine_eager.aggregation.compute_std(['VHM0']),
        engine_dask.aggregation.compute_std(['VHM0']).compute()
    )

    # Correlation
    corr_eager = engine_eager.correlation.compute_pairwise(['VHM0', 'VTPK'])
    corr_dask = engine_dask.correlation.compute_pairwise(['VHM0', 'VTPK'])
    xr.testing.assert_allclose(corr_eager, corr_dask, atol=1e-6)
```

### 3.3 Distribution Fitting Edge Cases

```python
def test_distribution_fit_all_nan():
    """Distribution fit on all-NaN data returns NaN parameters."""
    ds = xr.Dataset({
        'wind': (['time', 'latitude', 'longitude'],
                 np.full((20, 1, 1), np.nan, dtype=np.float32)),
    }, coords={'time': range(20), 'latitude': [0], 'longitude': [0]})
    ds = ds.expand_dims({'latitude': 1, 'longitude': 1})

    engine = WeatherStatisticsEngine(ds)
    result = engine.distributions.fit('wind', distribution='weibull')
    assert result['wind_weibull_shape'].isnull().all()

def test_distribution_fit_insufficient_samples():
    """Fewer than min_samples valid points returns NaN."""
    ds = xr.Dataset({
        'wind': (['time', 'latitude', 'longitude'],
                 np.random.rand(10, 1, 1).astype(np.float32)),
    }, coords={'time': range(10), 'latitude': [0], 'longitude': [0]})

    engine = WeatherStatisticsEngine(ds)
    result = engine.distributions.fit('wind', min_samples=30)
    assert result['wind_weibull_shape'].isnull().all()
```

### 3.4 Edge Case Tests

```python
def test_single_timestep_mean(single_timestep_dataset):
    engine = WeatherStatisticsEngine(single_timestep_dataset)
    result = engine.aggregation.compute_mean(['VHM0'], dim='time')
    # Mean of single value = the value itself
    xr.testing.assert_equal(
        result['VHM0'], single_timestep_dataset['VHM0'].squeeze('time'))

def test_single_point_spatial_mean(single_point_dataset):
    engine = WeatherStatisticsEngine(single_point_dataset)
    result = engine.aggregation.compute_spatial_mean(['VHM0'])
    # Spatial mean of single point = the value itself
    xr.testing.assert_allclose(
        result['VHM0'], single_point_dataset['VHM0'].squeeze(['latitude', 'longitude']))

def test_empty_variable_list():
    ds = xr.Dataset({'VHM0': (['time', 'latitude', 'longitude'], np.zeros((5, 5, 5)))},
                     coords={'time': range(5), 'latitude': range(5), 'longitude': range(5)})
    engine = WeatherStatisticsEngine(ds)
    result = engine.aggregation.compute_mean(variables=None, dim='time')
    assert 'VHM0' in result

def test_longitude_normalization():
    """Test 0-360 longitude is normalized to -180..180."""
    ds = xr.Dataset({'VHM0': (['time', 'latitude', 'longitude'], np.zeros((5, 5, 5)))},
                     coords={'time': range(5), 'latitude': range(5),
                             'longitude': np.linspace(0, 360, 5)})
    engine = WeatherStatisticsEngine(ds)
    assert float(engine.ds.longitude.max()) <= 180
```

## 4. Integration Tests

```python
def test_end_to_end_pipeline(tmp_path, synthetic_weather_dataset):
    """Full pipeline: load → analyze → save → reload → verify."""
    nc_path = tmp_path / 'weather.nc'
    synthetic_weather_dataset.to_netcdf(nc_path)

    ds = xr.open_dataset(nc_path, chunks={'time': 10})
    engine = WeatherStatisticsEngine(ds)

    mean = engine.aggregation.compute_mean(dim='time')
    corr = engine.correlation.compute_pairwise()

    zarr_path = str(tmp_path / 'stats.zarr')
    engine.storage.save(mean.compute(), zarr_path)

    reloaded = xr.open_zarr(zarr_path)
    xr.testing.assert_equal(mean.compute(), reloaded)
```

## 5. Backward Compatibility

```python
def test_existing_tests_unaffected():
    """Import statistics module without breaking existing code."""
    import WeatherRoutingTool.statistics  # noqa: F401
    # No circular imports, no side effects

def test_read_dataset_without_chunks_unchanged(tmp_path, synthetic_weather_dataset):
    nc_path = tmp_path / 'test.nc'
    synthetic_weather_dataset.to_netcdf(nc_path)

    # Without chunks — same as current behavior
    ds = xr.open_dataset(nc_path)
    assert not any(hasattr(ds[v].data, 'dask') for v in ds.data_vars)
```

## 6. Coverage Targets

| Module | Target | Est. Tests |
|--------|--------|-----------|
| `engine.py` | 95% | 10 |
| `aggregation.py` | 95% | 12 |
| `temporal.py` | 90% | 8 |
| `distributions.py` | 85% | 10 |
| `correlation.py` | 90% | 8 |
| `storage.py` | 90% | 8 |
| `visualization.py` | 70% (smoke) | 5 |
| `route_analysis.py` | 90% | 8 |
| `chunk_optimizer.py` | 90% | 6 |
| `nan_handling` (cross-cutting) | 95% | 10 |
| `edge_cases` | 90% | 8 |
| **Total** | **≥ 90%** | **~93** |
