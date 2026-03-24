# 02 — Weather Data Pipeline

## 1. Data Ingestion

### 1.1 Data Sources

WRT ingests weather data from three external sources plus two alternative paths, all producing a unified `xr.Dataset`:

```
┌──────────────────────────────┐
│   GFS (NOAA)                 │ 0.25° resolution, 3-hourly
│   Variables:                 │ Temperature_surface,
│   u/v-component_of_wind,     │ Pressure_reduced_to_MSL_msl
│   height_above_ground: 10m   │ ← NOTE: extra dimension
└──────────┬───────────────────┘
           │ DownloaderFactory.get_downloader('xarray', 'gfs')
           ▼
┌──────────────────────────────┐
│   CMEMS Waves                │ 0.083° resolution, 3-hourly
│   Variables:                 │ VHM0, VMDR, VTPK
└──────────┬───────────────────┘
           │ DownloaderFactory.get_downloader('cmtapi', 'cmems')
           ▼
┌──────────────────────────────┐
│   CMEMS Physics              │ 0.083° resolution, 1-hourly
│   Variables:                 │ thetao, so
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   CMEMS Currents             │ 0.083° resolution, 1-hourly
│   Variables:                 │ utotal, vtotal
└──────────┴───────────────────┘
```

### 1.2 File-Based Ingestion

In `WeatherCondFromFile.read_dataset()` (weather.py:615–619):

```python
def read_dataset(self, filepath=None):
    if filepath is None:
        raise RuntimeError("filepath must not be None for data_mode = 'from_file'")
    self.ds = xr.open_dataset(filepath)  # ← EAGER LOAD, NO CHUNKING
```

**Proposed change (opt-in Dask chunking):**

```python
def read_dataset(self, filepath=None, chunks=None):
    if filepath is None:
        raise RuntimeError("filepath must not be None for data_mode = 'from_file'")
    if chunks is not None:
        self.ds = xr.open_dataset(filepath, chunks=chunks)
    else:
        self.ds = xr.open_dataset(filepath)  # Unchanged default behavior
```

---

## 2. Cross-Product Interpolation and Its Statistical Implications

### 2.1 Merging Code Path

Multiple datasets are merged by interpolating to a common grid (weather.py:298-310):

```python
phys_interpolated = ds_CMEMS_phys.interp_like(ds_CMEMS_wave)  # 1h → 3h temporal
curr_interpolated = ds_CMEMS_curr.interp_like(ds_CMEMS_wave)
full_CMEMS_data = xr.merge([curr_interpolated, phys_interpolated, ds_CMEMS_wave])

GFS_interpolated = ds_GFS.interp_like(full_CMEMS_data)  # 0.25° → 0.083° spatial
self.ds = xr.merge([full_CMEMS_data, GFS_interpolated])
```

### 2.2 Statistical Validity Consequences

| Variable Source | Native Resolution | After Merge | Statistical Implication |
|----------------|-------------------|-------------|------------------------|
| CMEMS wave (VHM0, VMDR, VTPK) | 0.083°, 3h | **Unchanged** (reference grid) | Full native variability preserved |
| CMEMS physics (thetao, so) | 0.083°, 1h | 0.083°, **3h** (temporal interpolation) | Temporal variability smoothed — `std(dim='time')` will underestimate |
| CMEMS currents (utotal, vtotal) | 0.083°, 1h | 0.083°, **3h** | Same temporal smoothing |
| GFS wind/pressure/temp | **0.25°**, 3h | **0.083°**, 3h (spatial interpolation) | **Spatial variability artificially smoothed** — `std(dim='latitude')` will underestimate for GFS variables |

> **Impact on statistics module:** The engine must track and report native resolution per variable via `ds[var].attrs['native_resolution']`. Users should be warned when computing spatial statistics on spatially interpolated variables.

```python
def _check_interpolation_provenance(self, variable: str) -> Optional[str]:
    native_res = self.ds[variable].attrs.get('native_resolution')
    if native_res and native_res != 'native':
        logger.warning(
            f"Variable '{variable}' was interpolated from native "
            f"resolution {native_res}. Spatial statistics may "
            f"underestimate true variability."
        )
    return native_res
```

---

## 3. GFS height_above_ground Dimension

GFS wind data includes a `height_above_ground` dimension (10m, 80m, etc.). The merged dataset selects 10m height during download:

```python
sel_dict_GFS = {
    'height_above_ground2': slice(10, 20),  # Selects 10m level
    ...
}
```

**Impact on vectorized lookup:** After the merge, the `height_above_ground` dimension may still exist if not squeezed. The vectorized `.sel()` must handle this:

```python
# Pre-select height dimension before batch lookup
wind_u = weather_data['u-component_of_wind_height_above_ground']
if 'height_above_ground' in wind_u.dims:
    wind_u = wind_u.sel(height_above_ground=10)
ship_params.wind_u = wind_u.sel(
    latitude=lat_da, longitude=lon_da, time=time_da,
    method='nearest').values
```

---

## 4. Nearest-Neighbor Lookup

In `Boat.approx_weather()` (ship.py:92-100):

```python
def approx_weather(self, var, lats, lons, time, height=None, depth=None):
    ship_var = var.sel(latitude=lats, longitude=lons, time=time,
                       method='nearest', drop=False)
```

**Behavior with Dask arrays:** `.sel(method='nearest')` on a Dask-backed DataArray returns a Dask scalar. It triggers reading only the chunk containing the selected coordinate, not the entire dataset. This is Dask-compatible without modification.

**Edge case: out-of-bounds coordinates.** If a route point falls outside the dataset's spatial extent, `.sel(method='nearest')` returns the boundary value (no error). This is correct for routing (the `StayOnMap` constraint prevents this) but could produce misleading statistics if unchecked. The statistics module validates coordinate bounds in `_validate_dataset()`.

---

## 5. Proposed Enhanced Pipeline

```
                        ┌─────────────────────┐
                        │   Raw Data Sources   │
                        │ (GFS, CMEMS, Files,  │
                        │  ODC, FakeWeather)   │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  WeatherCondFromFile │
                        │  .read_dataset()     │
                        │  chunks=config.chunks│  ← opt-in Dask
                        └──────────┬──────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
          ┌────────────┐  ┌────────────────┐  ┌──────────┐
          │  Routing    │  │  Statistics     │  │  Ship    │
          │  Algorithms │  │  Engine         │  │  Model   │
          │             │  │                 │  │          │
          │ (unchanged) │  │ .aggregation    │  │(proposed:│
          │             │  │ .temporal       │  │ vectorize│
          │             │  │ .correlation    │  │ lookups) │
          └─────────────┘  └────────┬────────┘  └──────────┘
                                    │
                           ┌────────▼────────┐
                           │  DataStorage    │
                           │  Manager        │
                           │ .save(zarr/nc)  │
                           │ .cache()        │
                           └─────────────────┘
```

### 5.1 Vectorized Weather Lookup

**Current** (`ship.py:52-75`, sequential loop):
```python
for i_coord in range(0, n_coords):
    wave_height.append(
        self.approx_weather(weather_data['VHM0'], lats[i_coord], lons[i_coord], time[i_coord]))
```

**Proposed** (vectorized batch lookup):
```python
def evaluate_weather_vectorized(self, ship_params, lats, lons, time, weather_data):
    lat_da = xr.DataArray(lats, dims='points')
    lon_da = xr.DataArray(lons, dims='points')
    time_da = xr.DataArray(time, dims='points')

    # Handle height_above_ground dimension if present
    for var_name in ['VHM0', 'VMDR', 'VTPK', ...]:
        var = weather_data[var_name]
        if 'height_above_ground' in var.dims:
            var = var.sel(height_above_ground=10)
        ship_params[var_name] = var.sel(
            latitude=lat_da, longitude=lon_da, time=time_da,
            method='nearest').values
    return ship_params
```

**Expected improvement:** Eliminates Python loop overhead (6,600 sequential `.sel()` → 11 vectorized calls). True per-element cost remains — actual speedup will be measured during benchmarking.

**Dask behavior with vectorized .sel():** When `weather_data` is Dask-backed, vectorized `.sel()` with DataArray indexers triggers reading only the chunks that contain the selected coordinates. If route points are clustered (typical for maritime routing), this reads far fewer chunks than the full dataset. If points span many chunks, performance may degrade — this will be tested in Week 2.
