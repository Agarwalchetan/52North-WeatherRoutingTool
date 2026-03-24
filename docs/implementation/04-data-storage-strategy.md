# 04 — Data Storage Strategy

## 1. Format Comparison

| Property | NetCDF4 | Zarr | Kerchunk |
|----------|---------|------|----------|
| File structure | Single monolithic file | Directory of chunk files | JSON reference over NetCDF |
| Parallel write | Possible with MPI-HDF5, not via default xarray | Yes (per-chunk) | N/A (read-only) |
| Compression | Per-variable (zlib) | Per-chunk (Blosc, LZ4, Zstd) | Inherits source |
| Cloud access | Requires full download | Chunk-level via fsspec | Chunk-level via fsspec |
| Append | Complex (`mode='a'`) | Natural (new chunk files) | N/A |
| Existing archives | Universal | Growing | **No conversion needed** |
| xarray API | `xr.open_dataset()` | `xr.open_zarr()` | `xr.open_dataset(engine='kerchunk')` |

**Decision:** Zarr for statistical results. NetCDF for backward compatibility. Kerchunk for accessing large existing NetCDF archives without conversion.

## 2. Chunking Strategies

### Time-Series Optimized
```python
chunks = {'time': -1, 'latitude': 100, 'longitude': 100}
```
Best for: `mean(dim='time')`, `std(dim='time')` — full time axis contiguous per chunk.

### Spatial Optimized
```python
chunks = {'time': 10, 'latitude': -1, 'longitude': -1}
```
Best for: Spatial mean, spatial correlation — full spatial grid per chunk.

### Balanced (Default)
```python
chunks = {'time': 24, 'latitude': 200, 'longitude': 200}
```
Target: 50–200 MB per chunk. Suitable for mixed workloads.

## 3. ChunkOptimizer (Improved)

```python
class ChunkOptimizer:
    """Automatic chunk size selection considering variable count and dtype."""

    @staticmethod
    def optimize(ds: xr.Dataset, analysis_type: str = 'balanced',
                 target_bytes: int = 128 * 1024 * 1024) -> dict:
        n_time = ds.dims['time']
        n_lat = ds.dims['latitude']
        n_lon = ds.dims['longitude']

        # Account for variable count and largest dtype
        n_vars = len(ds.data_vars)
        max_itemsize = max(ds[v].dtype.itemsize for v in ds.data_vars)
        # Per-chunk target accounts for all variables loaded simultaneously
        per_var_target = target_bytes // max(n_vars, 1)

        if analysis_type == 'timeseries':
            # Full time axis, chunk spatial dims
            spatial_elems = per_var_target // (n_time * max_itemsize)
            side = int(np.sqrt(max(spatial_elems, 1)))
            return {'time': n_time,
                    'latitude': min(side, n_lat),
                    'longitude': min(side, n_lon)}

        elif analysis_type == 'spatial':
            # Full spatial extent, chunk time
            time_chunk = max(1, per_var_target // (n_lat * n_lon * max_itemsize))
            return {'time': min(time_chunk, n_time),
                    'latitude': n_lat,
                    'longitude': n_lon}

        else:  # 'balanced'
            total_elems = per_var_target // max_itemsize
            side = int(round(total_elems ** (1/3)))
            return {'time': min(max(side, 1), n_time),
                    'latitude': min(max(side, 1), n_lat),
                    'longitude': min(max(side, 1), n_lon)}

    @staticmethod
    def validate_chunks(chunks: dict, ds: xr.Dataset, max_memory_mb: int = 256) -> bool:
        """Validate that chunks won't exceed memory limits."""
        chunk_elems = 1
        for dim, size in chunks.items():
            chunk_elems *= min(size, ds.dims.get(dim, size))
        n_vars = len(ds.data_vars)
        max_itemsize = max(ds[v].dtype.itemsize for v in ds.data_vars)
        chunk_mb = chunk_elems * n_vars * max_itemsize / (1024 * 1024)
        if chunk_mb > max_memory_mb:
            logger.warning(
                f"Chunk size {chunk_mb:.0f} MB exceeds limit {max_memory_mb} MB. "
                f"Reducing automatically."
            )
            return False
        return True
```

## 4. Materialization Boundaries

Existing `.to_numpy()` / `.compute()` barriers in WRT where Dask arrays materialize:

| Location | Code | Behavior |
|----------|------|---------|
| `weather.py:349` | `tws.to_numpy()` | Materializes for `RegularGridInterpolator` |
| `ship.py:98` | `ship_var.fillna(0).to_numpy()` | Materializes for ship parameter assignment |
| `constraints.py:680` | `rounded_ds.to_numpy()` | Materializes for depth constraint check |

These form natural compute barriers: lazy evaluation persists through all xarray operations, materializing only when numpy is needed by downstream scipy/numpy code.

## 5. DataStorageManager (Auto-Detect Format)

```python
class DataStorageManager:
    """Unified storage interface with format auto-detection."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def save(self, dataset: xr.Dataset, path: str, format: str = None, mode: str = 'w'):
        if format is None:
            format = self._detect_format(path)
        if format == 'zarr':
            if hasattr(dataset, 'compute'):
                dataset = dataset.compute()  # Materialize Dask before write
            dataset.to_zarr(path, mode=mode, consolidated=True)
        elif format == 'netcdf':
            if hasattr(dataset, 'compute'):
                dataset = dataset.compute()
            dataset.to_netcdf(path)

    def load(self, path: str, chunks: dict = None, **kwargs) -> xr.Dataset:
        format = self._detect_format(path)
        if format == 'zarr':
            return xr.open_zarr(path, chunks=chunks, **kwargs)
        elif format == 'netcdf':
            return xr.open_dataset(path, chunks=chunks, **kwargs)
        elif format == 'kerchunk':
            return xr.open_dataset(path, engine='kerchunk', chunks=chunks, **kwargs)

    @staticmethod
    def _detect_format(path: str) -> str:
        path = str(path)
        if path.endswith('.zarr') or Path(path).is_dir():
            return 'zarr'
        elif path.endswith('.json'):
            return 'kerchunk'
        else:
            return 'netcdf'

    def cache_result(self, key: str, source_hash: str,
                     compute_fn, *args, **kwargs):
        if self.cache_dir is None:
            return compute_fn(*args, **kwargs)
        cache_path = self.cache_dir / f"{source_hash}_{key}.zarr"
        if cache_path.exists():
            return xr.open_zarr(cache_path)
        result = compute_fn(*args, **kwargs)
        if hasattr(result, 'compute'):
            result = result.compute()
        result.to_zarr(cache_path, consolidated=True)
        return result
```

## 6. Size → Strategy Decision Matrix

| Dataset Size | Strategy | Justification |
|-------------|----------|---------------|
| < 500 MB | In-memory (eager) | Dask overhead > benefit |
| 500 MB – 5 GB | Dask chunked, local | Reduces peak RSS; single-machine |
| > 5 GB | Dask + Zarr intermediates | Efficient partial reads |
| Existing large NetCDF | Kerchunk | No conversion needed |
| Statistical results | Zarr | Chunk-aligned, appendable |
| Route summaries | NetCDF | Small output, backward compatible |
