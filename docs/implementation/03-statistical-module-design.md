# 03 — Statistical Module Design

## 1. Module Structure

```
WeatherRoutingTool/statistics/
├── __init__.py              # Public API exports
├── engine.py                # WeatherStatisticsEngine (core orchestrator)
├── aggregation.py           # Spatial/temporal aggregation operations
├── temporal.py              # Time-series analytics (rolling, resampling, trends)
├── distributions.py         # Distribution fitting (Weibull, Rayleigh, Normal)
├── correlation.py           # Cross-variable correlation analysis (Dask-compatible)
├── storage.py               # DataStorageManager (NetCDF, Zarr, cache)
├── visualization.py         # Statistical plot generation
├── route_analysis.py        # Route-specific weather analysis
├── config.py                # StatisticsConfig (pydantic model)
└── chunk_optimizer.py       # Automatic chunk size selection
```

> **Design Decision:** Composition over inheritance. Instead of mixins (which introduce MRO complexity with 6+ classes), the engine uses composition — each capability is a separate class accessed as attributes (`engine.aggregation`, `engine.temporal`, `engine.distributions`). This avoids silent method-name collisions and makes testing clearer.

## 2. Core API: `WeatherStatisticsEngine`

```python
"""WeatherRoutingTool/statistics/engine.py"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from WeatherRoutingTool.statistics.aggregation import Aggregation
from WeatherRoutingTool.statistics.temporal import Temporal
from WeatherRoutingTool.statistics.distributions import Distributions
from WeatherRoutingTool.statistics.correlation import Correlation
from WeatherRoutingTool.statistics.storage import DataStorageManager
from WeatherRoutingTool.statistics.config import StatisticsConfig

logger = logging.getLogger('WRT.Statistics')


class WeatherStatisticsEngine:
    """
    Central orchestrator for statistical analysis of WRT weather datasets.

    Operates on xr.Dataset objects with dimensions (time, latitude, longitude).
    All operations are NaN-aware and Dask-compatible.

    Parameters
    ----------
    dataset : xr.Dataset
        The weather dataset to analyze. Can be eager or Dask-backed.
    config : StatisticsConfig, optional
        Configuration for chunk sizes, precision, caching.
    cache_dir : Path, optional
        Directory for caching computed statistics.

    Examples
    --------
    >>> import xarray as xr
    >>> from WeatherRoutingTool.statistics import WeatherStatisticsEngine
    >>> ds = xr.open_dataset('weather.nc', chunks={'time': 10})
    >>> engine = WeatherStatisticsEngine(ds)
    >>> mean_wave = engine.aggregation.compute_mean(['VHM0'], dim='time')
    >>> corr = engine.correlation.compute_pairwise(['VHM0', 'VTPK'])
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        config: Optional[StatisticsConfig] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.ds = dataset
        self.config = config or StatisticsConfig()
        self.storage = DataStorageManager(cache_dir) if cache_dir else None

        self._validate_dataset()
        self._normalize_coordinates()

        # Composition: each capability is a separate object
        self.aggregation = Aggregation(self.ds, self.config)
        self.temporal = Temporal(self.ds, self.config)
        self.distributions = Distributions(self.ds, self.config)
        self.correlation = Correlation(self.ds, self.config)

        logger.info(
            f'StatisticsEngine initialized: {len(self.ds.data_vars)} vars, '
            f'{self._get_dataset_size_mb():.1f} MB, '
            f'dask_backed={self._is_dask_backed()}'
        )

    def _validate_dataset(self) -> None:
        """Validate dataset has expected dimensions."""
        required_dims = {'time', 'latitude', 'longitude'}
        actual_dims = set(self.ds.dims)
        missing = required_dims - actual_dims
        if missing:
            raise ValueError(
                f"Dataset missing required dimensions: {missing}. "
                f"Available: {actual_dims}"
            )

    def _normalize_coordinates(self) -> None:
        """Normalize longitude to -180..180 convention."""
        if float(self.ds.longitude.max()) > 180:
            self.ds = self.ds.assign_coords(
                longitude=((self.ds.longitude + 180) % 360) - 180
            ).sortby('longitude')
            logger.info('Normalized longitude from 0-360 to -180..180')

    def _get_dataset_size_mb(self) -> float:
        """Estimate total dataset size in MB."""
        return sum(var.nbytes for var in self.ds.data_vars.values()) / (1024 * 1024)

    def _is_dask_backed(self) -> bool:
        """Check if dataset uses Dask arrays."""
        return any(hasattr(self.ds[v], 'dask') for v in self.ds.data_vars)

    def get_nan_report(self) -> Dict[str, Dict]:
        """Report NaN statistics for all variables."""
        report = {}
        for var_name in self.ds.data_vars:
            var = self.ds[var_name]
            total = int(np.prod(var.shape))
            nan_count = int(var.isnull().sum().compute()
                           if self._is_dask_backed() else var.isnull().sum())
            report[var_name] = {
                'total_elements': total,
                'nan_count': nan_count,
                'nan_percent': 100.0 * nan_count / total if total > 0 else 0,
                'dtype': str(var.dtype),
                'units': var.attrs.get('units', 'unknown'),
            }
        return report

    def subset(
        self,
        lat_range: tuple = None,
        lon_range: tuple = None,
        time_range: tuple = None,
        variables: List[str] = None,
    ) -> 'WeatherStatisticsEngine':
        """Create a new engine on a spatiotemporally subsetted dataset."""
        ds = self.ds
        if variables:
            ds = ds[variables]
        if lat_range:
            ds = ds.sel(latitude=slice(*lat_range))
        if lon_range:
            ds = ds.sel(longitude=slice(*lon_range))
        if time_range:
            ds = ds.sel(time=slice(*time_range))
        return WeatherStatisticsEngine(
            ds, self.config,
            self.storage.cache_dir if self.storage else None
        )
```

## 3. Aggregation (NaN-Aware)

```python
"""WeatherRoutingTool/statistics/aggregation.py"""

from typing import List, Optional, Union
import numpy as np
import xarray as xr
from WeatherRoutingTool.statistics.config import StatisticsConfig


class Aggregation:
    """NaN-aware spatial and temporal aggregation operations."""

    def __init__(self, ds: xr.Dataset, config: StatisticsConfig):
        self.ds = ds
        self.config = config

    def _select(self, variables: Optional[List[str]]) -> xr.Dataset:
        return self.ds[variables] if variables else self.ds

    def compute_mean(
        self,
        variables: Optional[List[str]] = None,
        dim: Union[str, List[str]] = 'time',
        skipna: bool = True,
        min_count: int = 1,
    ) -> xr.Dataset:
        """
        NaN-aware mean over specified dimension(s).

        Parameters
        ----------
        skipna : bool
            If True, ignore NaN values. Default True.
        min_count : int
            Minimum number of valid (non-NaN) values required.
            If fewer valid values exist, result is NaN. Default 1.
        """
        return self._select(variables).astype(np.float64).mean(
            dim=dim, skipna=skipna, min_count=min_count)

    def compute_std(
        self,
        variables: Optional[List[str]] = None,
        dim: Union[str, List[str]] = 'time',
        ddof: int = 1,
        skipna: bool = True,
    ) -> xr.Dataset:
        """NaN-aware standard deviation."""
        return self._select(variables).astype(np.float64).std(
            dim=dim, ddof=ddof, skipna=skipna)

    def compute_percentile(
        self,
        variables: Optional[List[str]] = None,
        q: Union[float, List[float]] = 0.95,
        dim: Union[str, List[str]] = 'time',
        skipna: bool = True,
    ) -> xr.Dataset:
        """NaN-aware percentile computation."""
        return self._select(variables).astype(np.float64).quantile(
            q, dim=dim, skipna=skipna)

    def compute_spatial_mean(
        self,
        variables: Optional[List[str]] = None,
        weights: Optional[xr.DataArray] = None,
    ) -> xr.Dataset:
        """
        Area-weighted spatial mean.

        Uses cosine-latitude weighting by default to account for
        convergence of meridians at high latitudes.
        """
        subset = self._select(variables)
        if weights is None:
            weights = np.cos(np.deg2rad(subset.latitude))
        return subset.weighted(weights).mean(dim=['latitude', 'longitude'])
```

## 4. Correlation (Dask-Compatible — Fixed)

```python
"""WeatherRoutingTool/statistics/correlation.py"""

from typing import List, Optional
import numpy as np
import xarray as xr
from WeatherRoutingTool.statistics.config import StatisticsConfig


class Correlation:
    """
    Dask-compatible cross-variable correlation analysis.

    CRITICAL: Does NOT use .values.ravel() which would materialize the entire
    dataset. Uses xr.corr() for pairwise correlation, which operates on
    Dask arrays natively.
    """

    def __init__(self, ds: xr.Dataset, config: StatisticsConfig):
        self.ds = ds
        self.config = config

    def compute_pairwise(
        self,
        variables: Optional[List[str]] = None,
        dim: str = 'time',
    ) -> xr.DataArray:
        """
        Compute pairwise Pearson correlation matrix using xr.corr().

        This is Dask-compatible: xr.corr() operates on chunked arrays
        without materializing the full dataset.

        Parameters
        ----------
        variables : list of str, optional
            Variables to include. Default: all numeric variables.
        dim : str
            Dimension along which to compute correlation.

        Returns
        -------
        xr.DataArray
            Correlation matrix with dims (variable_1, variable_2).
            Values are spatial means of per-pixel correlations.
        """
        if variables is None:
            variables = [v for v in self.ds.data_vars
                         if np.issubdtype(self.ds[v].dtype, np.number)]

        n = len(variables)
        corr_values = np.full((n, n), np.nan)

        for i in range(n):
            corr_values[i, i] = 1.0
            for j in range(i + 1, n):
                # xr.corr() is Dask-compatible — no materialization
                corr_ij = xr.corr(
                    self.ds[variables[i]],
                    self.ds[variables[j]],
                    dim=dim,
                )
                # Spatial mean of per-pixel correlation
                weights = np.cos(np.deg2rad(self.ds.latitude))
                mean_corr = float(corr_ij.weighted(weights).mean().compute())
                corr_values[i, j] = mean_corr
                corr_values[j, i] = mean_corr

        return xr.DataArray(
            corr_values,
            dims=['variable_1', 'variable_2'],
            coords={'variable_1': variables, 'variable_2': variables},
            attrs={'method': 'pearson', 'dimension': dim,
                   'note': 'Spatial mean of per-pixel correlations'},
        )

    def compute_spatial_correlation(
        self,
        var1: str,
        var2: str,
    ) -> xr.DataArray:
        """
        Per-pixel correlation between two variables along time.

        Returns a 2D spatial map of correlation values.
        Dask-compatible via xr.corr().
        """
        return xr.corr(
            self.ds[var1],
            self.ds[var2],
            dim='time',
        )
```

## 5. Distribution Fitting (NaN-Safe)

```python
"""WeatherRoutingTool/statistics/distributions.py"""

from typing import List, Optional
import numpy as np
import xarray as xr
from scipy import stats as scipy_stats
from WeatherRoutingTool.statistics.config import StatisticsConfig


class Distributions:
    """NaN-safe parametric distribution fitting."""

    SUPPORTED = {
        'normal': scipy_stats.norm,
        'weibull': scipy_stats.weibull_min,
        'rayleigh': scipy_stats.rayleigh,
        'lognormal': scipy_stats.lognorm,
    }

    def __init__(self, ds: xr.Dataset, config: StatisticsConfig):
        self.ds = ds
        self.config = config

    def fit(
        self,
        variable: str,
        distribution: str = 'weibull',
        dim: str = 'time',
        min_samples: int = 30,
    ) -> xr.Dataset:
        """
        Fit a parametric distribution per spatial cell along `dim`.

        NaN handling: Filters NaN before fitting. Returns NaN parameters
        if fewer than min_samples valid values exist.

        Returns xr.Dataset with named parameter variables (not index-sliced).
        """
        if distribution not in self.SUPPORTED:
            raise ValueError(f"Unsupported: {distribution}. "
                             f"Options: {list(self.SUPPORTED.keys())}")

        dist_func = self.SUPPORTED[distribution]

        def _fit_cell(data):
            valid = data[~np.isnan(data)]
            if len(valid) < min_samples:
                return np.array([np.nan] * 5)
            try:
                params = dist_func.fit(valid, method='mle')
                ks_stat, p_val = scipy_stats.kstest(valid, distribution, args=params)
                padded = list(params) + [np.nan] * (3 - len(params))
                return np.array(padded[:3] + [ks_stat, p_val])
            except Exception:
                return np.array([np.nan] * 5)

        result = xr.apply_ufunc(
            _fit_cell,
            self.ds[variable],
            input_core_dims=[[dim]],
            output_core_dims=[['param']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={'output_sizes': {'param': 5}},
        )

        # Named variables instead of index slicing
        names = ['shape', 'loc', 'scale', 'ks_statistic', 'p_value']
        ds_out = xr.Dataset()
        for i, name in enumerate(names):
            ds_out[f'{variable}_{distribution}_{name}'] = result.isel(param=i)

        ds_out.attrs['distribution'] = distribution
        ds_out.attrs['source_variable'] = variable
        ds_out.attrs['min_samples'] = min_samples
        return ds_out
```

## 6. Temporal Analytics

```python
"""WeatherRoutingTool/statistics/temporal.py"""

from typing import List, Optional, Union
import numpy as np
import xarray as xr
from WeatherRoutingTool.statistics.config import StatisticsConfig


class Temporal:
    """Time-series analytics: rolling, resampling, rate-of-change."""

    def __init__(self, ds: xr.Dataset, config: StatisticsConfig):
        self.ds = ds
        self.config = config

    def compute_rolling(
        self,
        variables: Optional[List[str]] = None,
        window: int = 6,
        operation: str = 'mean',
        min_periods: int = 1,
    ) -> xr.Dataset:
        """
        Rolling window statistics along time dimension.

        min_periods controls NaN handling: if fewer than min_periods
        valid values exist in the window, the result is NaN.
        """
        subset = self.ds[variables] if variables else self.ds
        roller = subset.rolling(time=window, center=True, min_periods=min_periods)
        return getattr(roller, operation)()

    def compute_resample(
        self,
        variables: Optional[List[str]] = None,
        freq: str = '6H',
        operation: str = 'mean',
    ) -> xr.Dataset:
        """Temporal resampling (e.g., 3-hourly → 6-hourly)."""
        subset = self.ds[variables] if variables else self.ds
        resampler = subset.resample(time=freq)
        return getattr(resampler, operation)()

    def compute_rate_of_change(
        self, variable: str
    ) -> xr.DataArray:
        """First-order time derivative: ∂(variable)/∂t."""
        var = self.ds[variable].astype(np.float64)
        return var.differentiate('time')
```

## 7. Design Guarantees

| Property | Guarantee |
|----------|-----------|
| NaN safety | All aggregation functions accept `skipna` parameter; distributions pre-filter NaN |
| Dask compatibility | No `.values.ravel()` in hot paths; correlation uses `xr.corr()`; distributions use `apply_ufunc(dask='parallelized')` |
| Coordinate preservation | All returned datasets retain lat/lon/time coordinates |
| Numerical precision | All reductions upcast to float64 before computation |
| Thread safety | Each capability class is stateless (reads from `self.ds` but doesn't modify it) |
| No MRO issues | Composition pattern (attribute access) instead of mixin inheritance |
