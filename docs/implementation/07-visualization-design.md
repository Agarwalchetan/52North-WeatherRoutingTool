# 07 — Visualization Design

## 1. Design Principles

1. **Leverage xarray's `.plot()`** for standard plots (spatial heatmaps, time series).
2. **Reuse WRT's cartopy/matplotlib setup** from `utils/graphics.py` for projections.
3. **Variable-adaptive** — plots auto-configure for any subset of weather variables, not hard-coded to 4.
4. **NaN-aware** — mask land pixels in spatial plots.

## 2. Plot Types

### 2.1 Spatial Heatmap (Time-Averaged)

Uses `xr.DataArray.plot.pcolormesh()` with cartopy projection, consistent with WRT's existing plotting conventions.

```python
def plot_spatial_heatmap(
    data: xr.DataArray,
    ax=None,
    projection=None,
    title: str = None,
    cmap: str = 'viridis',
    robust: bool = True,
):
    """Spatial heatmap with projection handling and NaN masking."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    if projection is None:
        projection = ccrs.PlateCarree()
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': projection},
                                figsize=_adaptive_figsize(data))

    data.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap=cmap, robust=robust,  # robust=True clips to 2nd/98th percentile
        add_labels=True,
    )
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.gridlines(draw_labels=True,  alpha=0.3)
    if title:
        ax.set_title(title)
    return ax

def _adaptive_figsize(data: xr.DataArray) -> tuple:
    """Scale figure size based on spatial extent."""
    lat_range = float(data.latitude.max() - data.latitude.min())
    lon_range = float(data.longitude.max() - data.longitude.min())
    aspect = lon_range / max(lat_range, 1)
    w = min(max(aspect * 6, 8), 16)
    h = min(max(6, w / aspect), 12)
    return (w, h)
```

### 2.2 Time Series at a Point or Region

```python
def plot_time_series(
    ds: xr.Dataset,
    variables: list,
    lat: float = None, lon: float = None,
    ax=None,
):
    """Time series plot — either at a point (.sel) or spatial mean."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    for var in variables:
        if lat is not None and lon is not None:
            series = ds[var].sel(latitude=lat, longitude=lon, method='nearest')
        else:
            weights = np.cos(np.deg2rad(ds.latitude))
            series = ds[var].weighted(weights).mean(dim=['latitude', 'longitude'])

        series.plot(ax=ax, label=f"{var} ({ds[var].attrs.get('units', '')})")

    ax.legend()
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    return ax
```

### 2.3 Distribution Plot

```python
def plot_distribution(
    data: xr.DataArray,
    fit_result: xr.Dataset = None,
    variable_name: str = '',
    bins: int = 50,
):
    """Histogram with optional fitted distribution overlay."""
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    fig, ax = plt.subplots(figsize=(10, 6))

    values = data.values.ravel()
    values = values[~np.isnan(values)]

    ax.hist(values, bins=bins, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label='Data')

    if fit_result is not None:
        x = np.linspace(values.min(), values.max(), 200)
        # Extract fit parameters (spatial mean for display)
        shape = float(fit_result[f'{variable_name}_weibull_shape'].mean())
        loc = float(fit_result[f'{variable_name}_weibull_loc'].mean())
        scale = float(fit_result[f'{variable_name}_weibull_scale'].mean())
        pdf = scipy_stats.weibull_min.pdf(x, shape, loc, scale)
        ax.plot(x, pdf, 'r-', lw=2, label=f'Weibull fit (k={shape:.2f}, λ={scale:.2f})')

    ax.set_xlabel(f'{variable_name} ({data.attrs.get("units", "")})')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(f'Distribution: {variable_name}')
    return ax
```

### 2.4 Correlation Matrix

```python
def plot_correlation_matrix(
    corr: xr.DataArray,
):
    """Annotated heatmap of pairwise correlation matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    labels = corr.coords['variable_1'].values
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title('Cross-Variable Correlation')
    return ax
```

### 2.5 Route Weather Profile (Variable-Adaptive)

```python
def plot_route_weather_profile(
    route_stats: xr.Dataset,
    variables: list = None,
):
    """Multi-panel route weather profile — adapts to number of variables."""
    import matplotlib.pyplot as plt

    if variables is None:
        variables = list(route_stats.data_vars)

    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        data = route_stats[var]
        ax.plot(data.values, color='steelblue', linewidth=1.5)
        ax.fill_between(range(len(data)), data.values, alpha=0.15, color='steelblue')
        ax.set_ylabel(f'{var} ({route_stats[var].attrs.get("units", "")})')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Route Waypoint Index')
    fig.suptitle('Weather Conditions Along Route', fontweight='bold')
    plt.tight_layout()
    return fig
```

## 3. Integration with Existing WRT Plotting

WRT uses `utils/graphics.py` which provides:
- `custom_basemap()` — creates a basemap with coastlines for cartopy
- `plot_routes()` — overlays routes on maps

The statistics visualization module reuses:
- The same `cartopy.crs.PlateCarree()` projection
- The same coastline resolution (`50m`)
- The same `matplotlib.colors` normalization patterns

New plots can be composed with existing WRT route plots:
```python
fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                          subplot_kw={'projection': ccrs.PlateCarree()})
# WRT existing route plot
WRT_plot_routes(axes[0], route)
# New statistical heatmap
plot_spatial_heatmap(mean_wave_height, ax=axes[1], title='Mean VHM0')
```
