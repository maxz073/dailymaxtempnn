# Daily Max Temperature Neural Network

A bias-correction neural network that predicts daily maximum temperatures for 20 US cities by learning the residual between NWS recorded highs and numerical weather prediction (NWP) forecast ensemble means.

## How It Works

Instead of predicting temperature from scratch, the model corrects systematic biases in existing NWP forecasts:

```
Predicted Actual Temperature = NWP Forecast Ensemble Mean + Predicted Residual
```

The model outputs a heteroscedastic Gaussian (mu, sigma), providing both a point prediction and calibrated uncertainty estimate.

## Architecture

- **Type**: MLP with city embeddings
- **Input**: 96 continuous features + 8-dim city embedding
- **Hidden layers**: [128, 64, 32] with BatchNorm, GELU, dropout
- **Output**: (mu, sigma) — heteroscedastic Gaussian
- **Loss**: Gaussian negative log-likelihood
- **Parameters**: ~24,500

## Features (96 continuous)

| Category | Features | Count |
|---|---|---|
| Raw NWP forecasts | GFS, ECMWF, ICON, GEM, JMA, HRRR | 6 |
| Ensemble stats | mean, std, range, IQR | 4 |
| Forecast disagreement | median, median-mean gap, skewness, kurtosis | 4 |
| Pairwise spreads | ECMWF-GFS, ICON-GFS, GEM-GFS, JMA-GFS, HRRR-GFS | 5 |
| Forecast momentum | mean delta, std delta, 3-day trend, spread momentum | 4 |
| Climate indices | ENSO (ONI), AO, NAO, PNA | 4 |
| Lagged meteorology (lag-1) | cloud cover, dewpoint, wind, pressure, precip | 5 |
| Lagged meteorology (lag-2) | cloud cover, dewpoint, wind, pressure, precip | 5 |
| Moisture / humidity | dewpoint depression, precip×cloud, snow flag, RH proxy, high dewpoint flag | 5 |
| Pressure / frontal | pressure tendency, wind direction, wind×pressure change, pressure acceleration | 4 |
| Wind direction encoding | sin/cos wind direction lag-1 | 2 |
| Onshore wind | onshore wind component, wind speed × onshore | 2 |
| Wind direction change | circular difference lag-1 vs lag-2 | 1 |
| Lagged hourly temps | 6am, 9am, noon, 3pm, diurnal range (yesterday) | 5 |
| Overnight low | yesterday's minimum temperature | 1 |
| Rates of change | overnight cooling rate, morning warming rate | 2 |
| Temperature regime | 7-day volatility, 3-day range, yesterday rank (training quantiles) | 3 |
| Rolling forecast bias (mean) | 3-day, 7-day, 14-day, 30-day | 4 |
| Rolling forecast bias (std) | 7-day, 14-day, 30-day | 3 |
| Forecast bias regime | cold bias regime flag (14-day abs mean > 1°F) | 1 |
| Cloud / precip | cloud cover 2-day trend, precipitation probability proxy | 2 |
| Climatology anomalies | climatological normal, forecast anomaly, yesterday's anomaly | 3 |
| Normalized anomaly | forecast mean vs climatological extreme (using clim std) | 1 |
| Forecast ensemble bimodality | abs(median - mean) / (std + 0.1) | 1 |
| Seasonal model skill | per-city per-month mean abs residual (training data) | 1 |
| City static | lat, lon, elevation, coastal, desert, continentality | 6 |
| Calendar | sin/cos day-of-year, sin/cos month | 4 |
| Solar / astronomical | day length, solar declination, days since winter solstice | 3 |
| Cross interactions | mean×std, spread×cloud, elevation×pressure, coastal×wind, continentality×season | 5 |

## Performance

### Out-of-Sample (Test: Apr 2025–Apr 2026, full year)

| Metric | Value |
|---|---|
| MAE | 1.25°F |
| RMSE | 1.66°F |
| Bias | -0.01°F |
| R² | 0.992 |
| Correlation | 0.996 |

### vs NWP Forecasts (Test Set)

| Model | MAE | Within 1°F | Within 2°F |
|---|---|---|---|
| **NN Bias-Correction** | **1.25°F** | **50.7%** | **81.0%** |
| HRRR | 2.07°F | 37.9% | 63.5% |
| NWP Ens. Mean | 2.28°F | 26.3% | 52.4% |
| GFS | 2.37°F | 29.5% | 53.8% |
| ECMWF | 2.85°F | 21.6% | 42.1% |
| JMA | 4.34°F | 12.0% | 25.4% |

The model beats all 6 individual NWP models and their ensemble mean on every city.

## Data

All data is included in the `data/` directory:

- `data/nws_daily/` — NWS daily recorded highs from ACIS (ground truth target)
- `data/weather_forecasts/` — Historical NWP model forecasts from Open-Meteo (GFS, ECMWF, ICON, GEM, JMA, HRRR)
- `data/weather_archive/` — Historical daily + hourly weather from Open-Meteo (features)
- `data/climate_indices/` — ENSO, AO, NAO, PNA from NOAA CPC

**20 cities**: New York, Chicago, Miami, Boston, Los Angeles, Austin, San Francisco, Dallas, Philadelphia, Phoenix, Oklahoma City, Denver, Washington DC, San Antonio, Houston, Minneapolis, Atlanta, Seattle, Las Vegas, New Orleans

**Date range**: 2022-01-01 to 2026-04-16

## Usage

### Train the model

```bash
pip install -r requirements.txt
python model1_forecast.py
```

This builds features, trains the model with early stopping, and saves:
- `checkpoints/model1_best.pt` — best model weights
- `checkpoints/model1_scaler.pkl` — fitted StandardScaler
- `checkpoints/model1_preds_train.parquet` — in-sample predictions
- `checkpoints/model1_preds_val.parquet` — validation predictions
- `checkpoints/model1_preds_test.parquet` — test predictions

### Compare against NWP forecasts

```bash
python compare_models.py
```

Prints MAE/RMSE/bias tables comparing the neural net against GFS, ECMWF, ICON, GEM, JMA, HRRR, and their ensemble mean across train/val/test splits.

### Performance analysis notebook

```bash
jupyter notebook performance.ipynb
```

In-sample and out-of-sample evaluation: overall metrics, per-city breakdown, calibration plots, residual diagnostics, time series visualization, MAE heatmaps.

### Refresh data (optional)

```bash
python data_fetch.py
```

Re-fetches all data from APIs (NWS/ACIS, Open-Meteo, NOAA CPC). Existing files are skipped.

### Economic value analyses

```bash
python energy_value.py      # ERCOT energy procurement savings
python kalshi_backtest.py   # Kalshi temperature market backtest
```

See [ECONOMIC_ANALYSIS.md](ECONOMIC_ANALYSIS.md) for full methodology, assumptions, and parameter documentation.

## Data Splits

| Split | Date Range | Purpose |
|---|---|---|
| Train | 2022-01-01 to 2024-04-16 | Model training |
| Validation | 2024-04-17 to 2025-04-16 | Early stopping / hyperparameter tuning (1 full year) |
| Test | 2025-04-17 to 2026-04-16 | Out-of-sample evaluation (1 full year) |
