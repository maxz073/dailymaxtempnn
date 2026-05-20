"""
Forecast Bias-Correction Model.

Predicts the residual (nws_recorded_high - forecast_mean) using:
  - Raw forecast model outputs + ensemble statistics
  - Meteorological proxy features from archive data
  - Hourly temperature path features
  - Rolling historical forecast bias per city
  - City static features + calendar encoding
  - City embedding

Target: NWS daily recorded high (official recorded temperature).
At inference time: predicted_actual = forecast_mean + predicted_residual
"""
import os
import sys
import logging

import numpy as np
import pandas as pd
from scipy.stats import skew as _skew, kurtosis as _kurtosis

import torch
from torch.utils.data import DataLoader

# Shared modules
import config as cfg
import data_fetch
import feature_utils
import training
import evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORECAST_COLS = [
    "fcst_gfs_global",
    "fcst_ecmwf_ifs025",
    "fcst_icon_seamless",
    "fcst_gem_seamless",
    "fcst_jma_seamless",
    "fcst_ncep_hrrr_conus",
]


# ── Feature engineering ─────────────────────────────────────────────

def _build_hourly_temp_path(hourly: pd.DataFrame) -> pd.DataFrame:
    """Extract per-city-day temperature at 6am, 9am, noon, 3pm and diurnal range."""
    log.info("Building hourly temperature path features...")
    df = hourly.copy()
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.normalize()

    # Filter to hours of interest
    target_hours = [6, 9, 12, 15]
    df = df[df["hour"].isin(target_hours)]

    # Pivot: one column per hour
    pivot = df.pivot_table(
        index=["date", "ticker"],
        columns="hour",
        values="temperature_2m",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(
        columns={6: "temp_6am", 9: "temp_9am", 12: "temp_noon", 15: "temp_3pm"},
        inplace=True,
    )

    # Diurnal range from the four sampled hours
    hour_cols = ["temp_6am", "temp_9am", "temp_noon", "temp_3pm"]
    pivot["temp_path_range"] = pivot[hour_cols].max(axis=1) - pivot[hour_cols].min(axis=1)

    return pivot


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build the full feature matrix.

    Returns
    -------
    train, val, test : pd.DataFrame
        Each contains continuous features, city_idx, y_resid, date, ticker, forecast_mean.
    cont_cols : list[str]
        Names of continuous feature columns (everything the scaler should touch).
    """
    log.info("=== Building features ===")

    # ── Load raw data ───────────────────────────────────────────────
    archive_daily = data_fetch.load_archive_daily()
    archive_hourly = data_fetch.load_archive_hourly()
    forecasts = data_fetch.load_forecasts()
    nws_daily = data_fetch.load_nws_daily()
    log.info("Loaded archive_daily=%d, archive_hourly=%d, forecasts=%d, nws_daily=%d",
             len(archive_daily), len(archive_hourly), len(forecasts), len(nws_daily))

    # ── Merge forecasts with archive (features) and NWS (target) ────
    df = forecasts.merge(archive_daily, on=["date", "ticker"], how="inner")
    df = df.merge(nws_daily[["date", "ticker", "nws_high"]], on=["date", "ticker"], how="inner")
    log.info("After merge forecast+archive+nws: %d rows", len(df))

    # ── (a) Forecast ensemble statistics ────────────────────────────
    # Fill per-city mean for any NaN forecast columns
    for col in FORECAST_COLS:
        city_means = df.groupby("ticker")[col].transform("mean")
        df[col] = df[col].fillna(city_means)

    fcst_values = df[FORECAST_COLS]
    df["forecast_mean"] = fcst_values.mean(axis=1)
    df["forecast_std"] = fcst_values.std(axis=1)
    df["forecast_range"] = fcst_values.max(axis=1) - fcst_values.min(axis=1)
    df["forecast_iqr"] = fcst_values.quantile(0.75, axis=1) - fcst_values.quantile(0.25, axis=1)

    # Forecast disagreement structure
    df["forecast_median"] = fcst_values.median(axis=1)
    df["forecast_median_mean_gap"] = df["forecast_median"] - df["forecast_mean"]
    df["forecast_skew"] = _skew(fcst_values.values, axis=1, nan_policy="omit")
    df["forecast_kurtosis"] = _kurtosis(fcst_values.values, axis=1, nan_policy="omit")

    # Forecast ensemble bimodality (feature 18)
    df["forecast_ensemble_bimodality"] = (
        (df["forecast_median"] - df["forecast_mean"]).abs() / (df["forecast_std"] + 0.1)
    )

    # ── (a2) Forecast trend / momentum ─────────────────────────────
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    _fm_lag1 = df.groupby("ticker")["forecast_mean"].shift(1)
    _fs_lag1 = df.groupby("ticker")["forecast_std"].shift(1)
    _fr_lag1 = df.groupby("ticker")["forecast_range"].shift(1)
    df["forecast_mean_delta1"] = df["forecast_mean"] - _fm_lag1
    df["forecast_std_delta1"] = df["forecast_std"] - _fs_lag1
    df["forecast_mean_roll3_trend"] = (
        df.groupby("ticker")["forecast_mean_delta1"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["forecast_spread_momentum"] = df["forecast_range"] - _fr_lag1

    # ── (b) Target: residual (NWS recorded high minus forecast mean) ─
    df["y_resid"] = df["nws_high"] - df["forecast_mean"]

    # ── (b2) Seasonal model skill lookup (feature 19) ──────────────
    # Per-city per-month average absolute residual from training data only.
    _train_mask = df["date"] <= pd.Timestamp(cfg.TRAIN_END)
    _train_skill = df.loc[_train_mask].copy()
    _train_skill["_month"] = _train_skill["date"].dt.month
    _skill_lookup = (
        _train_skill.groupby(["ticker", "_month"])["y_resid"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .rename(columns={"y_resid": "seasonal_model_skill"})
    )
    df["_month"] = df["date"].dt.month
    df = df.merge(_skill_lookup, on=["ticker", "_month"], how="left")
    df.drop(columns=["_month"], inplace=True)

    # ── (c) Pairwise spreads ────────────────────────────────────────
    df["spread_ecmwf_gfs"] = df["fcst_ecmwf_ifs025"] - df["fcst_gfs_global"]
    df["spread_icon_gfs"] = df["fcst_icon_seamless"] - df["fcst_gfs_global"]
    df["spread_gem_gfs"] = df["fcst_gem_seamless"] - df["fcst_gfs_global"]
    df["spread_jma_gfs"] = df["fcst_jma_seamless"] - df["fcst_gfs_global"]
    df["spread_hrrr_gfs"] = df["fcst_ncep_hrrr_conus"] - df["fcst_gfs_global"]

    # ── (c2) Climate indices ───────────────────────────────────────
    climate = data_fetch.load_climate_indices()
    df = df.merge(climate, on="date", how="left")
    for col in ["enso_oni", "ao", "nao", "pna"]:
        df[col] = df[col].ffill()

    # ── (d) Lagged meteorological features (lag-1 and lag-2) ────────
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    meteo_raw = [
        "cloud_cover_mean", "dewpoint_2m_mean", "wind_speed_10m_max",
        "surface_pressure_mean", "precipitation_sum",
    ]
    for col in meteo_raw:
        lags = [1, 2, 3] if col == "surface_pressure_mean" else [1, 2]
        df = feature_utils.add_lags(df, col=col, lags=lags, group_col="ticker")
    meteo_cols_lag1 = [f"{c}_lag1" for c in meteo_raw]
    meteo_cols_lag2 = [f"{c}_lag2" for c in meteo_raw]

    # Wind direction lag-2 (lag-1 is added below in the pressure section)
    df = feature_utils.add_lags(df, col="wind_direction_10m_dominant", lags=[1, 2], group_col="ticker")
    # Rename to match existing convention (wind_direction_lag1 is used later)
    df.rename(columns={
        "wind_direction_10m_dominant_lag1": "_wind_dir_lag1_dup",
        "wind_direction_10m_dominant_lag2": "wind_direction_lag2",
    }, inplace=True)

    # ── (d2) Moisture / humidity derived ───────────────────────────
    df["dewpoint_depression_lag1"] = (
        df.groupby("ticker")["temperature_2m_max"].shift(1) - df["dewpoint_2m_mean_lag1"]
    )
    df["precip_x_cloud_lag1"] = df["precipitation_sum_lag1"] * df["cloud_cover_mean_lag1"]
    df["snow_flag_lag1"] = (df.groupby("ticker")["snowfall_sum"].shift(1) > 0).astype(float)

    # ── (d3) Pressure change / frontal indicators ──────────────────
    df["pressure_tendency_lag1"] = df["surface_pressure_mean_lag1"] - df["surface_pressure_mean_lag2"]
    df["wind_direction_lag1"] = df.groupby("ticker")["wind_direction_10m_dominant"].shift(1)
    df["wind_x_pressure_change_lag1"] = (
        df["wind_speed_10m_max_lag1"] * df["pressure_tendency_lag1"].abs()
    )

    # ── (e) Lagged temperature path (yesterday's hourly temps) ──────
    hourly_path = _build_hourly_temp_path(archive_hourly)
    hourly_path_cols = ["temp_6am", "temp_9am", "temp_noon", "temp_3pm", "temp_path_range"]
    hourly_path = hourly_path.rename(columns={c: f"{c}_lag1" for c in hourly_path_cols})
    df = df.merge(hourly_path, on=["date", "ticker"], how="left")
    for col in [f"{c}_lag1" for c in hourly_path_cols]:
        df[col] = df.groupby("ticker")[col].shift(1)
    df["temperature_2m_min_lag1"] = df.groupby("ticker")["temperature_2m_min"].shift(1)

    # ── (f) Rolling forecast bias (mean + std, shifted by 1) ───────
    df = feature_utils.add_rolling(
        df, col="y_resid", windows=[3, 7, 14, 30],
        stats=["mean", "std"], group_col="ticker",
    )

    # ── (g) City static features ────────────────────────────────────
    df = feature_utils.add_city_static_features(df)

    # ── (g1b) Wind direction encoding + onshore wind ────────────────
    df = feature_utils.add_wind_direction_encoding(df)
    df = feature_utils.add_onshore_wind_features(df)

    # ── (g2) Climatological normals + anomalies ────────────────────
    clim_normals = feature_utils.compute_climatological_normals(
        df, train_end=pd.Timestamp(cfg.TRAIN_END), temp_col="nws_high",
    )
    df["doy"] = df["date"].dt.dayofyear
    df = df.merge(clim_normals, on=["ticker", "doy"], how="left")
    df["forecast_clim_anomaly"] = df["forecast_mean"] - df["clim_normal"]
    _nws_lag1 = df.groupby("ticker")["nws_high"].shift(1)
    df["yesterday_high_clim_anomaly"] = _nws_lag1 - df["clim_normal"]

    # ── (g2b) Climatological std + normalized anomaly (feature 20) ─
    _train_clim = df[df["date"] <= pd.Timestamp(cfg.TRAIN_END)].copy()
    _clim_std = (_train_clim.groupby(["ticker", "doy"])["nws_high"]
                 .std().reset_index().rename(columns={"nws_high": "clim_std"}))
    df = df.merge(_clim_std, on=["ticker", "doy"], how="left")
    df["forecast_mean_vs_clim_extreme"] = (
        (df["forecast_mean"] - df["clim_normal"]).abs() / (df["clim_std"].fillna(0) + 0.1)
    )

    df.drop(columns=["doy"], inplace=True)

    # ── (g3) Solar / astronomical features ─────────────────────────
    df = feature_utils.add_solar_features(df)

    # ── (h) Calendar features ───────────────────────────────────────
    df = feature_utils.add_calendar_features(df)

    # ── (i) City index for embedding ────────────────────────────────
    df = feature_utils.add_city_index(df)

    # ── (j) Cross-feature interactions ─────────────────────────────
    df["forecast_mean_x_std"] = df["forecast_mean"] * df["forecast_std"]
    df["spread_range_x_cloud"] = df["forecast_range"] * df["cloud_cover_mean_lag1"]
    df["elevation_x_pressure"] = df["elevation_ft"] * df["surface_pressure_mean_lag1"]
    df["coastal_x_wind"] = df["coastal"] * df["wind_speed_10m_max_lag1"]

    # ── (k) Additional derived features ──────────────────────────────

    # Humidity/moisture (features 5-6)
    df["relative_humidity_proxy_lag1"] = (
        100.0 - 5.0 * (df.groupby("ticker")["temperature_2m_max"].shift(1) - df["dewpoint_2m_mean_lag1"])
    ).clip(0, 100)
    df["high_dewpoint_flag"] = (df["dewpoint_2m_mean_lag1"] > 70).astype(float)

    # Temperature regime (features 7-9)
    df["temp_volatility_7d"] = (
        df.groupby("ticker")["nws_high"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    )
    df["temp_range_3d"] = (
        df.groupby("ticker")["nws_high"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).max()
                   - x.shift(1).rolling(3, min_periods=1).min())
    )

    # max_temp_yesterday_rank: percentile rank using training quantiles only
    _train_quantiles = (
        df[df["date"] <= pd.Timestamp(cfg.TRAIN_END)]
        .groupby("ticker")["nws_high"]
        .quantile(np.arange(0, 1.01, 0.01))
        .reset_index()
        .rename(columns={"level_1": "q", "nws_high": "val"})
    )
    _nws_lag1_vals = df.groupby("ticker")["nws_high"].shift(1)
    _rank_values = np.zeros(len(df))
    for ticker in df["ticker"].Letunique():
        mask = df["ticker"] == ticker
        q_vals = _train_quantiles[_train_quantiles["ticker"] == ticker]["val"].values
        if len(q_vals) > 0:
            _rank_values[mask.values] = np.searchsorted(
                np.sort(q_vals), _nws_lag1_vals[mask].values, side="right"
            ) / len(q_vals)
    df["max_temp_yesterday_rank"] = _rank_values

    # Forecast bias regime (feature 10)
    df["forecast_cold_bias_regime"] = (df["y_resid_roll14_mean"].abs() > 1.0).astype(float)

    # Cloud/precip (features 11-12)
    df["cloud_cover_trend_2d"] = df["cloud_cover_mean_lag1"] - df["cloud_cover_mean_lag2"]
    df["precip_probability_proxy"] = (
        np.maximum(0, 1.0 - df["dewpoint_depression_lag1"] / 20.0)
        * df["cloud_cover_mean_lag1"] / 100.0
    )

    # Rates of change (features 13-14)
    df["overnight_cooling_rate"] = df["temp_6am_lag1"] - df["temperature_2m_min_lag1"]
    df["morning_warming_rate"] = (df["temp_noon_lag1"] - df["temp_6am_lag1"]) / 6.0

    # Pressure dynamics (feature 15)
    df["pressure_acceleration"] = (
        df["pressure_tendency_lag1"]
        - (df["surface_pressure_mean_lag2"] - df["surface_pressure_mean_lag3"])
    )

    # Wind direction change (feature 16)
    _wd_diff = (df["wind_direction_lag1"] - df["wind_direction_lag2"]).abs()
    df["wind_direction_change_lag12"] = np.minimum(_wd_diff, 360.0 - _wd_diff)

    # Interaction/seasonal (feature 17)
    df["continentality_x_season"] = df["continentality"] * df["cos_doy"]

    # ── Drop rows where NWS high is NaN ─────────────────────────────
    before = len(df)
    df = df.dropna(subset=["nws_high"]).reset_index(drop=True)
    log.info("Dropped %d rows with NaN NWS high (remaining: %d)", before - len(df), len(df))

    # ── Define continuous feature columns (75 total) ───────────────
    # Raw forecast model outputs (7: 6 Open-Meteo + 1 IEM GFS MOS)
    cont_cols = list(FORECAST_COLS)
    # Ensemble stats (4)
    cont_cols += ["forecast_mean", "forecast_std", "forecast_range", "forecast_iqr"]
    # Forecast disagreement structure (4)
    cont_cols += ["forecast_median", "forecast_median_mean_gap", "forecast_skew", "forecast_kurtosis"]
    # Pairwise spreads (5)
    cont_cols += ["spread_ecmwf_gfs", "spread_icon_gfs", "spread_gem_gfs", "spread_jma_gfs", "spread_hrrr_gfs"]
    # Forecast momentum (4)
    cont_cols += ["forecast_mean_delta1", "forecast_std_delta1", "forecast_mean_roll3_trend", "forecast_spread_momentum"]
    # Climate indices (4)
    cont_cols += ["enso_oni", "ao", "nao", "pna"]
    # Lagged meteorological lag-1 (5)
    cont_cols += meteo_cols_lag1
    # Lagged meteorological lag-2 (5)
    cont_cols += meteo_cols_lag2
    # Moisture / humidity derived (3)
    cont_cols += ["dewpoint_depression_lag1", "precip_x_cloud_lag1", "snow_flag_lag1"]
    # Pressure change / frontal (3)
    cont_cols += ["pressure_tendency_lag1", "wind_direction_lag1", "wind_x_pressure_change_lag1"]
    # Lagged hourly path (5)
    cont_cols += [f"{c}_lag1" for c in ["temp_6am", "temp_9am", "temp_noon", "temp_3pm", "temp_path_range"]]
    # Yesterday's overnight low (1)
    cont_cols += ["temperature_2m_min_lag1"]
    # Rolling bias mean (4)
    cont_cols += ["y_resid_roll3_mean", "y_resid_roll7_mean", "y_resid_roll14_mean", "y_resid_roll30_mean"]
    # Rolling bias std (3)
    cont_cols += ["y_resid_roll7_std", "y_resid_roll14_std", "y_resid_roll30_std"]
    # Climatology anomalies (3)
    cont_cols += ["clim_normal", "forecast_clim_anomaly", "yesterday_high_clim_anomaly"]
    # City static (6)
    cont_cols += ["lat", "lon", "elevation_ft", "coastal", "desert", "continentality"]
    # Calendar (4)
    cont_cols += ["sin_doy", "cos_doy", "sin_month", "cos_month"]
    # Solar / astronomical (3)
    cont_cols += ["day_length_hours", "solar_declination", "days_since_winter_solstice"]
    # Cross-feature interactions (4)
    cont_cols += ["forecast_mean_x_std", "spread_range_x_cloud", "elevation_x_pressure", "coastal_x_wind"]
    # Wind direction encoding (2)
    cont_cols += ["wind_dir_sin_lag1", "wind_dir_cos_lag1"]
    # Onshore wind (2)
    cont_cols += ["onshore_wind_component", "wind_speed_x_onshore"]
    # Humidity/moisture (2)
    cont_cols += ["relative_humidity_proxy_lag1", "high_dewpoint_flag"]
    # Temperature regime (3)
    cont_cols += ["temp_volatility_7d", "temp_range_3d", "max_temp_yesterday_rank"]
    # Forecast bias regime (1)
    cont_cols += ["forecast_cold_bias_regime"]
    # Cloud/precip (2)
    cont_cols += ["cloud_cover_trend_2d", "precip_probability_proxy"]
    # Rates of change (2)
    cont_cols += ["overnight_cooling_rate", "morning_warming_rate"]
    # Pressure dynamics (1)
    cont_cols += ["pressure_acceleration"]
    # Wind direction change (1)
    cont_cols += ["wind_direction_change_lag12"]
    # Interaction/seasonal (1)
    cont_cols += ["continentality_x_season"]
    # Forecast ensemble bimodality (1)
    cont_cols += ["forecast_ensemble_bimodality"]
    # Seasonal model skill (1)
    cont_cols += ["seasonal_model_skill"]
    # Normalized anomaly (1)
    cont_cols += ["forecast_mean_vs_clim_extreme"]

    assert len(cont_cols) == 96, f"Expected 96 continuous features, got {len(cont_cols)}"

    # Fill remaining NaNs in continuous features with 0
    df[cont_cols] = df[cont_cols].fillna(0.0)

    log.info("Total continuous features: %d", len(cont_cols))

    # ── Split ───────────────────────────────────────────────────────
    train, val, test = feature_utils.split_data(df)
    log.info("Split sizes — train: %d, val: %d, test: %d", len(train), len(val), len(test))

    return train, val, test, cont_cols


# ── Training ────────────────────────────────────────────────────────

def train():
    """Build features, train model, evaluate on test set."""
    train_df, val_df, test_df, cont_cols = build_features()

    # Save unscaled values needed for evaluation before scaling
    train_forecast_mean_raw = train_df["forecast_mean"].values.copy()
    train_actual_temp_raw = train_df["nws_high"].values.copy()
    val_forecast_mean_raw = val_df["forecast_mean"].values.copy()
    val_actual_temp_raw = val_df["nws_high"].values.copy()
    test_forecast_mean_raw = test_df["forecast_mean"].values.copy()
    test_actual_temp_raw = test_df["nws_high"].values.copy()

    # ── Scale continuous features (fit on train only) ───────────────
    scaler = feature_utils.ScalerWrapper()
    train_df = scaler.fit_transform(train_df, cont_cols)
    val_df = scaler.transform(val_df)
    test_df = scaler.transform(test_df)

    # Save scaler for later inference
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, "model1_scaler.pkl")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    scaler.save(scaler_path)
    log.info("Scaler saved to %s", scaler_path)

    # ── Create TensorDatasets ───────────────────────────────────────
    train_ds = training.make_dataset(
        train_df[cont_cols].values,
        train_df["city_idx"].values,
        train_df["y_resid"].values,
    )
    val_ds = training.make_dataset(
        val_df[cont_cols].values,
        val_df["city_idx"].values,
        val_df["y_resid"].values,
    )
    test_ds = training.make_dataset(
        test_df[cont_cols].values,
        test_df["city_idx"].values,
        test_df["y_resid"].values,
    )

    hp = cfg.MODEL1_HP
    train_loader = training.make_loader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = training.make_loader(val_ds, batch_size=hp["batch_size"], shuffle=False)
    test_loader = training.make_loader(test_ds, batch_size=hp["batch_size"], shuffle=False)

    # ── Build model ─────────────────────────────────────────────────
    model = training.TemperatureMLP(
        n_continuous=len(cont_cols),
        n_cities=cfg.N_CITIES,
        city_embed_dim=hp["city_embed_dim"],
        hidden_dims=hp["hidden_dims"],
        dropout=hp["dropout"],
    )
    log.info("Model architecture:\n%s", model)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Total parameters: %d", n_params)

    # ── Train ───────────────────────────────────────────────────────
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "model1_best.pt")
    history = training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hp=hp,
        checkpoint_path=checkpoint_path,
    )

    # ── Evaluate on test set ────────────────────────────────────────
    log.info("=== Test set evaluation ===")
    mu_resid, sigma_resid = training.predict(model, test_loader)

    # Convert residual predictions back to actual temperature
    test_df = test_df.copy()
    test_df["pred_resid"] = mu_resid
    test_df["pred_sigma"] = sigma_resid
    test_df["pred_temp"] = test_forecast_mean_raw + mu_resid
    test_df["actual_temp"] = test_actual_temp_raw

    # Overall metrics
    overall = evaluation.compute_metrics(test_df["actual_temp"].values, test_df["pred_temp"].values)
    log.info("Overall test metrics:")
    for k, v in overall.items():
        log.info("  %s: %.4f", k, v)

    # Per-city metrics
    city_metrics = evaluation.metrics_by_city(test_df, "actual_temp", "pred_temp")
    log.info("Per-city test MAE:\n%s", city_metrics[["city", "mae", "rmse", "bias"]].to_string(index=False))

    # Calibration check on residual
    cal = evaluation.calibration_check(test_df["y_resid"].values, mu_resid, sigma_resid)
    log.info("Calibration:\n%s", cal.to_string(index=False))

    # Residual metrics (what the model directly predicts)
    resid_metrics = evaluation.compute_metrics(test_df["y_resid"].values, mu_resid)
    log.info("Residual prediction metrics:")
    for k, v in resid_metrics.items():
        log.info("  %s: %.4f", k, v)

    # ── Save train + val + test predictions ──────────────────────────
    train_loader_eval = training.make_loader(
        training.make_dataset(train_df[cont_cols].values, train_df["city_idx"].values, train_df["y_resid"].values),
        batch_size=cfg.MODEL1_HP["batch_size"], shuffle=False)
    mu_train_r, sigma_train_r = training.predict(model, train_loader_eval)
    val_loader_eval = training.make_loader(
        training.make_dataset(val_df[cont_cols].values, val_df["city_idx"].values, val_df["y_resid"].values),
        batch_size=cfg.MODEL1_HP["batch_size"], shuffle=False)
    mu_val_r, sigma_val_r = training.predict(model, val_loader_eval)
    for split_name, split_df, mu_r, sigma_r, fcst_raw, actual_raw in [
        ("train", train_df, mu_train_r, sigma_train_r, train_forecast_mean_raw, train_actual_temp_raw),
        ("val", val_df, mu_val_r, sigma_val_r, val_forecast_mean_raw, val_actual_temp_raw),
        ("test", test_df, mu_resid, sigma_resid, test_forecast_mean_raw, test_actual_temp_raw),
    ]:
        pred_df = pd.DataFrame({
            "date": split_df["date"].values,
            "ticker": split_df["ticker"].values,
            "mu": fcst_raw + mu_r,  # convert residual back to temp
            "sigma": sigma_r,
            "y_true": actual_raw,
        })
        path = os.path.join(cfg.CHECKPOINT_DIR, f"model1_preds_{split_name}.parquet")
        pred_df.to_parquet(path, index=False)
        log.info("Saved %s predictions to %s", split_name, path)

    return model, history, test_df


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
