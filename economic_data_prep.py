"""
Shared data preparation for economic value analyses.

Joins model test predictions with raw NWP forecasts so both the Kalshi
backtest and ERCOT energy analysis can work from a single DataFrame.
"""

import pandas as pd
import config as cfg
from data_fetch import load_forecasts

NWP_COLS = [
    "fcst_gfs_global",
    "fcst_ecmwf_ifs025",
    "fcst_icon_seamless",
    "fcst_gem_seamless",
    "fcst_jma_seamless",
    "fcst_ncep_hrrr_conus",
]


def load_test_with_nwp() -> pd.DataFrame:
    """Load model test predictions joined with raw NWP forecasts.

    Returns DataFrame with columns:
        date, ticker, mu, sigma, y_true,
        fcst_gfs_global, fcst_ecmwf_ifs025, fcst_icon_seamless,
        fcst_gem_seamless, fcst_jma_seamless, fcst_ncep_hrrr_conus,
        nwp_ensemble_mean
    """
    import os

    preds = pd.read_parquet(
        os.path.join(cfg.CHECKPOINT_DIR, "model1_preds_test.parquet")
    )
    preds["date"] = pd.to_datetime(preds["date"])

    fcst = load_forecasts()
    fcst["date"] = pd.to_datetime(fcst["date"])
    fcst = fcst[fcst["date"] >= pd.Timestamp(cfg.TEST_START)]

    merged = preds.merge(fcst, on=["date", "ticker"], how="left")
    merged["nwp_ensemble_mean"] = merged[NWP_COLS].mean(axis=1)

    return merged


if __name__ == "__main__":
    df = load_test_with_nwp()
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"NWP NaN counts:\n{df[NWP_COLS].isna().sum()}")
    print(df.head())
