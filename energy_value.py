
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import config as cfg
from economic_data_prep import load_test_with_nwp, NWP_COLS

NWP_DISPLAY_NAMES = {
    "fcst_gfs_global": "GFS",
    "fcst_ecmwf_ifs025": "ECMWF",
    "fcst_icon_seamless": "ICON",
    "fcst_gem_seamless": "GEM",
    "fcst_jma_seamless": "JMA",
    "fcst_ncep_hrrr_conus": "HRRR",
}

ERCOT_CITIES = {
    "KXHIGHTHOU":  0.35,
    "KXHIGHTDAL":  0.30,
    "KXHIGHTSATX": 0.20,
    "KXHIGHAUS":   0.15,
}

DEFAULT_SENSITIVITY_MW_PER_F = 500.0

DEFAULT_COOLING_THRESHOLD_F = 65.0

DEFAULT_DA_RT_SPREAD = 20.0

PEAK_HOURS = 8

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "energy")

def build_ercot_daily(df: pd.DataFrame) -> pd.DataFrame:
    ercot = df[df["ticker"].isin(ERCOT_CITIES)].copy()
    ercot["weight"] = ercot["ticker"].map(ERCOT_CITIES)

    def _agg(g):
        w = g["weight"].values
        w_sum = w.sum()
        result = {
            "nn_forecast": (g["mu"].values * w).sum() / w_sum,
            "nwp_forecast": (g["nwp_ensemble_mean"].values * w).sum() / w_sum,
            "actual": (g["y_true"].values * w).sum() / w_sum,
            "nn_sigma": np.sqrt(((g["sigma"].values ** 2) * (w ** 2)).sum()) / w_sum,
        }
        for col in NWP_COLS:
            if col in g.columns:
                result[col] = (g[col].values * w).sum() / w_sum
        return pd.Series(result)

    agg = (
        ercot.groupby("date")
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    agg["month"] = agg["date"].dt.month
    return agg

def compute_load_error(
    forecast: np.ndarray,
    actual: np.ndarray,
    sensitivity: float = DEFAULT_SENSITIVITY_MW_PER_F,
    threshold: float = DEFAULT_COOLING_THRESHOLD_F,
) -> np.ndarray:
    cooling_relevant = (forecast > threshold) | (actual > threshold)
    error_f = forecast - actual
    return np.where(cooling_relevant, error_f * sensitivity, 0.0)

def compute_daily_cost(
    load_error_mw: np.ndarray,
    da_rt_spread: float | np.ndarray = DEFAULT_DA_RT_SPREAD,
) -> np.ndarray:
    return np.abs(load_error_mw) * np.abs(da_rt_spread) * PEAK_HOURS

def load_real_ercot_prices() -> pd.DataFrame:
    path = os.path.join(cfg.DATA_DIR, "ercot_prices.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Real ERCOT price data not found at {path}. "
            "Run `python fetch_ercot_prices.py` first."
        )
    prices = pd.read_csv(path, parse_dates=["date"])
    return prices

def run_ercot_analysis(
    df: pd.DataFrame,
    sensitivity: float = DEFAULT_SENSITIVITY_MW_PER_F,
    threshold: float = DEFAULT_COOLING_THRESHOLD_F,
) -> pd.DataFrame:
    daily = build_ercot_daily(df)

    daily["nn_load_error_mw"] = compute_load_error(
        daily["nn_forecast"].values, daily["actual"].values,
        sensitivity, threshold,
    )
    daily["nwp_load_error_mw"] = compute_load_error(
        daily["nwp_forecast"].values, daily["actual"].values,
        sensitivity, threshold,
    )

    for col in NWP_COLS:
        if col in daily.columns:
            daily[f"{col}_load_error_mw"] = compute_load_error(
                daily[col].values, daily["actual"].values,
                sensitivity, threshold,
            )

    prices = load_real_ercot_prices()
    daily = daily.merge(
        prices[["date", "da_rt_spread", "dam_price_avg", "rtm_price_avg"]],
        on="date", how="left",
    )
    spread = daily["da_rt_spread"].values

    daily["nn_cost"] = compute_daily_cost(daily["nn_load_error_mw"].values, spread)
    daily["nwp_cost"] = compute_daily_cost(daily["nwp_load_error_mw"].values, spread)
    daily["daily_savings"] = daily["nwp_cost"] - daily["nn_cost"]
    daily["cumulative_savings"] = daily["daily_savings"].cumsum()

    for col in NWP_COLS:
        if col in daily.columns:
            daily[f"{col}_cost"] = compute_daily_cost(
                daily[f"{col}_load_error_mw"].values, spread,
            )

    return daily

def sensitivity_analysis(
    df: pd.DataFrame,
    sensitivities: list[float] = None,
) -> pd.DataFrame:
    if sensitivities is None:
        sensitivities = [300, 400, 500, 600, 700]

    rows = []
    for sens in sensitivities:
        result = run_ercot_analysis(df, sensitivity=sens)
        annual = result["daily_savings"].sum()
        rows.append({"sensitivity_mw_per_f": sens, "annual_savings": annual})
    return pd.DataFrame(rows)

def _dollar_fmt(x, _):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"

def plot_all(daily: pd.DataFrame, sens_df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["cumulative_savings"], color="green", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title("Cumulative Procurement Cost Savings: NN vs NWP Ensemble Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Savings ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_savings.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["nn_load_error_mw"].abs(), alpha=0.5,
            label="NN |load error|", color="blue", lw=0.8)
    ax.plot(daily["date"], daily["nwp_load_error_mw"].abs(), alpha=0.5,
            label="NWP |load error|", color="red", lw=0.8)
    ax.set_title("Daily |Load Forecast Error| (MW): NN vs NWP")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "daily_load_error.png"), dpi=150)
    plt.close(fig)

    monthly = daily.groupby(daily["date"].dt.to_period("M"))["daily_savings"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if v >= 0 else "red" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right")
    ax.set_title("Monthly Procurement Savings")
    ax.set_ylabel("Savings ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "monthly_savings.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [f"{int(s)} MW/°F" for s in sens_df["sensitivity_mw_per_f"]],
        sens_df["annual_savings"],
        color="green",
    )
    for i, val in enumerate(sens_df["annual_savings"]):
        ax.text(i, val + sens_df["annual_savings"].max() * 0.02,
                f"${val / 1e6:.1f}M", ha="center", fontsize=10)
    ax.set_xlabel("Temperature-Load Sensitivity")
    ax.set_ylabel("Annual Savings ($)")
    ax.set_title("Annual Savings vs Load Sensitivity (Real ERCOT Prices)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sensitivity.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    cooling_days = daily[daily["actual"] > DEFAULT_COOLING_THRESHOLD_F]
    ax.scatter(cooling_days["actual"], cooling_days["daily_savings"],
               alpha=0.4, s=15, color="green")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Actual Temperature (°F)")
    ax.set_ylabel("Daily Savings ($)")
    ax.set_title("Daily Savings vs Actual Temperature (Cooling Days Only)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "savings_vs_temp.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(daily["nn_load_error_mw"], bins=50, alpha=0.6, label="NN", color="blue")
    ax.hist(daily["nwp_load_error_mw"], bins=50, alpha=0.6, label="NWP", color="red")
    ax.set_xlabel("Load Forecast Error (MW)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Daily Load Forecast Errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "error_distributions.png"), dpi=150)
    plt.close(fig)

def print_summary(daily: pd.DataFrame, sens_df: pd.DataFrame):
    total_savings = daily["daily_savings"].sum()
    nn_total_cost = daily["nn_cost"].sum()
    nwp_total_cost = daily["nwp_cost"].sum()
    nn_mae = np.abs(daily["nn_forecast"] - daily["actual"]).mean()
    nwp_mae = np.abs(daily["nwp_forecast"] - daily["actual"]).mean()
    cooling_days = (daily["actual"] > DEFAULT_COOLING_THRESHOLD_F).sum()
    positive_days = (daily["daily_savings"] > 0).sum()
    total_days = len(daily)

    print("=" * 65)
    print("  ERCOT ENERGY PROCUREMENT VALUE ANALYSIS (Real Prices)")
    print("=" * 65)
    print(f"\n  Test period:   {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Total days:    {total_days}")
    print(f"  Cooling days:  {cooling_days} (actual > {DEFAULT_COOLING_THRESHOLD_F}°F)")
    print(f"  Price source:  ERCOT HB_HOUSTON peak-hour (HE 13-20) settlement prices")

    if "da_rt_spread" in daily.columns:
        spread_vals = daily["da_rt_spread"]
        print(f"\n  DA-RT spread stats:")
        print(f"    Mean:   ${spread_vals.mean():>8.2f}/MWh")
        print(f"    Median: ${spread_vals.median():>8.2f}/MWh")
        print(f"    Std:    ${spread_vals.std():>8.2f}/MWh")
        print(f"    Range:  ${spread_vals.min():>8.2f} to ${spread_vals.max():.2f}/MWh")

    print(f"\n  NN temperature MAE:  {nn_mae:.2f}°F")
    print(f"  NWP temperature MAE: {nwp_mae:.2f}°F")
    print(f"  MAE improvement:     {nwp_mae - nn_mae:.2f}°F ({(nwp_mae - nn_mae) / nwp_mae * 100:.1f}%)")
    print(f"\n  NN annual procurement cost:  ${nn_total_cost:>12,.0f}")
    print(f"  NWP annual procurement cost: ${nwp_total_cost:>12,.0f}")
    print(f"  Annual savings (NN vs NWP):  ${total_savings:>12,.0f}")
    print(f"\n  Days with positive savings:  {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)")
    print(f"  Avg daily savings:           ${total_savings / total_days:>10,.0f}")

    summer = daily[(daily["month"] >= 6) & (daily["month"] <= 9)]
    winter = daily[~((daily["month"] >= 6) & (daily["month"] <= 9))]
    print(f"\n  Summer (Jun-Sep) savings:    ${summer['daily_savings'].sum():>12,.0f}")
    print(f"  Non-summer savings:          ${winter['daily_savings'].sum():>12,.0f}")

    print(f"\n  Sensitivity analysis (real prices, varying MW/°F):")
    for _, row in sens_df.iterrows():
        marker = " <-- base case" if row["sensitivity_mw_per_f"] == 500 else ""
        print(f"    {int(row['sensitivity_mw_per_f']):>4} MW/°F: ${row['annual_savings']:>12,.0f}{marker}")
    print("=" * 65)

def main():
    print("Loading data...")
    df = load_test_with_nwp()

    print("Running ERCOT analysis (real ERCOT prices)...")
    daily = run_ercot_analysis(df)

    print("Running sensitivity analysis...")
    sens_df = sensitivity_analysis(df)

    print_summary(daily, sens_df)

    print(f"\nSaving plots to {OUTPUT_DIR}/")
    plot_all(daily, sens_df)

    daily.to_csv(os.path.join(OUTPUT_DIR, "ercot_daily_results.csv"), index=False)
    sens_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_results.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    main()
