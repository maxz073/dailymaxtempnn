"""
Kalshi Temperature Event Market Backtesting (Real Prices).

Uses real Kalshi contract prices from data/kalshi_prices.csv.
Compares our NN model probability against actual market last_price
to identify mispriced contracts.

Strategy A: Hold to expiry — buy mispriced contracts, settle at end of day.
Strategy B: Trade on convergence — exit when edge narrows, with 3× stop-loss.

Also runs the same strategies using GFS, HRRR, and ECMWF individually.

Two analysis scopes:
  1. Austin full-year (364 days, all of test period)
  2. All 4 ERCOT cities on overlapping period (~64 days, Feb-Apr 2026)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter

import config as cfg
from economic_data_prep import load_test_with_nwp, NWP_COLS

# ── Constants ───────────────────────────────────────────────────────────

CONTRACTS_PER_TRADE = 250
DEFAULT_EDGE_THRESHOLD = 0.10
DEFAULT_STOP_LOSS_MULT = 3.0
DEFAULT_CONVERGENCE_RATE = 0.6

# Kalshi taker fee: lesser of 7 cents/contract or 7% of potential profit.
KALSHI_FEE_RATE = 0.07
KALSHI_FEE_CAP_CENTS = 7.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "kalshi")


# ── Load real Kalshi data ───────────────────────────────────────────────

def load_kalshi_prices() -> pd.DataFrame:
    """Load real Kalshi contract data."""
    path = os.path.join(cfg.DATA_DIR, "kalshi_prices.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Kalshi price data not found at {path}. "
            "Run `python fetch_kalshi_prices.py` first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Probability functions ───────────────────────────────────────────────

def model_prob_for_contract(
    mu: float, sigma: float,
    strike_type: str, floor_strike: float, cap_strike: float,
) -> float:
    """Compute model P(YES) for a Kalshi contract given NN (mu, sigma).

    - greater: P(T > floor_strike)
    - less:    P(T < cap_strike)
    - between: P(floor_strike < T < cap_strike)
    """
    if strike_type == "greater":
        return float(norm.sf(floor_strike, loc=mu, scale=sigma))
    elif strike_type == "less":
        return float(norm.cdf(cap_strike, loc=mu, scale=sigma))
    elif strike_type == "between":
        return float(
            norm.cdf(cap_strike, loc=mu, scale=sigma)
            - norm.cdf(floor_strike, loc=mu, scale=sigma)
        )
    return np.nan


def nwp_prob_for_contract(
    forecast: float, hist_std: float,
    strike_type: str, floor_strike: float, cap_strike: float,
) -> float:
    """Compute P(YES) for a contract using a single NWP model."""
    return model_prob_for_contract(forecast, hist_std,
                                   strike_type, floor_strike, cap_strike)


# ── Build contracts with model probabilities ────────────────────────────

def build_contracts(
    kalshi: pd.DataFrame,
    preds: pd.DataFrame,
) -> pd.DataFrame:
    """Join real Kalshi contracts with NN model probabilities.

    Returns one row per contract with:
      market_prob (real last_price), model_prob (NN), edge, settlement.
    """
    # Merge on date + series_ticker = ticker
    merged = kalshi.merge(
        preds[["date", "ticker", "mu", "sigma"]],
        left_on=["date", "series_ticker"],
        right_on=["date", "ticker"],
        how="inner",
    )

    # Compute model probability for each contract
    merged["model_prob"] = merged.apply(
        lambda r: model_prob_for_contract(
            r["mu"], r["sigma"],
            r["strike_type"], r["floor_strike"], r["cap_strike"],
        ),
        axis=1,
    )

    # Market probability = last traded price (0-1)
    merged["market_prob"] = merged["last_price"].astype(float)

    # Edge
    merged["edge"] = merged["model_prob"] - merged["market_prob"]

    # Settlement (already 0 or 1 in the data)
    merged["settlement"] = merged["settlement_value"].astype(float)

    return merged


def build_nwp_contracts(
    kalshi: pd.DataFrame,
    preds: pd.DataFrame,
    nwp_col: str,
    hist_std: float,
) -> pd.DataFrame:
    """Build contracts using a single NWP model as probability source."""
    merged = kalshi.merge(
        preds[["date", "ticker", nwp_col]],
        left_on=["date", "series_ticker"],
        right_on=["date", "ticker"],
        how="inner",
    )

    merged["model_prob"] = merged.apply(
        lambda r: nwp_prob_for_contract(
            r[nwp_col], hist_std,
            r["strike_type"], r["floor_strike"], r["cap_strike"],
        ),
        axis=1,
    )
    merged["market_prob"] = merged["last_price"].astype(float)
    merged["edge"] = merged["model_prob"] - merged["market_prob"]
    merged["settlement"] = merged["settlement_value"].astype(float)

    return merged


# ── Fee calculation ─────────────────────────────────────────────────────

def kalshi_taker_fee_cents(entry_price_cents: np.ndarray) -> np.ndarray:
    """Kalshi taker fee per contract in cents.

    Fee = min(7% of potential profit, 7 cents).
    Potential profit = 100 - entry_price.
    """
    potential_profit = 100.0 - entry_price_cents
    pct_fee = KALSHI_FEE_RATE * potential_profit
    return np.minimum(pct_fee, KALSHI_FEE_CAP_CENTS)


# ── Strategy A: Hold to Expiry ──────────────────────────────────────────

def backtest_hold_to_expiry(
    contracts: pd.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Buy mispriced contracts, hold to settlement."""
    trades = contracts[contracts["edge"].abs() > edge_threshold].copy()
    if trades.empty:
        return trades

    trades["direction"] = np.where(trades["edge"] > 0, "YES", "NO")

    trades["entry_price"] = np.where(
        trades["direction"] == "YES",
        trades["market_prob"] * 100,
        (1 - trades["market_prob"]) * 100,
    )

    trades["settle_value"] = np.where(
        trades["direction"] == "YES",
        trades["settlement"] * 100,
        (1 - trades["settlement"]) * 100,
    )

    trades["fee_per_contract"] = kalshi_taker_fee_cents(trades["entry_price"].values)
    trades["pnl_per_contract_gross"] = trades["settle_value"] - trades["entry_price"]
    trades["pnl_per_contract"] = trades["pnl_per_contract_gross"] - trades["fee_per_contract"]
    trades["fee_dollars"] = trades["fee_per_contract"] * CONTRACTS_PER_TRADE / 100
    trades["pnl_dollars"] = trades["pnl_per_contract"] * CONTRACTS_PER_TRADE / 100

    return trades


# ── Strategy B: Convergence + Stop-Loss ─────────────────────────────────

def backtest_convergence_trade(
    contracts: pd.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    stop_loss_mult: float = DEFAULT_STOP_LOSS_MULT,
    convergence_rate: float = DEFAULT_CONVERGENCE_RATE,
) -> pd.DataFrame:
    """Trade on edge, exit when market converges or stop-loss triggers."""
    trades = contracts[contracts["edge"].abs() > edge_threshold].copy()
    if trades.empty:
        return trades

    trades["direction"] = np.where(trades["edge"] > 0, "YES", "NO")
    trades["entry_price"] = np.where(
        trades["direction"] == "YES",
        trades["market_prob"] * 100,
        (1 - trades["market_prob"]) * 100,
    )
    trades["entry_edge"] = trades["edge"].abs()

    converged_market = (
        trades["market_prob"]
        + convergence_rate * (trades["model_prob"] - trades["market_prob"])
    )

    exit_price_converge = np.where(
        trades["direction"] == "YES",
        converged_market * 100,
        (1 - converged_market) * 100,
    )

    stop_loss_market = np.where(
        trades["direction"] == "YES",
        trades["market_prob"] - stop_loss_mult * trades["entry_edge"],
        trades["market_prob"] + stop_loss_mult * trades["entry_edge"],
    )
    stop_loss_market = np.clip(stop_loss_market, 0, 1)

    stop_loss_price = np.where(
        trades["direction"] == "YES",
        stop_loss_market * 100,
        (1 - stop_loss_market) * 100,
    )

    trade_correct = np.where(
        trades["direction"] == "YES",
        trades["settlement"] == 1.0,
        trades["settlement"] == 0.0,
    )

    trades["exit_type"] = np.where(trade_correct, "convergence", "stop_loss")
    trades["exit_price"] = np.where(trade_correct, exit_price_converge, stop_loss_price)

    trades["fee_per_contract"] = kalshi_taker_fee_cents(trades["entry_price"].values)
    trades["pnl_per_contract_gross"] = trades["exit_price"] - trades["entry_price"]
    trades["pnl_per_contract"] = trades["pnl_per_contract_gross"] - trades["fee_per_contract"]
    trades["fee_dollars"] = trades["fee_per_contract"] * CONTRACTS_PER_TRADE / 100
    trades["pnl_dollars"] = trades["pnl_per_contract"] * CONTRACTS_PER_TRADE / 100

    return trades


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if trades.empty:
        return {"label": label, "n_trades": 0, "total_pnl": 0,
                "avg_pnl_per_trade": 0, "win_rate": 0, "sharpe": 0,
                "max_drawdown": 0, "trading_days": 0}

    daily_pnl = trades.groupby("date")["pnl_dollars"].sum()
    cumulative = daily_pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()

    return {
        "label": label,
        "n_trades": len(trades),
        "total_pnl": trades["pnl_dollars"].sum(),
        "total_fees": trades["fee_dollars"].sum() if "fee_dollars" in trades.columns else 0,
        "avg_pnl_per_trade": trades["pnl_dollars"].mean(),
        "win_rate": (trades["pnl_dollars"] > 0).mean(),
        "sharpe": (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                   if daily_pnl.std() > 0 else 0),
        "max_drawdown": drawdown.min(),
        "trading_days": len(daily_pnl),
    }


def metrics_by_city(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    col = "series_ticker" if "series_ticker" in trades.columns else "ticker"
    return (
        trades.groupby(col)
        .agg(
            n_trades=("pnl_dollars", "count"),
            total_pnl=("pnl_dollars", "sum"),
            win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
        )
        .sort_values("total_pnl", ascending=False)
    )


def metrics_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["month"] = trades["date"].dt.to_period("M")
    return (
        trades.groupby("month")
        .agg(
            n_trades=("pnl_dollars", "count"),
            total_pnl=("pnl_dollars", "sum"),
            win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
        )
    )


# ── Sensitivity ─────────────────────────────────────────────────────────

def run_sensitivity(
    contracts: pd.DataFrame,
    edge_thresholds: list[float] = None,
) -> pd.DataFrame:
    if edge_thresholds is None:
        edge_thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    rows = []
    for et in edge_thresholds:
        for strat_name, fn in [
            ("A_hold", backtest_hold_to_expiry),
            ("B_convergence", backtest_convergence_trade),
        ]:
            trades = fn(contracts, edge_threshold=et)
            m = compute_metrics(trades, f"{strat_name}_et{et}")
            m["edge_threshold"] = et
            m["strategy"] = strat_name
            rows.append(m)

    return pd.DataFrame(rows)


# ── Visualization ───────────────────────────────────────────────────────

def _dollar_fmt(x, _):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def plot_all(
    trades_a: pd.DataFrame,
    trades_b: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    nwp_results: dict[str, dict],
    label_suffix: str = "",
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sfx = f"_{label_suffix}" if label_suffix else ""

    # 1. Cumulative P&L: Strategy A vs B
    fig, ax = plt.subplots(figsize=(12, 5))
    for trades, label, color in [
        (trades_a, "A: Hold to Expiry", "blue"),
        (trades_b, "B: Convergence + Stop-Loss", "green"),
    ]:
        if trades.empty:
            continue
        daily = trades.groupby("date")["pnl_dollars"].sum().sort_index()
        ax.plot(daily.index, daily.cumsum(), label=label, lw=2, color=color)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(f"Cumulative P&L: Strategy A vs B (Real Kalshi Prices{', ' + label_suffix if label_suffix else ''})")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"cumulative_pnl_ab{sfx}.png"), dpi=150)
    plt.close(fig)

    # 2. NN vs NWP (Strategy A)
    fig, ax = plt.subplots(figsize=(12, 5))
    if not trades_a.empty:
        daily_nn = trades_a.groupby("date")["pnl_dollars"].sum().sort_index()
        ax.plot(daily_nn.index, daily_nn.cumsum(), label="NN Ensemble", lw=2.5, color="blue")
    for model_name, res in nwp_results.items():
        t = res.get("trades_a")
        if t is not None and not t.empty:
            daily_m = t.groupby("date")["pnl_dollars"].sum().sort_index()
            ax.plot(daily_m.index, daily_m.cumsum(), label=model_name, lw=1.5, alpha=0.7)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(f"NN vs NWP Models — Strategy A{' (' + label_suffix + ')' if label_suffix else ''}")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"cumulative_pnl_nn_vs_nwp{sfx}.png"), dpi=150)
    plt.close(fig)

    # 3. P&L by month
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly = metrics_by_month(trades_a)
    if not monthly.empty:
        colors = ["green" if v >= 0 else "red" for v in monthly["total_pnl"]]
        ax.bar(range(len(monthly)), monthly["total_pnl"], color=colors)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right")
        ax.set_ylabel("P&L ($)")
        ax.set_title(f"Monthly P&L — Strategy A{' (' + label_suffix + ')' if label_suffix else ''}")
        ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"pnl_by_month{sfx}.png"), dpi=150)
    plt.close(fig)

    # 4. Sensitivity
    if not sensitivity_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for strat in ["A_hold", "B_convergence"]:
            s = sensitivity_df[sensitivity_df["strategy"] == strat]
            ax1.plot(s["edge_threshold"], s["sharpe"], marker="o", label=strat)
            ax2.plot(s["edge_threshold"], s["win_rate"], marker="o", label=strat)
        ax1.set_xlabel("Edge Threshold"); ax1.set_ylabel("Sharpe"); ax1.set_title("Sharpe vs Edge"); ax1.legend()
        ax2.set_xlabel("Edge Threshold"); ax2.set_ylabel("Win Rate"); ax2.set_title("Win Rate vs Edge"); ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"sensitivity{sfx}.png"), dpi=150)
        plt.close(fig)


# ── Print summary ───────────────────────────────────────────────────────

def print_run_summary(
    title: str,
    contracts: pd.DataFrame,
    trades_a: pd.DataFrame,
    trades_b: pd.DataFrame,
    nwp_results: dict[str, dict],
):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  Contracts per trade: {CONTRACTS_PER_TRADE}")
    print(f"  Edge threshold: {DEFAULT_EDGE_THRESHOLD}")
    print(f"  Taker fee: min({KALSHI_FEE_RATE:.0%} of potential profit, {KALSHI_FEE_CAP_CENTS:.0f}¢/contract)")
    print(f"  Total real contracts: {len(contracts):,}")

    if not contracts.empty:
        cities = contracts["series_ticker"].unique()
        dates = contracts["date"]
        print(f"  Cities: {', '.join(sorted(cities))}")
        print(f"  Date range: {dates.min().date()} to {dates.max().date()} ({dates.dt.date.nunique()} days)")

    for label, trades in [
        ("Strategy A: Hold to Expiry", trades_a),
        ("Strategy B: Convergence + Stop-Loss", trades_b),
    ]:
        m = compute_metrics(trades, label)
        print(f"\n  --- {label} ---")
        print(f"  Total trades:     {m['n_trades']:,}")
        print(f"  Gross P&L:        ${m['total_pnl'] + m['total_fees']:>12,.2f}")
        print(f"  Total fees:       ${m['total_fees']:>12,.2f}")
        print(f"  Net P&L:          ${m['total_pnl']:>12,.2f}")
        print(f"  Avg P&L/trade:    ${m['avg_pnl_per_trade']:>10,.2f}")
        print(f"  Win rate:         {m['win_rate']:.1%}")
        print(f"  Sharpe ratio:     {m['sharpe']:.2f}")
        print(f"  Max drawdown:     ${m['max_drawdown']:>10,.2f}")

        if not trades.empty and "exit_type" in trades.columns:
            stop_count = (trades["exit_type"] == "stop_loss").sum()
            print(f"  Stop-loss trades: {stop_count} ({stop_count/len(trades)*100:.1f}%)")

    if nwp_results:
        print(f"\n  --- NWP Baseline Comparison (Strategy A) ---")
        print(f"  {'Model':<20} {'Trades':>8} {'Net P&L':>14} {'Win Rate':>10} {'Sharpe':>8}")
        m_nn = compute_metrics(trades_a)
        print(f"  {'NN Ensemble':<20} {m_nn['n_trades']:>8,} ${m_nn['total_pnl']:>12,.2f} "
              f"{m_nn['win_rate']:>9.1%} {m_nn['sharpe']:>8.2f}")
        for name, res in nwp_results.items():
            m = res["metrics_a"]
            print(f"  {name:<20} {m['n_trades']:>8,} ${m['total_pnl']:>12,.2f} "
                  f"{m['win_rate']:>9.1%} {m['sharpe']:>8.2f}")

    print("=" * 70)


# ── Run a single analysis scope ─────────────────────────────────────────

def run_scope(
    title: str,
    kalshi: pd.DataFrame,
    preds: pd.DataFrame,
    label_suffix: str = "",
):
    """Run full backtest for a given scope (city/date subset)."""
    print(f"\n{'─' * 70}")
    print(f"  Scope: {title}")
    print(f"{'─' * 70}")

    print("  Building contracts (NN)...")
    contracts = build_contracts(kalshi, preds)
    print(f"    {len(contracts):,} contracts with model probabilities")

    print("  Running Strategy A...")
    trades_a = backtest_hold_to_expiry(contracts)
    print(f"    {len(trades_a):,} trades")

    print("  Running Strategy B...")
    trades_b = backtest_convergence_trade(contracts)
    print(f"    {len(trades_b):,} trades")

    print("  Sensitivity analysis...")
    sens_df = run_sensitivity(contracts)

    # NWP baselines
    print("  NWP baselines...")
    nwp_results = {}
    for name, col in [("GFS", "fcst_gfs_global"), ("HRRR", "fcst_ncep_hrrr_conus"), ("ECMWF", "fcst_ecmwf_ifs025")]:
        residuals = preds[col] - preds["y_true"]
        hist_std = residuals.std()
        nwp_contracts = build_nwp_contracts(kalshi, preds, col, hist_std)
        t_a = backtest_hold_to_expiry(nwp_contracts)
        t_b = backtest_convergence_trade(nwp_contracts)
        nwp_results[name] = {
            "trades_a": t_a, "trades_b": t_b,
            "metrics_a": compute_metrics(t_a, f"{name}_A"),
            "metrics_b": compute_metrics(t_b, f"{name}_B"),
        }

    print_run_summary(title, contracts, trades_a, trades_b, nwp_results)

    print(f"  Saving plots ({label_suffix})...")
    plot_all(trades_a, trades_b, sens_df, nwp_results, label_suffix)

    # Save CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sfx = f"_{label_suffix}" if label_suffix else ""
    trades_a.to_csv(os.path.join(OUTPUT_DIR, f"trades_a{sfx}.csv"), index=False)
    trades_b.to_csv(os.path.join(OUTPUT_DIR, f"trades_b{sfx}.csv"), index=False)
    sens_df.to_csv(os.path.join(OUTPUT_DIR, f"sensitivity{sfx}.csv"), index=False)

    return contracts, trades_a, trades_b, sens_df, nwp_results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    preds = load_test_with_nwp()
    kalshi = load_kalshi_prices()

    print(f"Kalshi contracts loaded: {len(kalshi):,}")
    print(f"Cities: {sorted(kalshi['series_ticker'].unique())}")

    # Scope 1: Austin full year
    kalshi_aus = kalshi[kalshi["series_ticker"] == "KXHIGHAUS"]
    preds_aus = preds[preds["ticker"] == "KXHIGHAUS"]
    run_scope(
        "Austin Full Year (364 days, real Kalshi prices)",
        kalshi_aus, preds_aus, label_suffix="austin",
    )

    # Scope 2: All 4 ERCOT cities, overlapping period only
    ercot_tickers = ["KXHIGHAUS", "KXHIGHTDAL", "KXHIGHTHOU", "KXHIGHTSATX"]
    # Find overlapping dates (all 4 cities have data)
    date_counts = kalshi[kalshi["series_ticker"].isin(ercot_tickers)].groupby("date")["series_ticker"].nunique()
    overlap_dates = date_counts[date_counts == 4].index

    if len(overlap_dates) > 0:
        kalshi_4 = kalshi[
            kalshi["series_ticker"].isin(ercot_tickers) &
            kalshi["date"].isin(overlap_dates)
        ]
        preds_4 = preds[
            preds["ticker"].isin(ercot_tickers) &
            preds["date"].isin(overlap_dates)
        ]
        run_scope(
            f"4 ERCOT Cities ({len(overlap_dates)} overlapping days, real Kalshi prices)",
            kalshi_4, preds_4, label_suffix="ercot4",
        )
    else:
        print("\nNo overlapping dates found for all 4 ERCOT cities.")

    print("\nDone.")


if __name__ == "__main__":
    main()
