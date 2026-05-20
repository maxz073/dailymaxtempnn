"""
Kalshi Temperature Event Market Backtesting.

Backtests two trading strategies on simulated Kalshi daily high temperature
contracts using our bias-correction neural network vs NWP-derived market prices.

Strategy A: Hold to expiry — buy mispriced contracts, settle at end of day.
Strategy B: Trade on convergence — exit when edge narrows, with 3× stop-loss.

Also runs the same strategies using GFS, HRRR, and ECMWF individually as
the "model" probability source for comparison.
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
THRESHOLDS = list(range(20, 125, 5))  # 5°F intervals from 20 to 120
DEFAULT_EDGE_THRESHOLD = 0.10
DEFAULT_EXIT_THRESHOLD = 0.03
DEFAULT_STOP_LOSS_MULT = 3.0
DEFAULT_CONVERGENCE_RATE = 0.6

# Kalshi taker fee: lesser of 7 cents/contract or 7% of potential profit.
# Potential profit = max payout (100 cents) minus price paid.
KALSHI_FEE_RATE = 0.07
KALSHI_FEE_CAP_CENTS = 7.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "kalshi")


# ── Probability functions ───────────────────────────────────────────────

def model_prob_above(mu: float, sigma: float, threshold: float) -> float:
    """P(T > threshold) from model Gaussian."""
    return float(norm.sf(threshold, loc=mu, scale=sigma))


def nwp_discrete_prob(nwp_values: np.ndarray, threshold: float) -> float:
    """Fraction of NWP models forecasting above threshold."""
    valid = nwp_values[~np.isnan(nwp_values)]
    if len(valid) == 0:
        return np.nan
    return float((valid > threshold).mean())


def nwp_smoothed_prob(nwp_values: np.ndarray, threshold: float) -> float:
    """P(T > threshold) from Gaussian fit to NWP ensemble."""
    valid = nwp_values[~np.isnan(nwp_values)]
    if len(valid) < 2:
        return nwp_discrete_prob(nwp_values, threshold)
    mu = valid.mean()
    sigma = max(valid.std(ddof=1), 0.5)  # floor at 0.5°F
    return float(norm.sf(threshold, loc=mu, scale=sigma))


def single_nwp_prob(forecast: float, historical_std: float,
                    threshold: float) -> float:
    """P(T > threshold) from single NWP model + its historical error std."""
    return float(norm.sf(threshold, loc=forecast, scale=historical_std))


# ── Contract generation ─────────────────────────────────────────────────

def generate_contracts(df: pd.DataFrame, thresholds: list[int] = None,
                       use_smoothed_market: bool = True) -> pd.DataFrame:
    """Generate all city-day-threshold contracts with model and market probs.

    Only generates contracts where the threshold is within a reasonable
    range of the forecast (±25°F of model mu) to avoid trivial 0/1 contracts.
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    records = []
    for _, row in df.iterrows():
        mu, sigma = row["mu"], row["sigma"]
        nwp_vals = row[NWP_COLS].values.astype(float)
        y_true = row["y_true"]

        for thr in thresholds:
            if abs(mu - thr) > 25:
                continue

            m_prob = model_prob_above(mu, sigma, thr)
            if use_smoothed_market:
                mkt_prob = nwp_smoothed_prob(nwp_vals, thr)
            else:
                mkt_prob = nwp_discrete_prob(nwp_vals, thr)

            if np.isnan(mkt_prob):
                continue

            records.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "threshold": thr,
                "model_prob": m_prob,
                "market_prob": mkt_prob,
                "edge": m_prob - mkt_prob,
                "settlement": 1.0 if y_true > thr else 0.0,
                "y_true": y_true,
                "mu": mu,
                "sigma": sigma,
            })

    return pd.DataFrame(records)


def generate_single_nwp_contracts(
    df: pd.DataFrame,
    nwp_col: str,
    historical_std: float,
    thresholds: list[int] = None,
    use_smoothed_market: bool = True,
) -> pd.DataFrame:
    """Generate contracts using a single NWP model as the probability source."""
    if thresholds is None:
        thresholds = THRESHOLDS

    records = []
    for _, row in df.iterrows():
        fcst = row[nwp_col]
        if np.isnan(fcst):
            continue
        nwp_vals = row[NWP_COLS].values.astype(float)
        y_true = row["y_true"]

        for thr in thresholds:
            if abs(fcst - thr) > 25:
                continue

            m_prob = single_nwp_prob(fcst, historical_std, thr)
            if use_smoothed_market:
                mkt_prob = nwp_smoothed_prob(nwp_vals, thr)
            else:
                mkt_prob = nwp_discrete_prob(nwp_vals, thr)

            if np.isnan(mkt_prob):
                continue

            records.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "threshold": thr,
                "model_prob": m_prob,
                "market_prob": mkt_prob,
                "edge": m_prob - mkt_prob,
                "settlement": 1.0 if y_true > thr else 0.0,
            })

    return pd.DataFrame(records)


# ── Fee calculation ─────────────────────────────────────────────────────

def kalshi_taker_fee_cents(entry_price_cents: np.ndarray) -> np.ndarray:
    """Kalshi taker fee per contract in cents.

    Fee = min(7% of potential profit, 7 cents).
    Potential profit = 100 - entry_price (max payout minus price paid).
    Charged on every trade at entry.
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

    # Direction: buy YES when model_prob > market_prob, buy NO otherwise
    trades["direction"] = np.where(trades["edge"] > 0, "YES", "NO")

    # Entry price in cents (0-100)
    trades["entry_price"] = np.where(
        trades["direction"] == "YES",
        trades["market_prob"] * 100,  # buy YES at market price
        (1 - trades["market_prob"]) * 100,  # buy NO at 100 - market
    )

    # Settlement value per contract
    trades["settle_value"] = np.where(
        trades["direction"] == "YES",
        trades["settlement"] * 100,
        (1 - trades["settlement"]) * 100,
    )

    # Taker fee (charged at entry on every trade)
    trades["fee_per_contract"] = kalshi_taker_fee_cents(trades["entry_price"].values)

    # P&L per contract (cents), then scale by position size
    trades["pnl_per_contract_gross"] = trades["settle_value"] - trades["entry_price"]
    trades["pnl_per_contract"] = trades["pnl_per_contract_gross"] - trades["fee_per_contract"]
    trades["fee_dollars"] = trades["fee_per_contract"] * CONTRACTS_PER_TRADE / 100
    trades["pnl_dollars"] = (
        trades["pnl_per_contract"] * CONTRACTS_PER_TRADE / 100
    )

    return trades


# ── Strategy B: Convergence + Stop-Loss ─────────────────────────────────

def backtest_convergence_trade(
    contracts: pd.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
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

    # Simulate market convergence: market moves toward model_prob
    # converged_market_prob = market_prob + convergence_rate * (model_prob - market_prob)
    converged_market = (
        trades["market_prob"]
        + convergence_rate * (trades["model_prob"] - trades["market_prob"])
    )

    # Exit price if we exit on convergence
    exit_price_converge = np.where(
        trades["direction"] == "YES",
        converged_market * 100,
        (1 - converged_market) * 100,
    )

    # Stop-loss: market moves against us by 3× the entry spread
    # For YES: market drops by 3× edge (bad for us)
    # For NO: market rises by 3× edge (bad for us)
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

    # Determine exit type: convergence or stop-loss
    # Stop-loss triggers if the "worst case" market movement exceeds 3× spread.
    # We model this probabilistically: the chance of stop-loss triggering is
    # proportional to how close the model_prob and actual settlement diverge.
    # Simplified: stop-loss triggers if settlement goes against our direction
    # AND the edge was small.
    #
    # For the backtest, we check: did the actual outcome validate the trade?
    # If yes → convergence exit. If no → check if stop-loss would have saved us.

    trade_correct = np.where(
        trades["direction"] == "YES",
        trades["settlement"] == 1.0,  # we bet YES and it settled YES
        trades["settlement"] == 0.0,  # we bet NO and it settled NO1
    )

    # Convergence exit: we capture the convergence move
    # Stop-loss exit: we lose 3× the entry edge (capped)
    trades["exit_type"] = np.where(trade_correct, "convergence", "stop_loss")

    trades["exit_price"] = np.where(
        trade_correct,
        exit_price_converge,
        stop_loss_price,
    )

    # Taker fee (charged at entry on every trade)
    trades["fee_per_contract"] = kalshi_taker_fee_cents(trades["entry_price"].values)

    trades["pnl_per_contract_gross"] = trades["exit_price"] - trades["entry_price"]
    trades["pnl_per_contract"] = trades["pnl_per_contract_gross"] - trades["fee_per_contract"]
    trades["fee_dollars"] = trades["fee_per_contract"] * CONTRACTS_PER_TRADE / 100
    trades["pnl_dollars"] = (
        trades["pnl_per_contract"] * CONTRACTS_PER_TRADE / 100
    )

    return trades


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    """Compute performance metrics for a set of trades."""
    if trades.empty:
        return {"label": label, "n_trades": 0}

    daily_pnl = trades.groupby("date")["pnl_dollars"].sum()
    cumulative = daily_pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    return {
        "label": label,
        "n_trades": len(trades),
        "total_pnl": trades["pnl_dollars"].sum(),
        "avg_pnl_per_trade": trades["pnl_dollars"].mean(),
        "win_rate": (trades["pnl_dollars"] > 0).mean(),
        "sharpe": (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                   if daily_pnl.std() > 0 else 0),
        "max_drawdown": drawdown.min(),
        "trading_days": len(daily_pnl),
    }


def metrics_by_city(trades: pd.DataFrame) -> pd.DataFrame:
    """P&L breakdown by city."""
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby("ticker")
        .agg(
            n_trades=("pnl_dollars", "count"),
            total_pnl=("pnl_dollars", "sum"),
            win_rate=("pnl_dollars", lambda x: (x > 0).mean()),
        )
        .sort_values("total_pnl", ascending=False)
    )


def metrics_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    """P&L breakdown by month."""
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
    """Run both strategies across multiple edge thresholds."""
    if edge_thresholds is None:
        edge_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    rows = []
    for et in edge_thresholds:
        for strat_name, fn in [
            ("A_hold", backtest_hold_to_expiry),
            ("B_convergence", backtest_convergence_trade),
        ]:
            kwargs = {"edge_threshold": et}
            trades = fn(contracts, **kwargs)
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
):
    """Generate and save all Kalshi plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    ax.set_title("Cumulative P&L: Strategy A vs B (NN Model)")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_pnl_ab.png"), dpi=150)
    plt.close(fig)

    # 2. Cumulative P&L: NN vs individual NWP models (Strategy A)
    fig, ax = plt.subplots(figsize=(12, 5))
    if not trades_a.empty:
        daily_nn = trades_a.groupby("date")["pnl_dollars"].sum().sort_index()
        ax.plot(daily_nn.index, daily_nn.cumsum(), label="NN Ensemble",
                lw=2.5, color="blue")
    for model_name, res in nwp_results.items():
        t = res.get("trades_a")
        if t is not None and not t.empty:
            daily_m = t.groupby("date")["pnl_dollars"].sum().sort_index()
            ax.plot(daily_m.index, daily_m.cumsum(), label=model_name,
                    lw=1.5, alpha=0.7)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title("Cumulative P&L: NN vs Individual NWP Models (Strategy A)")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_pnl_nn_vs_nwp.png"), dpi=150)
    plt.close(fig)

    # 3. P&L by city (Strategy A)
    fig, ax = plt.subplots(figsize=(10, 7))
    city_pnl = metrics_by_city(trades_a)
    if not city_pnl.empty:
        city_names = [cfg.CITIES[t][0] for t in city_pnl.index]
        colors = ["green" if v >= 0 else "red" for v in city_pnl["total_pnl"]]
        ax.barh(city_names, city_pnl["total_pnl"], color=colors)
        ax.set_xlabel("Total P&L ($)")
        ax.set_title("P&L by City (Strategy A: Hold to Expiry)")
        ax.xaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pnl_by_city.png"), dpi=150)
    plt.close(fig)

    # 4. P&L by month
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly = metrics_by_month(trades_a)
    if not monthly.empty:
        colors = ["green" if v >= 0 else "red" for v in monthly["total_pnl"]]
        ax.bar(range(len(monthly)), monthly["total_pnl"], color=colors)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right")
        ax.set_ylabel("P&L ($)")
        ax.set_title("Monthly P&L (Strategy A: Hold to Expiry)")
        ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pnl_by_month.png"), dpi=150)
    plt.close(fig)

    # 5. Sharpe & win rate vs edge threshold
    if not sensitivity_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for strat in ["A_hold", "B_convergence"]:
            s = sensitivity_df[sensitivity_df["strategy"] == strat]
            ax1.plot(s["edge_threshold"], s["sharpe"], marker="o", label=strat)
            ax2.plot(s["edge_threshold"], s["win_rate"], marker="o", label=strat)
        ax1.set_xlabel("Edge Threshold")
        ax1.set_ylabel("Annualized Sharpe Ratio")
        ax1.set_title("Sharpe vs Edge Threshold")
        ax1.legend()
        ax2.set_xlabel("Edge Threshold")
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Win Rate vs Edge Threshold")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "sensitivity.png"), dpi=150)
        plt.close(fig)

    # 6. Stop-loss analysis (Strategy B)
    if not trades_b.empty and "exit_type" in trades_b.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        exit_counts = trades_b["exit_type"].value_counts()
        ax.bar(exit_counts.index, exit_counts.values, color=["green", "red"])
        ax.set_ylabel("Number of Trades")
        ax.set_title("Strategy B: Exit Type Distribution")
        for i, (label, val) in enumerate(exit_counts.items()):
            ax.text(i, val + 5, str(val), ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "stop_loss_frequency.png"), dpi=150)
        plt.close(fig)


def print_summary(
    trades_a: pd.DataFrame,
    trades_b: pd.DataFrame,
    nwp_results: dict[str, dict],
):
    """Print summary statistics."""
    print("=" * 70)
    print("  KALSHI TEMPERATURE MARKET BACKTEST (with taker fees)")
    print("=" * 70)
    print(f"  Contracts per trade: {CONTRACTS_PER_TRADE}")
    print(f"  Edge threshold: {DEFAULT_EDGE_THRESHOLD}")
    print(f"  Taker fee: min({KALSHI_FEE_RATE:.0%} of potential profit, {KALSHI_FEE_CAP_CENTS:.0f}¢/contract)")

    for label, trades in [
        ("Strategy A: Hold to Expiry", trades_a),
        ("Strategy B: Convergence + Stop-Loss", trades_b),
    ]:
        m = compute_metrics(trades, label)
        total_fees = trades["fee_dollars"].sum() if not trades.empty and "fee_dollars" in trades.columns else 0
        print(f"\n  --- {label} ---")
        print(f"  Total trades:     {m.get('n_trades', 0):,}")
        print(f"  Gross P&L:        ${m.get('total_pnl', 0) + total_fees:>12,.2f}")
        print(f"  Total fees:       ${total_fees:>12,.2f}")
        print(f"  Net P&L:          ${m.get('total_pnl', 0):>12,.2f}")
        print(f"  Avg P&L/trade:    ${m.get('avg_pnl_per_trade', 0):>10,.2f}")
        print(f"  Win rate:         {m.get('win_rate', 0):.1%}")
        print(f"  Sharpe ratio:     {m.get('sharpe', 0):.2f}")
        print(f"  Max drawdown:     ${m.get('max_drawdown', 0):>10,.2f}")

        if not trades.empty and "exit_type" in trades.columns:
            stop_count = (trades["exit_type"] == "stop_loss").sum()
            print(f"  Stop-loss trades: {stop_count} ({stop_count/len(trades)*100:.1f}%)")

    print(f"\n  --- NWP Baseline Comparison (Strategy A) ---")
    nn_pnl = trades_a["pnl_dollars"].sum() if not trades_a.empty else 0
    print(f"  {'Model':<20} {'Trades':>8} {'Total P&L':>14} {'Win Rate':>10} {'Sharpe':>8}")
    print(f"  {'NN Ensemble':<20} {len(trades_a):>8,} ${nn_pnl:>12,.2f} "
          f"{(trades_a['pnl_dollars'] > 0).mean() if not trades_a.empty else 0:>9.1%} "
          f"{compute_metrics(trades_a).get('sharpe', 0):>8.2f}")
    for model_name, res in nwp_results.items():
        t = res.get("trades_a", pd.DataFrame())
        m = res.get("metrics_a", {})
        pnl = m.get("total_pnl", 0)
        wr = m.get("win_rate", 0)
        sh = m.get("sharpe", 0)
        print(f"  {model_name:<20} {m.get('n_trades', 0):>8,} ${pnl:>12,.2f} {wr:>9.1%} {sh:>8.2f}")

    print("=" * 70)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_test_with_nwp()

    print("Generating contracts (NN model)...")
    contracts = generate_contracts(df)
    print(f"  {len(contracts):,} contracts generated")

    print("Running Strategy A (hold to expiry)...")
    trades_a = backtest_hold_to_expiry(contracts)
    print(f"  {len(trades_a):,} trades executed")

    print("Running Strategy B (convergence + stop-loss)...")
    trades_b = backtest_convergence_trade(contracts)
    print(f"  {len(trades_b):,} trades executed")

    print("Running sensitivity analysis...")
    sensitivity_df = run_sensitivity(contracts)

    # NWP baseline comparisons
    print("Running NWP baseline comparisons...")
    nwp_models = {
        "GFS": ("fcst_gfs_global", None),
        "HRRR": ("fcst_ncep_hrrr_conus", None),
        "ECMWF": ("fcst_ecmwf_ifs025", None),
    }

    # Compute historical std for each NWP model (on test set, vs actuals)
    for name, (col, _) in nwp_models.items():
        residuals = df[col] - df["y_true"]
        nwp_models[name] = (col, residuals.std())
        print(f"  {name} historical std: {residuals.std():.2f}°F")

    nwp_results = {}
    for name, (col, hist_std) in nwp_models.items():
        print(f"  Backtesting {name}...")
        nwp_contracts = generate_single_nwp_contracts(df, col, hist_std)
        t_a = backtest_hold_to_expiry(nwp_contracts)
        t_b = backtest_convergence_trade(nwp_contracts)
        nwp_results[name] = {
            "trades_a": t_a,
            "trades_b": t_b,
            "metrics_a": compute_metrics(t_a, f"{name}_A"),
            "metrics_b": compute_metrics(t_b, f"{name}_B"),
        }

    print_summary(trades_a, trades_b, nwp_results)

    print(f"\nSaving plots to {OUTPUT_DIR}/")
    plot_all(trades_a, trades_b, sensitivity_df, nwp_results)

    # Save data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades_a.to_csv(os.path.join(OUTPUT_DIR, "trades_strategy_a.csv"), index=False)
    trades_b.to_csv(os.path.join(OUTPUT_DIR, "trades_strategy_b.csv"), index=False)
    sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
