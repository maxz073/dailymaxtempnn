# Economic Value Analysis

This document describes two quantitative analyses that demonstrate the economic
value of the bias-correction neural network for daily maximum temperature
prediction.

---

## Analysis 1: Kalshi Temperature Event Markets

### Overview

[Kalshi](https://kalshi.com) operates regulated event contracts on daily high
temperatures in US cities.  A contract pays $1 if the recorded high exceeds a
stated threshold (e.g., "NYC high > 85 °F") and $0 otherwise.  The market
price reflects the crowd-implied probability of that event.

Our neural network outputs a calibrated Gaussian (μ, σ) for each city-day,
which gives us `P(T > threshold)` for any threshold via the normal CDF.
When our probability diverges from the market-implied probability, we trade.

### Decision changed

A trader using our model identifies mispriced contracts — those whose market
price understates or overstates the probability of exceeding a temperature
threshold — and takes a position.

### Data

| Source | Description |
|--------|-------------|
| `model1_preds_test.parquet` | NN predictions (μ, σ) for 20 cities, Apr 2025 – Apr 2026 |
| NWP forecast CSVs | 6 raw NWP model forecasts per city-day (GFS, ECMWF, ICON, GEM, JMA, HRRR) |

**Market price proxy:** We do not have historical Kalshi prices, so we construct
a synthetic market price by fitting a Gaussian to the 6 NWP forecasts
(mean ± std) and computing `P(T > threshold)`.  This represents what a
naive forecaster using raw NWP output would price.

### Strategies

**Strategy A — Hold to Expiry:**
Enter when `|model_prob − market_prob| > 0.10`.  Hold until settlement
at end of day.  Captures the full edge but bears settlement risk.

**Strategy B — Convergence + Stop-Loss:**
Same entry rule.  Exit when the simulated market price converges toward
the model probability (convergence rate = 60%).  If the market moves
against us by 3× the entry spread, cut the position (stop-loss).

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Position size | 250 contracts/trade | Fixed; small enough for price-taking |
| Entry threshold | 10% edge | Filters noise; ensures meaningful mispricing |
| Contract thresholds | 5 °F intervals, 20–120 °F | Matches Kalshi granularity |
| Convergence rate | 60% | Market moves ~60% toward true prob by settlement |
| Stop-loss | 3× entry spread | Caps downside at 3× the initial edge |

### Assumptions

1. **Price-taking** — our 250-contract positions do not move the market.
2. **NWP-derived market price** — the smoothed Gaussian fit to 6 NWP models
   is a reasonable proxy for less-sophisticated market participants.
3. **No transaction fees** — Kalshi currently charges no maker fees.
4. **Execution at mid-market** — no bid-ask spread friction modeled.
5. **No information leakage** — model uses only data available before the
   trading day (lagged features, prior-day forecasts).

### Baseline comparison

We run the same Strategy A using GFS, HRRR, and ECMWF individually as the
"model" probability source (point forecast ± historical residual std) to
show that the NN's edge is not replicable with a single NWP model.

### How to run

```bash
python kalshi_backtest.py
```

Outputs are saved to `outputs/kalshi/`.

---

## Analysis 2: ERCOT Energy Procurement Value

### Overview

Electricity grid operators purchase power in the **day-ahead (DA)** market
based on forecast demand.  When the forecast is wrong, they must buy or sell
the difference in the **real-time (RT)** market at a premium.  Since peak
electricity demand is heavily driven by air-conditioning load, which in turn
depends on daily maximum temperature, a more accurate temperature forecast
directly reduces procurement costs.

### Decision changed

A utility serving the ERCOT (Texas) grid uses our model instead of the raw
NWP ensemble mean to forecast tomorrow's peak temperature.  This changes
the quantity of electricity purchased in the day-ahead market.

### Data

| Source | Description |
|--------|-------------|
| `model1_preds_test.parquet` | NN predictions for 4 ERCOT cities |
| NWP forecast CSVs | NWP ensemble mean as baseline forecast |

**ERCOT cities (population-weighted):**

| City | Weight | Rationale |
|------|--------|-----------|
| Houston | 0.35 | Largest metro, highest load share |
| Dallas | 0.30 | Second-largest metro |
| San Antonio | 0.20 | Third-largest |
| Austin | 0.15 | Smallest of the four |

### Methodology

1. Population-weight the city forecasts to get an ERCOT-wide temperature.
2. For days where temperature > 65 °F (cooling threshold):
   - `load_error = (forecast − actual) × sensitivity_MW_per_F`
   - `cost = |load_error| × DA_RT_spread × peak_hours`
3. Compute savings = NWP cost − NN cost.

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Temperature-load sensitivity | 500 MW/°F | ERCOT CDR reports; Considine (2000); Sailor & Muñoz (1997) |
| Cooling threshold | 65 °F | Standard cooling degree day base |
| DA-RT spread (seasonal) | $40/MWh Jun–Sep, $10 rest | ERCOT historical averages |
| Peak hours | 8/day (HE 13–20) | Standard ERCOT peak window |

### Assumptions

1. **Linear temperature-demand relationship** above the cooling threshold.
   Real demand curves are slightly nonlinear, but the linear approximation
   is standard in the literature and holds well for small perturbations.
2. **Price-taking utility** — the utility's procurement change does not
   affect wholesale prices.
3. **Only daily max temperature** — real load forecasting uses hourly
   profiles, humidity, etc.  Our analysis isolates the temperature channel.
4. **Population-weighted city average ≈ ERCOT-wide temperature** — a
   simplification; ERCOT spans all of Texas.
5. **No capacity market or ancillary service value** — only energy
   procurement savings are counted.  The total value including reserves,
   frequency regulation, and avoided demand response is likely higher.

### Sensitivity analysis

A heatmap of annual savings is computed across:
- Temperature-load sensitivity: 300–700 MW/°F
- DA-RT spread: $10–$50/MWh

### How to run

```bash
python energy_value.py
```

Outputs are saved to `outputs/energy/`.

---

## File inventory

| File | Purpose |
|------|---------|
| `economic_data_prep.py` | Shared data loading (joins predictions with NWP forecasts) |
| `kalshi_backtest.py` | Kalshi market backtest (Strategies A & B, NWP baselines) |
| `energy_value.py` | ERCOT energy procurement value analysis |
| `outputs/kalshi/` | Kalshi plots, trade logs, sensitivity CSVs |
| `outputs/energy/` | Energy plots, daily results, sensitivity CSVs |
