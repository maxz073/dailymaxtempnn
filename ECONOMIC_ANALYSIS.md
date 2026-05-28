# ERCOT Energy Procurement Value Analysis

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
| `energy_value.py` | ERCOT energy procurement value analysis |
| `outputs/energy/` | Energy plots, daily results, sensitivity CSVs |
