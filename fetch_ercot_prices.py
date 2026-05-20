"""
Fetch and process real ERCOT Day-Ahead and Real-Time Settlement Point Prices.

Data source: ERCOT MIS (Market Information System) — public, no API key required.
  - DAM SPPs: reportTypeId 13060 (DAMLZHBSPP annual zip files)
  - RTM SPPs: reportTypeId 13061 (RTMLZHBSPP annual zip files)

Downloads annual zip files containing Excel workbooks, then extracts
hourly DAM and 15-minute RTM prices for a chosen settlement point
(default: HB_HOUSTON). Computes daily averages over peak hours
(HE 13-20, i.e., noon-8pm CPT) and the DA-RT spread.

Output: data/ercot_prices.csv with columns:
  date, dam_price_avg, rtm_price_avg, da_rt_spread

Usage:
  python fetch_ercot_prices.py                    # download + process
  python fetch_ercot_prices.py --no-download      # process already-downloaded files
  python fetch_ercot_prices.py --settlement-point HB_HUBAVG  # use hub average
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Configuration ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ERCOT_DIR = DATA_DIR / "ercot_prices"
OUTPUT_CSV = DATA_DIR / "ercot_prices.csv"

# Test period from config.py
DATE_START = "2025-04-17"
DATE_END = "2026-04-16"

# ERCOT MIS download base URL (public, no auth required)
MIS_BASE = "https://www.ercot.com/misdownload/servlets/mirDownload"

# Annual file document lookup IDs (from ERCOT MIS public portal)
# DAM: reportTypeId=13060  (DAMLZHBSPP)
DAM_FILES = {
    2025: 1177667469,
    2026: 1227788397,
}

# RTM: reportTypeId=13061  (RTMLZHBSPP)
RTM_FILES = {
    2025: 1177737535,
    2026: 1227789873,
}

# Default settlement point — Houston hub is the most relevant for our
# ERCOT analysis since Houston is the largest load zone.
DEFAULT_SP = "HB_HOUSTON"

# Peak hours: Hour Ending 13 through 20 (noon to 8pm CPT).
# These are the hours when temperature drives cooling load.
PEAK_HOURS_START = 13
PEAK_HOURS_END = 20


# ── Download ─────────────────────────────────────────────────────────────

def download_ercot_file(doc_id: int, dest_path: Path, label: str = "") -> Path:
    """Download a single ERCOT MIS zip file."""
    url = f"{MIS_BASE}?mimic_duns=000000000&doclookupId={doc_id}"
    print(f"  Downloading {label} (docId={doc_id})...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)
    size_mb = len(resp.content) / 1e6
    print(f"    -> {dest_path.name} ({size_mb:.1f} MB)")
    return dest_path


def download_all():
    """Download DAM and RTM annual zip files for our test period years."""
    ERCOT_DIR.mkdir(parents=True, exist_ok=True)

    for year, doc_id in DAM_FILES.items():
        dest = ERCOT_DIR / f"DAMLZHBSPP_{year}.zip"
        if dest.exists():
            print(f"  Already have {dest.name}, skipping.")
        else:
            download_ercot_file(doc_id, dest, f"DAM {year}")

    for year, doc_id in RTM_FILES.items():
        dest = ERCOT_DIR / f"RTMLZHBSPP_{year}.zip"
        if dest.exists():
            print(f"  Already have {dest.name}, skipping.")
        else:
            download_ercot_file(doc_id, dest, f"RTM {year}")


# ── Parse ────────────────────────────────────────────────────────────────

def read_dam_excel(zip_path: Path, settlement_point: str) -> pd.DataFrame:
    """Read DAM SPP Excel from a zip file, filter to one settlement point.

    DAM columns: Delivery Date, Hour Ending, Repeated Hour Flag,
                 Settlement Point, Settlement Point Price
    """
    print(f"  Reading {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        xlsx_names = [n for n in zf.namelist() if n.endswith(".xlsx")]
        if not xlsx_names:
            raise FileNotFoundError(f"No xlsx in {zip_path}")
        with zf.open(xlsx_names[0]) as f:
            # Read ALL sheets (Jan-Dec) and concatenate
            sheets = pd.read_excel(
                io.BytesIO(f.read()), engine="openpyxl", sheet_name=None
            )

    frames = []
    for sheet_name, sdf in sheets.items():
        if sdf.empty:
            continue
        sdf.columns = [c.strip() for c in sdf.columns]
        frames.append(sdf)

    if not frames:
        raise ValueError(f"No data found in {zip_path}")

    df = pd.concat(frames, ignore_index=True)

    # Filter to our settlement point
    sp_col = "Settlement Point"
    df = df[df[sp_col] == settlement_point].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")

    # Parse hour ending — handle "24:00" as hour 24
    df["hour_ending"] = df["Hour Ending"].astype(str).str.replace(":00", "").astype(int)

    # Exclude repeated hours (DST fall-back)
    df = df[df["Repeated Hour Flag"] == "N"]

    df = df.rename(columns={"Settlement Point Price": "dam_price"})
    return df[["date", "hour_ending", "dam_price"]]


def read_rtm_excel(zip_path: Path, settlement_point: str) -> pd.DataFrame:
    """Read RTM SPP Excel from a zip file, filter to one settlement point.

    RTM columns: Delivery Date, Delivery Hour, Delivery Interval,
                 Repeated Hour Flag, Settlement Point Name,
                 Settlement Point Type, Settlement Point Price

    RTM has 4 intervals per hour (15-min). We average to hourly first.
    """
    print(f"  Reading {zip_path.name} (large file, may take a moment)...")
    with zipfile.ZipFile(zip_path) as zf:
        xlsx_names = [n for n in zf.namelist() if n.endswith(".xlsx")]
        if not xlsx_names:
            raise FileNotFoundError(f"No xlsx in {zip_path}")
        with zf.open(xlsx_names[0]) as f:
            # Read ALL sheets (Jan-Dec) and concatenate
            sheets = pd.read_excel(
                io.BytesIO(f.read()), engine="openpyxl", sheet_name=None
            )

    frames = []
    for sheet_name, sdf in sheets.items():
        if sdf.empty:
            continue
        sdf.columns = [c.strip() for c in sdf.columns]
        frames.append(sdf)

    if not frames:
        raise ValueError(f"No data found in {zip_path}")

    df = pd.concat(frames, ignore_index=True)

    # Filter to settlement point
    sp_col = "Settlement Point Name"
    df = df[df[sp_col] == settlement_point].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")
    df["hour_ending"] = df["Delivery Hour"].astype(int)

    # Exclude repeated hours (DST)
    df = df[df["Repeated Hour Flag"] == "N"]

    # Average the 4 intervals within each hour
    hourly = (
        df.groupby(["date", "hour_ending"])["Settlement Point Price"]
        .mean()
        .reset_index()
    )
    hourly = hourly.rename(columns={"Settlement Point Price": "rtm_price"})
    return hourly


# ── Aggregate ────────────────────────────────────────────────────────────

def compute_daily_spreads(
    dam: pd.DataFrame,
    rtm: pd.DataFrame,
    peak_start: int = PEAK_HOURS_START,
    peak_end: int = PEAK_HOURS_END,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> pd.DataFrame:
    """Compute daily peak-hour averages and DA-RT spread.

    Parameters
    ----------
    dam : DataFrame with columns [date, hour_ending, dam_price]
    rtm : DataFrame with columns [date, hour_ending, rtm_price]
    peak_start, peak_end : inclusive hour-ending range for peak hours
    date_start, date_end : filter to this date range
    """
    # Filter to peak hours
    dam_peak = dam[
        (dam["hour_ending"] >= peak_start) & (dam["hour_ending"] <= peak_end)
    ]
    rtm_peak = rtm[
        (rtm["hour_ending"] >= peak_start) & (rtm["hour_ending"] <= peak_end)
    ]

    # Daily averages
    dam_daily = (
        dam_peak.groupby("date")["dam_price"]
        .mean()
        .reset_index()
        .rename(columns={"dam_price": "dam_price_avg"})
    )
    rtm_daily = (
        rtm_peak.groupby("date")["rtm_price"]
        .mean()
        .reset_index()
        .rename(columns={"rtm_price": "rtm_price_avg"})
    )

    # Merge
    daily = dam_daily.merge(rtm_daily, on="date", how="inner")

    # DA-RT spread: positive means DA was more expensive than RT
    daily["da_rt_spread"] = daily["dam_price_avg"] - daily["rtm_price_avg"]

    # Filter to test period
    mask = (daily["date"] >= date_start) & (daily["date"] <= date_end)
    daily = daily[mask].sort_values("date").reset_index(drop=True)

    return daily


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch and process ERCOT DAM/RTM settlement point prices."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading; use already-downloaded zip files.",
    )
    parser.add_argument(
        "--settlement-point",
        default=DEFAULT_SP,
        help=f"Settlement point to extract (default: {DEFAULT_SP}). "
        "Options: HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST, HB_HUBAVG, "
        "LZ_HOUSTON, LZ_NORTH, etc.",
    )
    parser.add_argument(
        "--all-hours",
        action="store_true",
        help="Use all 24 hours instead of just peak hours (HE 13-20).",
    )
    args = parser.parse_args()

    sp = args.settlement_point
    print(f"ERCOT Price Data — Settlement Point: {sp}")
    print(f"Date range: {DATE_START} to {DATE_END}")
    print(f"Peak hours: HE {PEAK_HOURS_START}-{PEAK_HOURS_END}"
          + (" (using ALL hours)" if args.all_hours else ""))
    print()

    # Step 1: Download
    if not args.no_download:
        print("Step 1: Downloading ERCOT price files...")
        download_all()
    else:
        print("Step 1: Skipping download (--no-download).")
    print()

    # Step 2: Parse DAM files
    print("Step 2: Parsing DAM (Day-Ahead Market) prices...")
    dam_frames = []
    for year in sorted(DAM_FILES.keys()):
        zip_path = ERCOT_DIR / f"DAMLZHBSPP_{year}.zip"
        if not zip_path.exists():
            print(f"  WARNING: {zip_path.name} not found, skipping.")
            continue
        dam_frames.append(read_dam_excel(zip_path, sp))

    if not dam_frames:
        print("ERROR: No DAM data found. Run without --no-download first.")
        sys.exit(1)

    dam = pd.concat(dam_frames, ignore_index=True)
    print(f"  DAM records: {len(dam):,} hourly prices")
    print(f"  Date range: {dam['date'].min().date()} to {dam['date'].max().date()}")
    print()

    # Step 3: Parse RTM files
    print("Step 3: Parsing RTM (Real-Time Market) prices...")
    rtm_frames = []
    for year in sorted(RTM_FILES.keys()):
        zip_path = ERCOT_DIR / f"RTMLZHBSPP_{year}.zip"
        if not zip_path.exists():
            print(f"  WARNING: {zip_path.name} not found, skipping.")
            continue
        rtm_frames.append(read_rtm_excel(zip_path, sp))

    if not rtm_frames:
        print("ERROR: No RTM data found. Run without --no-download first.")
        sys.exit(1)

    rtm = pd.concat(rtm_frames, ignore_index=True)
    print(f"  RTM records: {len(rtm):,} hourly prices (averaged from 15-min)")
    print(f"  Date range: {rtm['date'].min().date()} to {rtm['date'].max().date()}")
    print()

    # Step 4: Compute daily spreads
    print("Step 4: Computing daily peak-hour DA-RT spreads...")
    peak_start = 1 if args.all_hours else PEAK_HOURS_START
    peak_end = 24 if args.all_hours else PEAK_HOURS_END

    daily = compute_daily_spreads(dam, rtm, peak_start, peak_end)
    print(f"  Days with complete data: {len(daily)}")
    print()

    # Step 5: Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_CSV, index=False)
    print(f"Step 5: Saved to {OUTPUT_CSV}")
    print()

    # Summary statistics
    print("=" * 65)
    print("  ERCOT PRICE DATA SUMMARY")
    print("=" * 65)
    print(f"  Settlement point:  {sp}")
    print(f"  Date range:        {daily['date'].min().date()} to "
          f"{daily['date'].max().date()}")
    print(f"  Total trading days: {len(daily)}")
    print()
    print(f"  DAM price (peak avg):  "
          f"mean=${daily['dam_price_avg'].mean():.2f}/MWh, "
          f"median=${daily['dam_price_avg'].median():.2f}/MWh")
    print(f"  RTM price (peak avg):  "
          f"mean=${daily['rtm_price_avg'].mean():.2f}/MWh, "
          f"median=${daily['rtm_price_avg'].median():.2f}/MWh")
    print()
    print(f"  DA-RT spread:          "
          f"mean=${daily['da_rt_spread'].mean():.2f}/MWh, "
          f"median=${daily['da_rt_spread'].median():.2f}/MWh")
    print(f"  DA-RT spread std dev:  ${daily['da_rt_spread'].std():.2f}/MWh")
    print(f"  DA-RT spread range:    "
          f"${daily['da_rt_spread'].min():.2f} to "
          f"${daily['da_rt_spread'].max():.2f}/MWh")
    print()

    # Monthly breakdown
    daily_copy = daily.copy()
    daily_copy["month"] = daily_copy["date"].dt.to_period("M")
    monthly = daily_copy.groupby("month").agg(
        dam_avg=("dam_price_avg", "mean"),
        rtm_avg=("rtm_price_avg", "mean"),
        spread_avg=("da_rt_spread", "mean"),
        spread_std=("da_rt_spread", "std"),
        days=("date", "count"),
    )
    print("  Monthly breakdown:")
    print(f"  {'Month':<10} {'DAM':>8} {'RTM':>8} {'Spread':>8} {'StdDev':>8} {'Days':>5}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for idx, row in monthly.iterrows():
        print(f"  {str(idx):<10} ${row['dam_avg']:>6.1f} ${row['rtm_avg']:>6.1f} "
              f"${row['spread_avg']:>6.1f} ${row['spread_std']:>6.1f} {int(row['days']):>5}")

    print("=" * 65)

    # Compare with our previous assumptions
    print()
    print("  Comparison with previous assumptions:")
    summer = daily_copy[daily_copy["date"].dt.month.isin([6, 7, 8, 9])]
    winter = daily_copy[~daily_copy["date"].dt.month.isin([6, 7, 8, 9])]
    if len(summer) > 0:
        print(f"    Summer (Jun-Sep) avg spread: ${summer['da_rt_spread'].mean():.2f}/MWh "
              f"(was assumed $40/MWh)")
    if len(winter) > 0:
        print(f"    Non-summer avg spread:       ${winter['da_rt_spread'].mean():.2f}/MWh "
              f"(was assumed $10/MWh)")
    print(f"    Overall avg spread:          ${daily['da_rt_spread'].mean():.2f}/MWh "
          f"(was assumed $20/MWh flat)")
    print()


if __name__ == "__main__":
    main()
