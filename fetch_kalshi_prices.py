"""
Fetch historical Kalshi temperature contract data for backtesting.

Pulls settlement prices, last-traded prices, and trade history for all 20
KXHIGH daily-max-temperature series across the test period (2025-04-17 to
2026-04-16).

All endpoints used are PUBLIC (no authentication required):
  - GET /historical/markets?event_ticker=...   (settled before 2026-03-20)
  - GET /markets?event_ticker=...              (settled after  2026-03-20)
  - GET /historical/trades?ticker=...          (trades before  2026-03-20)
  - GET /markets/trades?ticker=...             (trades after   2026-03-20)

Ticker format:
  Series:  KXHIGHNY
  Event:   KXHIGHNY-25APR17       (series-YYMMMDD)
  Market:  KXHIGHNY-25APR17-T67   (event-Tthreshold)

Usage:
    python fetch_kalshi_prices.py                  # fetch all cities, full period
    python fetch_kalshi_prices.py --city KXHIGHNY  # single city
    python fetch_kalshi_prices.py --start 2025-06-01 --end 2025-06-30
    python fetch_kalshi_prices.py --trades          # also fetch individual trades
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import date, timedelta, datetime
from typing import Optional

import requests

# ── Configuration ────────────────────────────────────────────────────────

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# Historical / live cutoff (from /historical/cutoff, checked 2026-05-19)
HISTORICAL_CUTOFF = date(2026, 3, 20)

TEST_START = date(2025, 4, 17)
TEST_END = date(2026, 4, 16)

# All 20 city series tickers
SERIES_TICKERS = [
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHTBOS", "KXHIGHLAX",
    "KXHIGHAUS", "KXHIGHTSFO", "KXHIGHTDAL", "KXHIGHPHIL", "KXHIGHTPHX",
    "KXHIGHTOKC", "KXHIGHDEN", "KXHIGHTDC", "KXHIGHTSATX", "KXHIGHTHOU",
    "KXHIGHTMIN", "KXHIGHTATL", "KXHIGHTSEA", "KXHIGHTLV", "KXHIGHTNOLA",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Rate-limit: Kalshi public API is generous but be polite
REQUEST_DELAY = 1.0  # seconds between requests


# ── Helpers ──────────────────────────────────────────────────────────────

MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def date_to_event_suffix(d: date) -> str:
    """Convert date to Kalshi event suffix: '25APR17'."""
    return f"{d.year % 100:02d}{MONTH_ABBR[d.month]}{d.day:02d}"


def event_ticker(series: str, d: date) -> str:
    """Build event ticker: KXHIGHNY-25APR17."""
    return f"{series}-{date_to_event_suffix(d)}"


def parse_market_date(event_tick: str) -> Optional[date]:
    """Extract the date from an event ticker like KXHIGHNY-25APR17."""
    suffix = event_tick.split("-")[-1]  # '25APR17'
    if len(suffix) < 6:
        return None
    try:
        yr = int(suffix[:2]) + 2000
        mon_str = suffix[2:5]
        day = int(suffix[5:])
        mon = {v: k for k, v in MONTH_ABBR.items()}[mon_str]
        return date(yr, mon, day)
    except (ValueError, KeyError):
        return None


def parse_threshold(ticker: str) -> Optional[float]:
    """Extract threshold from market ticker like KXHIGHNY-25APR17-T67."""
    parts = ticker.split("-T")
    if len(parts) < 2:
        # Try B-suffix (bracket markets): KXHIGHNY-25APR17-B63.5
        parts = ticker.split("-B")
        if len(parts) < 2:
            return None
    try:
        return float(parts[-1])
    except ValueError:
        return None


def api_get(endpoint: str, params: dict = None, max_retries: int = 3) -> dict:
    """Make a GET request to the Kalshi API with retries."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise RuntimeError(f"API request failed after {max_retries} tries: {e}")
    return {}


def paginate_markets(endpoint: str, params: dict) -> list[dict]:
    """Paginate through a markets endpoint, collecting all results."""
    all_markets = []
    cursor = None
    while True:
        p = {**params, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        data = api_get(endpoint, p)
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor", "")
        if not cursor or not markets:
            break
        time.sleep(REQUEST_DELAY)
    return all_markets


def paginate_trades(endpoint: str, params: dict) -> list[dict]:
    """Paginate through a trades endpoint, collecting all results."""
    all_trades = []
    cursor = None
    while True:
        p = {**params, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        data = api_get(endpoint, p)
        trades = data.get("trades", [])
        all_trades.extend(trades)
        cursor = data.get("cursor", "")
        if not cursor or not trades:
            break
        time.sleep(REQUEST_DELAY)
    return all_trades


# ── Core fetch functions ─────────────────────────────────────────────────

def fetch_markets_for_event(series: str, d: date) -> list[dict]:
    """Fetch all contract markets for a single city-day event.

    Routes to historical or live endpoint based on cutoff date.
    """
    ev = event_ticker(series, d)
    if d < HISTORICAL_CUTOFF:
        endpoint = "/historical/markets"
    else:
        endpoint = "/markets"
    return paginate_markets(endpoint, {"event_ticker": ev})


def fetch_trades_for_market(ticker: str, market_date: date) -> list[dict]:
    """Fetch all trades for a single market ticker."""
    if market_date < HISTORICAL_CUTOFF:
        endpoint = "/historical/trades"
    else:
        endpoint = "/markets/trades"
    return paginate_trades(endpoint, {"ticker": ticker})


def extract_market_row(m: dict) -> dict:
    """Extract a flat row from a market API response object."""
    ticker = m.get("ticker", "")
    event_tick = m.get("event_ticker", "")
    series = event_tick.rsplit("-", 1)[0] if "-" in event_tick else ""

    mdate = parse_market_date(event_tick)
    threshold = parse_threshold(ticker)

    # Determine strike direction
    strike_type = m.get("strike_type", "")
    floor_strike = m.get("floor_strike")
    cap_strike = m.get("cap_strike")

    return {
        "date": mdate.isoformat() if mdate else "",
        "series_ticker": series,
        "event_ticker": event_tick,
        "market_ticker": ticker,
        "strike_type": strike_type,       # "greater", "less", "between"
        "threshold": threshold,
        "floor_strike": floor_strike if floor_strike is not None else "",
        "cap_strike": cap_strike if cap_strike is not None else "",
        "last_price": m.get("last_price_dollars", ""),
        "yes_bid": m.get("yes_bid_dollars", ""),
        "yes_ask": m.get("yes_ask_dollars", ""),
        "no_bid": m.get("no_bid_dollars", ""),
        "no_ask": m.get("no_ask_dollars", ""),
        "volume": m.get("volume_fp", ""),
        "open_interest": m.get("open_interest_fp", ""),
        "result": m.get("result", ""),
        "settlement_value": m.get("settlement_value_dollars", ""),
        "expiration_value": m.get("expiration_value", ""),
        "status": m.get("status", ""),
    }


# ── Main pipeline ────────────────────────────────────────────────────────

def fetch_all_markets(
    series_list: list[str],
    start: date,
    end: date,
    progress: bool = True,
) -> list[dict]:
    """Fetch market data for all cities across the date range."""
    rows = []
    total_days = (end - start).days + 1

    for si, series in enumerate(series_list):
        if progress:
            print(f"\n[{si+1}/{len(series_list)}] {series}")

        d = start
        day_count = 0
        while d <= end:
            if progress and day_count % 30 == 0:
                print(f"  {d.isoformat()} ...", end="", flush=True)

            markets = fetch_markets_for_event(series, d)

            if markets:
                for m in markets:
                    rows.append(extract_market_row(m))
                if progress and day_count % 30 == 0:
                    print(f" {len(markets)} contracts")
            else:
                if progress and day_count % 30 == 0:
                    print(" (no contracts)")

            d += timedelta(days=1)
            day_count += 1
            time.sleep(REQUEST_DELAY)

    return rows


def fetch_all_trades(
    market_rows: list[dict],
    progress: bool = True,
) -> list[dict]:
    """Fetch trade-level data for all markets found."""
    trade_rows = []
    tickers = [(r["market_ticker"], r["date"]) for r in market_rows
               if r["market_ticker"] and r["volume"] and float(r["volume"]) > 0]

    if progress:
        print(f"\nFetching trades for {len(tickers)} markets with volume...")

    for i, (ticker, dstr) in enumerate(tickers):
        if progress and i % 50 == 0:
            print(f"  [{i+1}/{len(tickers)}] {ticker}")

        mdate = date.fromisoformat(dstr) if dstr else HISTORICAL_CUTOFF
        trades = fetch_trades_for_market(ticker, mdate)

        for t in trades:
            trade_rows.append({
                "market_ticker": t.get("ticker", ""),
                "trade_id": t.get("trade_id", ""),
                "created_time": t.get("created_time", ""),
                "yes_price": t.get("yes_price_dollars", ""),
                "no_price": t.get("no_price_dollars", ""),
                "count": t.get("count_fp", ""),
                "taker_side": t.get("taker_outcome_side", ""),
                "taker_book_side": t.get("taker_book_side", ""),
            })

        time.sleep(REQUEST_DELAY)

    return trade_rows


def save_csv(rows: list[dict], filepath: str):
    """Write rows to CSV."""
    if not rows:
        print(f"  No data to save to {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = rows[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {filepath}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi temperature contract data for backtesting."
    )
    parser.add_argument(
        "--city", type=str, default=None,
        help="Single series ticker (e.g. KXHIGHNY). Default: all 20 cities.",
    )
    parser.add_argument(
        "--start", type=str, default=TEST_START.isoformat(),
        help=f"Start date YYYY-MM-DD (default: {TEST_START})",
    )
    parser.add_argument(
        "--end", type=str, default=TEST_END.isoformat(),
        help=f"End date YYYY-MM-DD (default: {TEST_END})",
    )
    parser.add_argument(
        "--trades", action="store_true",
        help="Also fetch individual trade history (much slower).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    series_list = [args.city] if args.city else SERIES_TICKERS
    progress = not args.quiet

    # Validate city tickers
    for s in series_list:
        if s not in SERIES_TICKERS:
            print(f"Warning: {s} not in known tickers. Proceeding anyway.")

    print(f"Kalshi Temperature Data Fetch")
    print(f"  Cities: {len(series_list)}")
    print(f"  Period: {start} to {end} ({(end - start).days + 1} days)")
    print(f"  Historical cutoff: {HISTORICAL_CUTOFF}")
    print(f"  Trades: {'yes' if args.trades else 'no'}")
    print()

    # ── Fetch market-level data ──────────────────────────────────────
    market_rows = fetch_all_markets(series_list, start, end, progress)

    markets_file = os.path.join(args.output_dir, "kalshi_prices.csv")
    save_csv(market_rows, markets_file)

    # Summary stats
    if market_rows:
        n_events = len(set(r["event_ticker"] for r in market_rows))
        n_markets = len(market_rows)
        n_with_volume = sum(1 for r in market_rows
                           if r["volume"] and float(r["volume"]) > 0)
        print(f"\n  Summary: {n_events} event-days, "
              f"{n_markets} contracts, {n_with_volume} with volume")

    # ── Fetch trade-level data (optional) ────────────────────────────
    if args.trades and market_rows:
        trade_rows = fetch_all_trades(market_rows, progress)
        trades_file = os.path.join(args.output_dir, "kalshi_trades.csv")
        save_csv(trade_rows, trades_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
