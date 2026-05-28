import os
import time
import logging

import pandas as pd
import numpy as np
import requests

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = cfg.DATA_DIR
ARCHIVE_DIR = os.path.join(DATA_DIR, "weather_archive")
FORECAST_DIR = os.path.join(DATA_DIR, "weather_forecasts")
CLIMATE_DIR = os.path.join(DATA_DIR, "climate_indices")
NWS_DIR = os.path.join(DATA_DIR, "nws_daily")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ACIS_URL = "https://data.rcc-acis.org/StnData"

START = cfg.TRAIN_START.isoformat()
END = cfg.TEST_END.isoformat()

def fetch_nws_daily():
    os.makedirs(NWS_DIR, exist_ok=True)

    for ticker, station_id in cfg.NWS_STATIONS.items():
        out = os.path.join(NWS_DIR, f"{ticker}_nws.csv")
        if os.path.exists(out):
            log.info("Skipping NWS for %s — already fetched", ticker)
            continue

        city_name = cfg.CITIES[ticker][0]
        log.info("Fetching NWS daily high for %s (station %s)...", city_name, station_id)

        payload = {
            "sid": station_id,
            "sdate": START,
            "edate": END,
            "elems": [{"name": "maxt"}],
            "output": "json",
        }

        for attempt in range(3):
            try:
                resp = requests.post(ACIS_URL, json=payload, timeout=60)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < 2:
                    log.warning("  Retry %d for %s: %s", attempt + 1, station_id, e)
                    time.sleep(5 * (attempt + 1))
                else:
                    log.error("  %s: failed after 3 retries: %s", station_id, e)
                    continue

        data = resp.json()
        if "data" not in data:
            log.error("  %s: no data in ACIS response", station_id)
            continue

        rows = []
        for entry in data["data"]:
            date_str = entry[0]
            val = entry[1]

            if isinstance(val, str) and val not in ("M", "T", "S", ""):
                try:
                    rows.append({"date": date_str, "nws_high": float(val)})
                except ValueError:
                    rows.append({"date": date_str, "nws_high": np.nan})
            elif isinstance(val, (int, float)):
                rows.append({"date": date_str, "nws_high": float(val)})
            else:
                rows.append({"date": date_str, "nws_high": np.nan})

        df = pd.DataFrame(rows)
        df.insert(0, "ticker", ticker)
        df.to_csv(out, index=False)
        log.info("  %s: %d rows (%d valid)",
                 city_name, len(df), df["nws_high"].notna().sum())
        time.sleep(1)

    log.info("NWS daily high fetch complete.")

def load_nws_daily() -> pd.DataFrame:
    frames = []
    for ticker in cfg.CITY_TICKERS:
        path = os.path.join(NWS_DIR, f"{ticker}_nws.csv")
        if not os.path.exists(path):
            log.warning("Missing NWS data for %s — run fetch_nws_daily() first", ticker)
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No NWS data found. Run fetch_nws_daily() first.")
    return pd.concat(frames, ignore_index=True)

def fetch_weather_archive():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    daily_vars = [
        "temperature_2m_max", "temperature_2m_min",
        "dewpoint_2m_mean", "surface_pressure_mean",
        "cloud_cover_mean", "wind_speed_10m_max", "wind_direction_10m_dominant",
        "precipitation_sum", "snowfall_sum",
    ]
    hourly_vars = ["temperature_2m"]

    for ticker, (name, tz, lat, lon) in cfg.CITIES.items():
        out_daily = os.path.join(ARCHIVE_DIR, f"{ticker}_daily.csv")
        out_hourly = os.path.join(ARCHIVE_DIR, f"{ticker}_hourly.csv")
        if os.path.exists(out_daily) and os.path.exists(out_hourly):
            log.info("Skipping %s — already fetched", name)
            continue

        log.info("Fetching archive for %s (%s)...", name, ticker)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": START,
            "end_date": END,
            "daily": ",".join(daily_vars),
            "hourly": ",".join(hourly_vars),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": tz,
        }
        for attempt in range(5):
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning("  Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            log.error("  %s: failed after 5 retries", name)
            continue

        data = resp.json()

        daily = pd.DataFrame(data["daily"])
        daily.rename(columns={"time": "date"}, inplace=True)
        daily.insert(0, "ticker", ticker)
        daily.to_csv(out_daily, index=False)

        hourly = pd.DataFrame(data["hourly"])
        hourly.rename(columns={"time": "datetime"}, inplace=True)
        hourly.insert(0, "ticker", ticker)
        hourly.to_csv(out_hourly, index=False)

        log.info("  %s: %d daily rows, %d hourly rows", name, len(daily), len(hourly))
        time.sleep(3)

    log.info("Weather archive fetch complete.")

def load_archive_daily() -> pd.DataFrame:
    frames = []
    for ticker in cfg.CITY_TICKERS:
        path = os.path.join(ARCHIVE_DIR, f"{ticker}_daily.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No archive data found. Run fetch_weather_archive() first.")
    return pd.concat(frames, ignore_index=True)

def load_archive_hourly() -> pd.DataFrame:
    frames = []
    for ticker in cfg.CITY_TICKERS:
        path = os.path.join(ARCHIVE_DIR, f"{ticker}_hourly.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No hourly archive data found.")
    return pd.concat(frames, ignore_index=True)

FORECAST_MODELS = ["gfs_global", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "jma_seamless", "ncep_hrrr_conus"]

def fetch_weather_forecasts():
    os.makedirs(FORECAST_DIR, exist_ok=True)

    for ticker, (name, tz, lat, lon) in cfg.CITIES.items():
        out = os.path.join(FORECAST_DIR, f"{ticker}_forecasts.csv")
        if os.path.exists(out):
            log.info("Skipping forecasts for %s — already fetched", name)
            continue

        log.info("Fetching historical forecasts for %s...", name)
        model_frames = []
        for model in FORECAST_MODELS:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": START,
                "end_date": END,
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": tz,
                "models": model,
            }
            try:
                for attempt in range(5):
                    resp = requests.get(HIST_FORECAST_URL, params=params, timeout=60)
                    if resp.status_code == 429:
                        wait = 30 * (attempt + 1)
                        log.warning("  Rate limited on %s/%s, waiting %ds...", name, model, wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                else:
                    log.warning("  %s/%s: failed after retries", name, model)
                    continue
                data = resp.json()
                df = pd.DataFrame(data["daily"])
                df.rename(columns={"time": "date", "temperature_2m_max": f"fcst_{model}"}, inplace=True)
                model_frames.append(df.set_index("date"))
                log.info("  %s/%s: %d rows", name, model, len(df))
            except Exception as e:
                log.warning("  %s/%s failed: %s", name, model, e)
            time.sleep(2)

        if model_frames:
            combined = pd.concat(model_frames, axis=1)
            combined.insert(0, "ticker", ticker)
            combined.to_csv(out)
            log.info("  %s: saved %d forecast rows", name, len(combined))
        time.sleep(1)

    log.info("Forecast fetch complete.")

def fix_gfs_forecasts():
    os.makedirs(FORECAST_DIR, exist_ok=True)

    for ticker, (name, tz, lat, lon) in cfg.CITIES.items():
        path = os.path.join(FORECAST_DIR, f"{ticker}_forecasts.csv")
        if not os.path.exists(path):
            log.warning("No forecast CSV for %s — run fetch_weather_forecasts() first", name)
            continue

        df = pd.read_csv(path, parse_dates=["date"])

        if "fcst_gfs_global" in df.columns and "fcst_gfs_seamless" not in df.columns:
            log.info("Skipping %s — already migrated to gfs_global", name)
            continue

        log.info("Fetching gfs_global for %s...", name)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": START,
            "end_date": END,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": tz,
            "models": "gfs_global",
        }
        try:
            for attempt in range(5):
                resp = requests.get(HIST_FORECAST_URL, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    log.warning("  Rate limited on %s, waiting %ds...", name, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                log.warning("  %s: failed after retries", name)
                continue

            data = resp.json()
            gfs_df = pd.DataFrame(data["daily"])
            gfs_df.rename(columns={"time": "date", "temperature_2m_max": "fcst_gfs_global"}, inplace=True)
            gfs_df["date"] = pd.to_datetime(gfs_df["date"])

            if "fcst_gfs_seamless" in df.columns:
                df.drop(columns=["fcst_gfs_seamless"], inplace=True)

            if "fcst_gfs_global" in df.columns:
                df.drop(columns=["fcst_gfs_global"], inplace=True)

            df = df.merge(gfs_df[["date", "fcst_gfs_global"]], on="date", how="left")

            cols = ["date", "ticker", "fcst_gfs_global"] + [
                c for c in df.columns if c not in ("date", "ticker", "fcst_gfs_global")
            ]
            df = df[cols]
            df.to_csv(path, index=False)
            log.info("  %s: updated with gfs_global (%d rows)", name, len(df))

        except Exception as e:
            log.warning("  %s failed: %s", name, e)

        time.sleep(2)

    log.info("GFS forecast migration complete.")

def load_forecasts() -> pd.DataFrame:
    frames = []
    for ticker in cfg.CITY_TICKERS:
        path = os.path.join(FORECAST_DIR, f"{ticker}_forecasts.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No forecast data found. Run fetch_weather_forecasts() first.")
    return pd.concat(frames, ignore_index=True)

def fetch_climate_indices():
    os.makedirs(CLIMATE_DIR, exist_ok=True)

    _fetch_enso()
    _fetch_teleconnection("ao", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table")
    _fetch_teleconnection("nao", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table")
    _fetch_teleconnection("pna", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table")

    log.info("Climate indices fetch complete.")

def _fetch_enso():
    out = os.path.join(CLIMATE_DIR, "enso_oni.csv")
    if os.path.exists(out):
        log.info("ENSO ONI already fetched")
        return
    log.info("Fetching ENSO ONI...")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                season = parts[0]
                year = int(parts[1])
                oni = float(parts[-1])
                month_map = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5,
                             "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10,
                             "OND": 11, "NDJ": 12}
                month = month_map.get(season, 1)
                rows.append({"date": pd.Timestamp(year, month, 1), "enso_oni": oni})
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        log.info("  ENSO: %d rows", len(df))
    except Exception as e:
        log.warning("ENSO fetch failed: %s", e)

def _fetch_teleconnection(name: str, url: str):
    out = os.path.join(CLIMATE_DIR, f"{name}.csv")
    if os.path.exists(out):
        log.info("%s already fetched", name.upper())
        return
    log.info("Fetching %s...", name.upper())
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            for month_idx, val_str in enumerate(parts[1:13], start=1):
                try:
                    val = float(val_str)
                    if val < -90:
                        val = np.nan
                    rows.append({"date": pd.Timestamp(year, month_idx, 1), f"{name}": val})
                except ValueError:
                    pass
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        log.info("  %s: %d rows", name.upper(), len(df))
    except Exception as e:
        log.warning("%s fetch failed: %s", name.upper(), e)

def load_climate_indices() -> pd.DataFrame:
    date_range = pd.date_range(cfg.TRAIN_START, cfg.TEST_END, freq="D")
    result = pd.DataFrame({"date": date_range})

    for name in ["enso_oni", "ao", "nao", "pna"]:
        path = os.path.join(CLIMATE_DIR, f"{name}.csv")
        if not os.path.exists(path):
            log.warning("Missing climate index: %s", name)
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.as_unit("ns")
        col = [c for c in df.columns if c != "date"][0]
        df = df.sort_values("date")
        result["date"] = pd.to_datetime(result["date"]).dt.as_unit("ns")
        result = pd.merge_asof(result.sort_values("date"), df, on="date", direction="backward")

    return result

if __name__ == "__main__":
    fetch_nws_daily()
    fetch_weather_archive()
    fetch_weather_forecasts()
    fix_gfs_forecasts()
    fetch_climate_indices()
    log.info("All data fetched successfully.")
