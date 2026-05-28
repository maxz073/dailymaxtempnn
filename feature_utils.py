import math
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg

def sin_cos_encode(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * math.pi * values / period
    return np.sin(angle), np.cos(angle)

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    doy = df[date_col].dt.dayofyear.values.astype(float)
    month = df[date_col].dt.month.values.astype(float)

    df["sin_doy"], df["cos_doy"] = sin_cos_encode(doy, 365.25)
    df["sin_month"], df["cos_month"] = sin_cos_encode(month, 12)
    return df

def add_solar_features(df: pd.DataFrame, date_col: str = "date",
                       lat_col: str = "lat") -> pd.DataFrame:
    doy = df[date_col].dt.dayofyear.values.astype(float)
    lat_rad = np.radians(df[lat_col].values)

    decl_rad = np.radians(23.44) * np.sin(np.radians((360.0 / 365.25) * (doy + 284)))
    df["solar_declination"] = np.degrees(decl_rad)

    cos_ha = -np.tan(lat_rad) * np.tan(decl_rad)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    hour_angle = np.degrees(np.arccos(cos_ha))
    df["day_length_hours"] = (2.0 / 15.0) * hour_angle

    df["days_since_winter_solstice"] = np.where(doy >= 355, doy - 355, doy + 10)

    return df

def city_static_features(ticker: str) -> dict:
    meta = cfg.CITY_META[ticker]
    _, _, lat, lon = cfg.CITIES[ticker]
    return {
        "lat": lat,
        "lon": lon,
        "elevation_ft": meta["elevation_ft"],
        "coastal": float(meta["coastal"]),
        "desert": float(meta["desert"]),
        "continentality": meta["continentality"],
    }

def add_city_static_features(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    static_rows = []
    for ticker in df[ticker_col]:
        static_rows.append(city_static_features(ticker))
    static_df = pd.DataFrame(static_rows, index=df.index)
    return pd.concat([df, static_df], axis=1)

def add_city_index(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    df["city_idx"] = df[ticker_col].map(cfg.TICKER_TO_IDX)
    return df

class ScalerWrapper:

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None

    def fit(self, df: pd.DataFrame, columns: list[str]):
        self.columns = columns
        self.scaler.fit(df[columns].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns].values)
        return df

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.fit(df, columns)
        return self.transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "columns": self.columns}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scaler = d["scaler"]
        self.columns = d["columns"]
        return self

def add_onshore_wind_features(df: pd.DataFrame,
                               ticker_col: str = "ticker") -> pd.DataFrame:
    bearing_map = cfg.WATER_BODY_BEARING
    bearing_series = df[ticker_col].map(bearing_map)
    has_bearing = bearing_series.notna()

    wind_dir_rad = np.radians(df["wind_direction_lag1"].values)
    bearing_rad = np.radians(bearing_series.fillna(0).values)

    onshore = np.where(has_bearing,
                       np.cos(wind_dir_rad - bearing_rad),
                       0.0)
    df["onshore_wind_component"] = onshore
    df["wind_speed_x_onshore"] = df["wind_speed_10m_max_lag1"].values * onshore
    return df

def add_wind_direction_encoding(df: pd.DataFrame) -> pd.DataFrame:
    angle = 2 * math.pi * df["wind_direction_lag1"].values / 360.0
    df["wind_dir_sin_lag1"] = np.sin(angle)
    df["wind_dir_cos_lag1"] = np.cos(angle)
    return df

def compute_climatological_normals(df: pd.DataFrame, train_end: pd.Timestamp,
                                    temp_col: str = "nws_high",
                                    ticker_col: str = "ticker",
                                    date_col: str = "date",
                                    smooth_window: int = 15) -> pd.DataFrame:
    train = df[df[date_col] <= train_end].copy()
    train["doy"] = train[date_col].dt.dayofyear

    normals = train.groupby([ticker_col, "doy"])[temp_col].mean().reset_index()
    normals.rename(columns={temp_col: "clim_normal"}, inplace=True)

    smoothed = []
    for ticker in normals[ticker_col].unique():
        city_normals = normals[normals[ticker_col] == ticker].sort_values("doy")

        padded = pd.concat([city_normals.tail(smooth_window),
                            city_normals,
                            city_normals.head(smooth_window)])
        padded["clim_normal"] = padded["clim_normal"].rolling(smooth_window, center=True).mean()
        smoothed.append(padded.iloc[smooth_window:-smooth_window])

    return pd.concat(smoothed, ignore_index=True)

def add_lags(df: pd.DataFrame, col: str, lags: list[int],
             group_col: str = "ticker") -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df

def add_rolling(df: pd.DataFrame, col: str, windows: list[int],
                stats: list[str] = None, group_col: str = "ticker") -> pd.DataFrame:
    if stats is None:
        stats = ["mean"]
    shifted = df.groupby(group_col)[col].shift(1)
    for w in windows:
        for stat in stats:
            df[f"{col}_roll{w}_{stat}"] = shifted.groupby(df[group_col]).transform(
                lambda x, s=stat, ww=w: getattr(x.rolling(ww, min_periods=1), s)()
            )
    return df

def split_data(df: pd.DataFrame, date_col: str = "date"):
    train = df[(df[date_col] >= pd.Timestamp(cfg.TRAIN_START)) &
               (df[date_col] <= pd.Timestamp(cfg.TRAIN_END))].copy()
    val = df[(df[date_col] >= pd.Timestamp(cfg.VAL_START)) &
             (df[date_col] <= pd.Timestamp(cfg.VAL_END))].copy()
    test = df[(df[date_col] >= pd.Timestamp(cfg.TEST_START)) &
              (df[date_col] <= pd.Timestamp(cfg.TEST_END))].copy()
    return train, val, test
