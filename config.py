from datetime import date
import os

TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2024, 4, 16)
VAL_START = date(2024, 4, 17)
VAL_END = date(2025, 4, 16)
TEST_START = date(2025, 4, 17)
TEST_END = date(2026, 4, 16)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

CITIES = {
    "KXHIGHNY":     ("New York",        "US/Eastern",   40.71, -74.01),
    "KXHIGHCHI":    ("Chicago",         "US/Central",   41.88, -87.63),
    "KXHIGHMIA":    ("Miami",           "US/Eastern",   25.76, -80.19),
    "KXHIGHTBOS":   ("Boston",          "US/Eastern",   42.36, -71.06),
    "KXHIGHLAX":    ("Los Angeles",     "US/Pacific",   34.05, -118.24),
    "KXHIGHAUS":    ("Austin",          "US/Central",   30.27, -97.74),
    "KXHIGHTSFO":   ("San Francisco",   "US/Pacific",   37.77, -122.42),
    "KXHIGHTDAL":   ("Dallas",          "US/Central",   32.78, -96.80),
    "KXHIGHPHIL":   ("Philadelphia",    "US/Eastern",   39.95, -75.17),
    "KXHIGHTPHX":   ("Phoenix",         "US/Arizona",   33.45, -112.07),
    "KXHIGHTOKC":   ("Oklahoma City",   "US/Central",   35.47, -97.52),
    "KXHIGHDEN":    ("Denver",          "US/Mountain",  39.74, -104.98),
    "KXHIGHTDC":    ("Washington DC",   "US/Eastern",   38.91, -77.04),
    "KXHIGHTSATX":  ("San Antonio",     "US/Central",   29.42, -98.49),
    "KXHIGHTHOU":   ("Houston",         "US/Central",   29.76, -95.37),
    "KXHIGHTMIN":   ("Minneapolis",     "US/Central",   44.98, -93.27),
    "KXHIGHTATL":   ("Atlanta",         "US/Eastern",   33.75, -84.39),
    "KXHIGHTSEA":   ("Seattle",         "US/Pacific",   47.61, -122.33),
    "KXHIGHTLV":    ("Las Vegas",       "US/Pacific",   36.17, -115.14),
    "KXHIGHTNOLA":  ("New Orleans",     "US/Central",   29.95, -90.07),
}

NWS_STATIONS = {
    "KXHIGHNY":    "NYC",
    "KXHIGHCHI":   "ORD",
    "KXHIGHMIA":   "MIA",
    "KXHIGHTBOS":  "BOS",
    "KXHIGHLAX":   "LAX",
    "KXHIGHAUS":   "AUS",
    "KXHIGHTSFO":  "SFO",
    "KXHIGHTDAL":  "DFW",
    "KXHIGHPHIL":  "PHL",
    "KXHIGHTPHX":  "PHX",
    "KXHIGHTOKC":  "OKC",
    "KXHIGHDEN":   "DEN",
    "KXHIGHTDC":   "DCA",
    "KXHIGHTSATX": "SAT",
    "KXHIGHTHOU":  "IAH",
    "KXHIGHTMIN":  "MSP",
    "KXHIGHTATL":  "ATL",
    "KXHIGHTSEA":  "SEA",
    "KXHIGHTLV":   "LAS",
    "KXHIGHTNOLA": "MSY",
}

CITY_TICKERS = list(CITIES.keys())
TICKER_TO_IDX = {t: i for i, t in enumerate(CITY_TICKERS)}
N_CITIES = len(CITY_TICKERS)

CITY_META = {
    "KXHIGHNY":    {"elevation_ft": 33,   "coastal": True,  "desert": False, "continentality": 0.40},
    "KXHIGHCHI":   {"elevation_ft": 594,  "coastal": False, "desert": False, "continentality": 0.70},
    "KXHIGHMIA":   {"elevation_ft": 6,    "coastal": True,  "desert": False, "continentality": 0.10},
    "KXHIGHTBOS":  {"elevation_ft": 20,   "coastal": True,  "desert": False, "continentality": 0.35},
    "KXHIGHLAX":   {"elevation_ft": 285,  "coastal": True,  "desert": False, "continentality": 0.15},
    "KXHIGHAUS":   {"elevation_ft": 489,  "coastal": False, "desert": False, "continentality": 0.55},
    "KXHIGHTSFO":  {"elevation_ft": 52,   "coastal": True,  "desert": False, "continentality": 0.10},
    "KXHIGHTDAL":  {"elevation_ft": 430,  "coastal": False, "desert": False, "continentality": 0.65},
    "KXHIGHPHIL":  {"elevation_ft": 39,   "coastal": False, "desert": False, "continentality": 0.45},
    "KXHIGHTPHX":  {"elevation_ft": 1086, "coastal": False, "desert": True,  "continentality": 0.90},
    "KXHIGHTOKC":  {"elevation_ft": 1201, "coastal": False, "desert": False, "continentality": 0.75},
    "KXHIGHDEN":   {"elevation_ft": 5280, "coastal": False, "desert": False, "continentality": 0.80},
    "KXHIGHTDC":   {"elevation_ft": 25,   "coastal": False, "desert": False, "continentality": 0.40},
    "KXHIGHTSATX": {"elevation_ft": 650,  "coastal": False, "desert": False, "continentality": 0.60},
    "KXHIGHTHOU":  {"elevation_ft": 80,   "coastal": True,  "desert": False, "continentality": 0.30},
    "KXHIGHTMIN":  {"elevation_ft": 830,  "coastal": False, "desert": False, "continentality": 0.85},
    "KXHIGHTATL":  {"elevation_ft": 1050, "coastal": False, "desert": False, "continentality": 0.50},
    "KXHIGHTSEA":  {"elevation_ft": 175,  "coastal": True,  "desert": False, "continentality": 0.20},
    "KXHIGHTLV":   {"elevation_ft": 2001, "coastal": False, "desert": True,  "continentality": 0.85},
    "KXHIGHTNOLA": {"elevation_ft": 3,    "coastal": True,  "desert": False, "continentality": 0.20},
}

WATER_BODY_BEARING = {
    "KXHIGHCHI":    90,
    "KXHIGHMIA":    180,
    "KXHIGHTBOS":   90,
    "KXHIGHLAX":    250,
    "KXHIGHNY":     135,
    "KXHIGHPHIL":   135,
    "KXHIGHTSEA":   270,
    "KXHIGHTSFO":   270,
    "KXHIGHTNOLA":  180,
    "KXHIGHTHOU":   150,
    "KXHIGHTATL":   None,
    "KXHIGHAUS":    None,
    "KXHIGHTDAL":   None,
    "KXHIGHTOKC":   None,
    "KXHIGHTPHX":   None,
    "KXHIGHTLV":    None,
    "KXHIGHDEN":    None,
    "KXHIGHTMIN":   0,
    "KXHIGHTSATX":  None,
    "KXHIGHTDC":    135,
}

MODEL1_HP = {
    "hidden_dims": [128, 64, 32],
    "dropout": [0.2, 0.15, 0.0],
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "batch_size": 256,
    "patience": 20,
    "city_embed_dim": 8,
    "n_restarts": 10,
}
