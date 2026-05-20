from pathlib import Path

# Repo root (works regardless of where you run from)
ROOT = Path(__file__).resolve().parents[1]

# Directory paths
DATA_RAW_DIR    = ROOT / 'data' / 'raw'
DATA_PROC_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR      = ROOT / 'models'
FIGURES_DIR     = ROOT / 'report' / 'figures'

# File paths
RAW_DATA_PATH   = DATA_RAW_DIR  / 'gambia_climate_raw.csv'       # P1's actual file name
PROC_DATA_PATH  = DATA_PROC_DIR / 'gambia_labelled_dataset.csv'

# Zones (as they appear in the 'zone' column of the processed CSV)
ZONES = ['Western', 'North Bank', 'Lower River', 'Central River', 'Upper River']

#  Feature columns (14 total — must match train.py and app.py exactly)
MODEL_FEATURES = [
    # Raw climate
    'rainfall_mm',
    'temp_min_C',
    'temp_max_C',
    'humidity_pct',
    'solar_rad_MJm2',
    # Engineered rainfall
    'rain_3d',
    'rain_7d',
    'rain_30d',
    'onset_flag',
    # Engineered temperature
    'temp_mean_C',
    'temp_range_C',
    # Circular seasonality
    'doy_sin',
    'doy_cos',
    'month',
]

LABEL_COL = 'suitable'
ZONE_COL  = 'zone'

# Temporal split
TRAIN_END_YEAR  = 2020
TEST_START_YEAR = 2021

# Agronomic thresholds (Sultan & Gaetani 2016, adapted for Gambia groundnut)
ONSET_THRESH    = 20   # rain_3d  >= 20 mm  → onset_flag = 1
SEASONAL_THRESH = 50   # rain_30d >= 50 mm
TEMP_MIN_SUIT   = 20   # temp_mean_C >= 20 °C
TEMP_MAX_SUIT   = 35   # temp_mean_C <= 35 °C

TARGET_CROP = 'groundnut'