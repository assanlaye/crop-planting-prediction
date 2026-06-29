"""
Gambia Crop Planting Advisor
============================
Streamlit web application — CPS 371: Artificial Intelligence
University of The Gambia, 2025/2026

Training data : NASA POWER (historical 1990-2023)
Live input     : Open-Meteo Archive API (monthly weather per zone)
Authors        : Alassan Saine (P1) & Baboucarr Sallah (P2)

Design priority: Illiterate smallholder farmers
  - Select month + zone → one button → AI decides
  - No manual data entry
  - Big visual green / red result
  - All if/else logic is DISPLAY ONLY — the RF model makes every prediction

Run with: streamlit run app/aapp.py  (from project root)
"""

import sys
import warnings
import calendar
from pathlib import Path
from datetime import date
from src.config import ZONES, MODEL_FEATURES

import joblib
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

warnings.filterwarnings("ignore")

# ── Resolve project root ──────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT    = APP_DIR.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    MODELS_DIR, ZONES,
    ONSET_THRESH, SEASONAL_THRESH,
    TEMP_MIN_SUIT, TEMP_MAX_SUIT,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Gambia Planting Advisor",
    page_icon="🌱",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
ZONE_COORDS = {
    "Western":       {"lat": 13.4549, "lon": -16.5790},
    "North Bank":    {"lat": 13.5460, "lon": -15.9000},
    "Lower River":   {"lat": 13.3700, "lon": -15.0000},
    "Central River": {"lat": 13.4900, "lon": -14.6500},
    "Upper River":   {"lat": 13.4700, "lon": -14.0500},
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

MONTH_ICONS = {
    1:"☀️",2:"☀️",3:"☀️",4:"🌤️",
    5:"🌧️",6:"🌧️",7:"⛈️",8:"⛈️",
    9:"🌧️",10:"🌤️",11:"☀️",12:"☀️",
}

# Historical monthly rainfall fallback (mm) per zone
ZONE_RAINFALL = {
    "Western":       [2,3,5,12,32,68,195,265,205,68,10,3],
    "North Bank":    [2,3,5,14,38,72,205,275,215,75,12,3],
    "Lower River":   [2,4,6,16,42,82,215,285,225,82,14,4],
    "Central River": [1,2,4,12,38,88,235,305,245,88,12,2],
    "Upper River":   [1,2,4,10,35,95,245,315,255,95,10,2],
}

# Planting calendar: 0=off-season, 1=possible, 2=good, 3=best
ZONE_CALENDAR = {
    "Western":       [0,0,0,0,0,2,3,2,0,0,0,0],
    "North Bank":    [0,0,0,0,0,2,3,2,0,0,0,0],
    "Lower River":   [0,0,0,0,0,1,3,2,0,0,0,0],
    "Central River": [0,0,0,0,0,1,3,3,1,0,0,0],
    "Upper River":   [0,0,0,0,0,0,3,3,1,0,0,0],
}

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #1A1F2E; }

[data-testid="stSidebar"] {
    background: #1A1F2E !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

.card {
    background: #FFFFFF; border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    border: 1px solid #2D3448;
}
.card-title {
    font-size: 0.72rem; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 0.9rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #F1F5F9;
}

.hero {
    background: linear-gradient(135deg, #14532D 0%, #166534 50%, #15803D 100%);
    border-radius: 16px; padding: 1.8rem 2.2rem; margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    border: 1px solid #16A34A;
}
.hero h1 { color: #F0FDF4; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.hero p  { color: #86EFAC; font-size: 0.88rem; margin: 0; }

.result-suitable {
    background: linear-gradient(135deg, #14532D, #166534);
    border-radius: 14px; padding: 1.5rem 1.8rem; margin: 0.8rem 0;
    box-shadow: 0 6px 20px rgba(22,101,52,0.4);
    border: 1px solid #16A34A;
}
.result-suitable h2 { color: #F0FDF4; font-size: 1.5rem; margin: 0 0 0.3rem 0; }
.result-suitable p  { color: #86EFAC; margin: 0; font-size: 0.9rem; line-height: 1.5; }

.result-unsuitable {
    background: linear-gradient(135deg, #7F1D1D, #991B1B);
    border-radius: 14px; padding: 1.5rem 1.8rem; margin: 0.8rem 0;
    box-shadow: 0 6px 20px rgba(153,27,27,0.4);
    border: 1px solid #EF4444;
}
.result-unsuitable h2 { color: #FEF2F2; font-size: 1.5rem; margin: 0 0 0.3rem 0; }
.result-unsuitable p  { color: #FCA5A5; margin: 0; font-size: 0.9rem; line-height: 1.5; }

.weather-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.8rem 0; }
.weather-tile {
    background: #F8FAFC; border-radius: 12px;
    padding: 1rem 0.8rem; flex: 1; min-width: 90px;
    border: 1px solid #E2E8F0; text-align: center;
}
.weather-tile .wt-icon { font-size: 1.5rem; display: block; margin-bottom: 4px; }
.weather-tile .wt-val  { font-size: 1.1rem; font-weight: 700; color: #0F172A; display: block; }
.weather-tile .wt-lbl  { font-size: 0.62rem; color: #64748B; text-transform: uppercase;
                          letter-spacing: 0.6px; margin-top: 3px; display: block; }

.metric-grid { display: flex; gap: 10px; margin: 0.8rem 0; flex-wrap: wrap; }
.metric-card {
    background: #FFFFFF; border-radius: 12px;
    padding: 0.9rem 1.1rem; flex: 1; min-width: 110px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    border: 1px solid #2D3448; border-top: 3px solid #16A34A;
}
.metric-card .label { font-size: 0.65rem; font-weight: 700; color: #64748B;
                       text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.25rem; }
.metric-card .value { font-size: 1.35rem; font-weight: 700; color: #0F172A; line-height: 1; }
.metric-card .sub   { font-size: 0.68rem; color: #94A3B8; margin-top: 0.2rem; }

.badge-pass {
    background: #DCFCE7; color: #14532D; padding: 3px 10px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    white-space: nowrap; border: 1px solid #86EFAC;
}
.badge-fail {
    background: #FEE2E2; color: #7F1D1D; padding: 3px 10px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    white-space: nowrap; border: 1px solid #FCA5A5;
}
.badge-warn {
    background: #FEF9C3; color: #713F12; padding: 3px 10px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    white-space: nowrap; border: 1px solid #FDE047;
}
.check-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.55rem 0; border-bottom: 1px solid #F1F5F9;
}
.check-row:last-child { border-bottom: none; }
.check-label { font-size: 0.85rem; color: #374151; flex: 1; }
.check-value {
    font-size: 0.8rem; color: #475569; background: #F8FAFC;
    padding: 2px 8px; border-radius: 6px; font-family: monospace;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab-list"] {
    background: #252B3B; border-radius: 12px; padding: 5px;
    gap: 4px; border: 1px solid #2D3448;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-weight: 600; font-size: 0.86rem;
    color: #94A3B8; padding: 0.45rem 1.1rem; transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: #2D3448; color: #CBD5E1;
}
.stTabs [aria-selected="true"] {
    background: #166534 !important; color: #F0FDF4 !important;
    box-shadow: 0 2px 6px rgba(22,101,52,0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #14532D, #166534) !important;
    color: #F0FDF4 !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    width: 100% !important;
    box-shadow: 0 3px 10px rgba(22,101,52,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

.stSelectbox label,
.stDateInput label { color: #CBD5E1 !important; }
.stSelectbox > div > div {
    background: #252B3B !important;
    color: #E2E8F0 !important;
    border-color: #2D3448 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        rf      = joblib.load(MODELS_DIR / "random_forest_model.pkl")
        scaler  = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        le      = joblib.load(MODELS_DIR / "zone_encoder.pkl")
        feat_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
        return rf, scaler, le, feat_cols, None
    except Exception as e:
        return None, None, None, None, str(e)

rf, scaler, le, feature_columns, load_err = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER FETCH  (Open-Meteo Archive — cached 24 h)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_weather(zone: str, month: int) -> dict:
    coords = ZONE_COORDS[zone]
    try:
        year  = date.today().year - 1
        days  = calendar.monthrange(year, month)[1]
        start = f"{year}-{str(month).zfill(2)}-01"
        end   = f"{year}-{str(month).zfill(2)}-{days}"
        url   = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&start_date={start}&end_date={end}"
            "&daily=precipitation_sum,temperature_2m_max,"
            "temperature_2m_min,relative_humidity_2m_mean,"
            "shortwave_radiation_sum"
            "&timezone=GMT"
        )
        r    = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()["daily"]

        precip = [v or 0.0 for v in data.get("precipitation_sum", [])]
        tmax   = [v for v in data.get("temperature_2m_max", [])          if v]
        tmin   = [v for v in data.get("temperature_2m_min", [])          if v]
        rh     = [v for v in data.get("relative_humidity_2m_mean", [])   if v]
        solar  = [v or 0.0 for v in data.get("shortwave_radiation_sum", [])]

        total_rain = sum(precip)
        avg_tmax   = sum(tmax)  / len(tmax)  if tmax  else 32.0
        avg_tmin   = sum(tmin)  / len(tmin)  if tmin  else 24.0
        avg_rh     = sum(rh)    / len(rh)    if rh    else 75.0
        avg_solar  = sum(solar) / len(solar) if solar else 17.0
        rain_3d    = sum(precip[-3:])
        rain_7d    = sum(precip[-7:])
        from_api   = True

    except Exception:
        # Fallback to historical climatology
        total_rain = ZONE_RAINFALL[zone][month - 1]
        avg_tmax   = 32.0
        avg_tmin   = 24.0
        avg_rh     = 75.0
        avg_solar  = 17.0
        rain_3d    = total_rain / 10
        rain_7d    = total_rain / 4
        from_api   = False

    avg_tmean  = (avg_tmax + avg_tmin) / 2
    temp_range = avg_tmax - avg_tmin
    mid_doy    = date(date.today().year, month, 15).timetuple().tm_yday
    doy_sin    = float(np.sin(2 * np.pi * mid_doy / 365))
    doy_cos    = float(np.cos(2 * np.pi * mid_doy / 365))
    onset_flag = int(rain_3d >= ONSET_THRESH)

    return {
        # ── model features (exact column names) ──────────────────────────────
        "rainfall_mm":    round(total_rain / 28, 2),
        "temp_min_C":     round(avg_tmin,   1),
        "temp_max_C":     round(avg_tmax,   1),
        "humidity_pct":   round(avg_rh,     1),
        "solar_rad_MJm2": round(avg_solar,  1),
        "rain_3d":        round(rain_3d,    2),
        "rain_7d":        round(rain_7d,    2),
        "rain_30d":       round(total_rain, 2),
        "onset_flag":     onset_flag,
        "temp_mean_C":    round(avg_tmean,  1),
        "temp_range_C":   round(temp_range, 1),
        "doy_sin":        round(doy_sin,    4),
        "doy_cos":        round(doy_cos,    4),
        # ── display extras (prefixed _ so they're excluded from model input) ─
        "_total_rain":    round(total_rain, 1),
        "_avg_tmax":      round(avg_tmax,   1),
        "_avg_tmin":      round(avg_tmin,   1),
        "_avg_tmean":     round(avg_tmean,  1),
        "_humidity":      round(avg_rh,     1),
        "_from_api":      from_api,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT  — the AI model decides, nothing else
# ══════════════════════════════════════════════════════════════════════════════
def predict(weather: dict, zone: str):
    """
    Build the 14-feature vector, run it through the Random Forest,
    and return (prediction, probability).

    This function contains zero business-logic overrides.
    The RF model is the sole decision maker.
    """
    zone_enc   = int(le.transform([zone])[0])

    # Build input dict from weather — skip display-only keys (prefixed _)
    input_dict = {k: v for k, v in weather.items() if not k.startswith("_")}
    input_dict["zone_encoded"] = zone_enc

    # Fill any missing columns with 0 (safety net)
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0.0

    X        = pd.DataFrame([input_dict])[feature_columns]
    X_scaled = scaler.transform(X)
    pred     = int(rf.predict(X_scaled)[0])
    proba    = float(rf.predict_proba(X_scaled)[0][1])

    return pred, proba


# ══════════════════════════════════════════════════════════════════════════════
# CALENDAR HTML  (display helper)
# ══════════════════════════════════════════════════════════════════════════════
def render_calendar(zone: str, current_month: int) -> str:
    cal   = ZONE_CALENDAR[zone]
    cells = ""
    for i, score in enumerate(cal):
        bg  = ["#2D3748", "#374151", "#166534", "#14532D"][score]
        fg  = ["#64748B", "#9CA3AF", "#F0FDF4", "#F0FDF4"][score]
        outline = (
            "outline:3px solid #F59E0B;outline-offset:-2px;"
            if i + 1 == current_month else ""
        )
        cells += (
            f'<div style="flex:1;height:36px;border-radius:5px;'
            f'background:{bg};color:{fg};display:flex;align-items:center;'
            f'justify-content:center;font-size:10px;font-weight:600;{outline}">'
            f'{MONTH_NAMES[i][0]}</div>'
        )
    legend = """
    <div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#94A3B8">
            <div style="width:12px;height:12px;border-radius:3px;background:#14532D"></div>Best
        </div>
        <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#94A3B8">
            <div style="width:12px;height:12px;border-radius:3px;background:#166534"></div>Good
        </div>
        <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#94A3B8">
            <div style="width:12px;height:12px;border-radius:3px;background:#2D3748"></div>Off-season
        </div>
        <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#94A3B8">
            <div style="width:12px;height:12px;border-radius:3px;
                        border:2px solid #F59E0B;background:transparent"></div>Now
        </div>
    </div>"""
    return (
        f'<div style="display:flex;gap:3px;margin-bottom:4px;">{cells}</div>'
        + legend
    )

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🌱  Farmer Edition",
    "📊  Sensitivity Analysis",
    "ℹ️   About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FARMER EDITION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div class="hero">
        <h1>🌱 Can I Plant Today?</h1>
        <p>Select your month and zone — the AI checks the weather data for you.</p>
    </div>
    """, unsafe_allow_html=True)

    if load_err:
        st.error(
            f"**Model not found.** Run `model_training.ipynb` first.\n\n`{load_err}`"
        )
        st.stop()

    # ── Zone & Month selectors ────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")
    with col1:
        month = st.selectbox(
            "📅 What month is it?",
            range(1, 13),
            format_func=lambda m: f"{MONTH_ICONS[m]}  {MONTH_NAMES[m-1]}",
            index=date.today().month - 1,
        )
    with col2:
        zone = st.selectbox("📍 Where is your farm?", ZONES, index=0)

    # ── Check button ──────────────────────────────────────────────────────────
    check_btn = st.button("🌱  Can I plant now?", width='stretch')

    if check_btn:
        with st.spinner(
            f"Fetching weather for {zone} — {MONTH_NAMES[month-1]}…"
        ):
            weather     = fetch_weather(zone, month)
            pred, proba = predict(weather, zone)

        src        = "Open-Meteo archive" if weather["_from_api"] else "historical data"
        temp_mean  = weather["temp_mean_C"]
        rain_30d   = weather["rain_30d"]
        rain_3d    = weather["rain_3d"]
        onset_flag = weather["onset_flag"]
        temp_ok    = TEMP_MIN_SUIT <= temp_mean <= TEMP_MAX_SUIT

        # ── Result banner — model decides ─────────────────────────────────────
        if pred == 1:
            st.markdown(f"""
            <div class="result-suitable">
                <h2>✅ &nbsp;Yes — Plant Now</h2>
                <p>The AI model predicts favourable planting conditions in
                   <b>{zone}</b> for <b>{MONTH_NAMES[month-1]}</b>.
                   Rainfall and temperature patterns support groundnut planting.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unsuitable">
                <h2>⏳ &nbsp;Wait — Not Yet</h2>
                <p>The AI model predicts conditions in <b>{zone}</b>
                   this <b>{MONTH_NAMES[month-1]}</b> are not yet suitable.
                   Consider waiting for more rainfall before planting.</p>
            </div>""", unsafe_allow_html=True)

            # ── Contextual hints (display only — do not change prediction) ───
            # These explain WHY the model likely said no.
            # They are shown AFTER the result, never before.
            if month == 6 and onset_flag and temp_ok and rain_30d < SEASONAL_THRESH:
                st.info(
                    "💡 **Note for June:** Onset rains have arrived but monthly "
                    "rainfall is still building. The model was trained on data "
                    "where full seasonal moisture had accumulated. Some farmers "
                    "in this zone plant at onset — use your own judgment and "
                    "wait for a follow-up rain event before sowing."
                )
            elif not temp_ok:
                st.warning(
                    f"🌡️ Temperature ({temp_mean}°C mean) is outside the "
                    f"optimal range ({TEMP_MIN_SUIT}–{TEMP_MAX_SUIT}°C)."
                )
            elif rain_30d < SEASONAL_THRESH and rain_3d < ONSET_THRESH:
                st.warning(
                    f"🌧️ Both monthly rainfall ({rain_30d:.0f}mm) and "
                    f"3-day rainfall ({rain_3d:.0f}mm) are below thresholds. "
                    f"Wait for the rains to establish."
                )

        # ── Confidence display ────────────────────────────────────────────────
        confidence = proba if pred == 1 else 1 - proba
        conf_word  = (
            "Very confident"   if confidence > 0.85 else
            "Fairly confident" if confidence > 0.65 else
            "Not very sure"
        )
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">AI Probability</div>
                <div class="value">{proba:.1%}</div>
                <div class="sub">P(Suitable)</div>
            </div>
            <div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value">{confidence:.0%}</div>
                <div class="sub">{conf_word}</div>
            </div>
            <div class="metric-card">
                <div class="label">Mean Temp</div>
                <div class="value">{temp_mean}°C</div>
                <div class="sub">Derived</div>
            </div>
            <div class="metric-card">
                <div class="label">Onset Flag</div>
                <div class="value">{"Yes ✅" if onset_flag else "No ❌"}</div>
                <div class="sub">rain_3d ≥ {ONSET_THRESH}mm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Weather tiles ─────────────────────────────────────────────────────
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">📡 Weather Data — {src}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="weather-grid">
            <div class="weather-tile">
                <span class="wt-icon">🌧️</span>
                <span class="wt-val">{weather['_total_rain']} mm</span>
                <span class="wt-lbl">Monthly Rain</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">🌂</span>
                <span class="wt-val">{weather['rain_3d']} mm</span>
                <span class="wt-lbl">3-day Rain</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">🌡️</span>
                <span class="wt-val">{weather['_avg_tmax']}°C</span>
                <span class="wt-lbl">Max Temp</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">🌡️</span>
                <span class="wt-val">{weather['_avg_tmin']}°C</span>
                <span class="wt-lbl">Min Temp</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">🌡️</span>
                <span class="wt-val">{weather['_avg_tmean']}°C</span>
                <span class="wt-lbl">Mean Temp</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">💧</span>
                <span class="wt-val">{weather['_humidity']}%</span>
                <span class="wt-lbl">Humidity</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">☀️</span>
                <span class="wt-val">{weather['solar_rad_MJm2']}</span>
                <span class="wt-lbl">Solar MJ/m²</span>
            </div>
            <div class="weather-tile">
                <span class="wt-icon">{"✅" if onset_flag else "❌"}</span>
                <span class="wt-val">{"Yes" if onset_flag else "No"}</span>
                <span class="wt-lbl">Onset Flag</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Threshold check + Calendar ────────────────────────────────────────
        checks = [
            (
                f"Seasonal moisture  (rain_30d ≥ {SEASONAL_THRESH}mm)",
                f"{rain_30d:.0f} mm",
                rain_30d >= SEASONAL_THRESH,
            ),
            (
                f"Monsoon onset  (rain_3d ≥ {ONSET_THRESH}mm)",
                f"{rain_3d:.0f} mm",
                rain_3d >= ONSET_THRESH,
            ),
            (
                f"Temperature window  ({TEMP_MIN_SUIT}–{TEMP_MAX_SUIT}°C)",
                f"{temp_mean}°C mean",
                temp_ok,
            ),
        ]
        rows_html = ""
        for lbl, val, passed in checks:
            badge = (
                '<span class="badge-pass">✓ PASS</span>' if passed
                else '<span class="badge-fail">✗ FAIL</span>'
            )
            rows_html += f"""
            <div class="check-row">
                {badge}
                <span class="check-label">{lbl}</span>
                <span class="check-value">{val}</span>
            </div>"""

        col_l, col_r = st.columns([1, 1], gap="large")
        with col_l:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Agronomic Threshold Reference</div>
                <p style="font-size:0.78rem;color:#64748B;margin:0 0 0.6rem 0;">
                    These thresholds inform the AI — the model learned from them
                    during training. They are shown for reference only.
                </p>
                {rows_html}
            </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown(
                '<div class="card">'
                '<div class="card-title">Best Planting Months — Historical</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_calendar(zone, month), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            f"📡 Source: {src}  ·  Zone: {zone}  ·  "
            f"{MONTH_NAMES[month-1]}  ·  "
            f"Model: Random Forest (300 trees, 14 features)  ·  "
            f"CPS 371 University of The Gambia"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="hero">
        <h1>📊 Sensitivity Analysis</h1>
        <p>Sweep one climate variable across its full range — all others held fixed —
           to find the exact tipping point where the AI predicts planting is suitable.</p>
    </div>
    """, unsafe_allow_html=True)

    if load_err:
        st.error(
            f"**Model not found.** Run `model_training.ipynb` first.\n\n`{load_err}`"
        )
        st.stop()

    col_ctrl, col_chart = st.columns([1, 2.4], gap="large")

    with col_ctrl:
        st.markdown(
            '<div class="card"><div class="card-title">📍 Zone & Date</div>',
            unsafe_allow_html=True,
        )
        s_zone = st.selectbox("Zone", ZONES, key="s_zone")
        s_date = st.date_input(
            "Date",
            value=date(date.today().year, date.today().month, 15),
            key="s_date",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">⚙️ Fixed Climate Inputs</div>',
            unsafe_allow_html=True,
        )
        s_r3 = st.slider("3-day rainfall (mm)", 0.0, 120.0, 15.0, 0.5, key="s_r3")
        s_r30 = st.slider("30-day rainfall (mm)", 0.0, 400.0, 60.0, 5.0, key="s_r30")
        s_r7 = st.slider("7-day rainfall (mm)", 0.0, 200.0, 25.0, 1.0, key="s_r7")
        s_tmin = st.slider("Min temperature (°C)", 9.0, 32.0, 22.0, 0.5, key="s_tmin")
        s_tmax = st.slider("Max temperature (°C)", 20.0, 47.0, 34.0, 0.5, key="s_tmax")
        s_hum = st.slider("Humidity (%)", 5.0, 95.0, 65.0, 1.0, key="s_hum")
        s_sol = st.slider("Solar radiation (MJ/m²)", 2.0, 29.0, 20.0, 0.5, key="s_sol")
        s_rain = st.slider("Daily rainfall (mm)", 0.0, 60.0, 5.0, 0.5, key="s_rain")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">🔬 Variable to Sweep</div>',
            unsafe_allow_html=True,
        )
        SWEEP_OPTIONS = {
            "3-day rainfall (mm)": ("rain_3d", 5, 0.0, 120.0),
            "30-day rainfall (mm)": ("rain_30d", 7, 0.0, 400.0),
            "Mean temperature (°C)": ("t_mean", 9, 10.0, 45.0),
            "Humidity (%)": ("humidity", 3, 5.0, 95.0),
            "Solar radiation (MJ/m²)": ("solar", 4, 2.0, 29.0),
        }
        sweep_label = st.selectbox(
            "Variable", list(SWEEP_OPTIONS.keys()), key="s_var"
        )
        run_btn = st.button(
            "▶  Run Analysis", width='stretch', key="run_sa"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart:
        if not run_btn:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:420px;background:#252B3B;border-radius:14px;
                        border:1px solid #2D3448;">
                <div style="text-align:center;color:#475569;">
                    <div style="font-size:3rem;margin-bottom:0.8rem">📊</div>
                    <div style="font-size:1rem;font-weight:600;color:#64748B;
                                margin-bottom:0.4rem;">No analysis yet</div>
                    <div style="font-size:0.85rem;color:#475569;line-height:1.6;">
                        Set inputs on the left,<br>choose a variable,<br>
                        then click
                        <b style="color:#16A34A">▶ Run Analysis</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            feat_key, feat_idx, v_min, v_max = SWEEP_OPTIONS[sweep_label]

            ts_s = pd.Timestamp(s_date)
            doy_s = ts_s.dayofyear
            sin_s = float(np.sin(2 * np.pi * doy_s / 365))
            cos_s = float(np.cos(2 * np.pi * doy_s / 365))
            tmean_s = (s_tmin + s_tmax) / 2
            trange_s = s_tmax - s_tmin
            zone_enc = int(le.transform([s_zone])[0])
            onset_s = int(s_r3 >= ONSET_THRESH)

            # Feature vector order must match feature_columns exactly:
            # rainfall_mm(0), temp_min_C(1), temp_max_C(2), humidity_pct(3),
            # solar_rad_MJm2(4), rain_3d(5), rain_7d(6), rain_30d(7),
            # onset_flag(8), temp_mean_C(9), temp_range_C(10),
            # doy_sin(11), doy_cos(12), zone_encoded(13)
            base = [
                s_rain,  # 0  rainfall_mm
                s_tmin,  # 1  temp_min_C
                s_tmax,  # 2  temp_max_C
                s_hum,  # 3  humidity_pct
                s_sol,  # 4  solar_rad_MJm2
                s_r3,  # 5  rain_3d
                s_r7,  # 6  rain_7d
                s_r30,  # 7  rain_30d
                onset_s,  # 8  onset_flag
                tmean_s,  # 9  temp_mean_C
                trange_s,  # 10 temp_range_C
                sin_s,  # 11 doy_sin
                cos_s,  # 12 doy_cos
                zone_enc,  # 13 zone_encoded
            ]

            sweep_vals = np.linspace(v_min, v_max, 150)
            probas = []

            for val in sweep_vals:
                row = base.copy()
                row[feat_idx] = val
                # Recompute derived features when sweeping
                if feat_idx == 5:  # rain_3d → update onset_flag
                    row[8] = int(val >= ONSET_THRESH)
                if feat_key == "t_mean":  # mean temp → update min/max/range
                    half = trange_s / 2
                    row[1] = val - half  # temp_min_C
                    row[2] = val + half  # temp_max_C
                    row[9] = val  # temp_mean_C

                X_row = pd.DataFrame(
                    [dict(zip(feature_columns, row))]
                )[feature_columns]
                X_scaled = scaler.transform(X_row)
                probas.append(float(rf.predict_proba(X_scaled)[0][1]))

            probas = np.array(probas)

            # ── Plot ─────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#1E2535")
            ax.set_facecolor("#1E2535")

            ax.fill_between(sweep_vals, probas, 0.5,
                            where=(probas >= 0.5),
                            alpha=0.20, color="#16A34A", label="Suitable zone")
            ax.fill_between(sweep_vals, probas, 0.5,
                            where=(probas < 0.5),
                            alpha=0.20, color="#EF4444", label="Not suitable zone")
            ax.plot(sweep_vals, probas, color="#4ADE80",
                    linewidth=2.5, zorder=4, label="P(Suitable)")
            ax.axhline(0.5, color="#94A3B8", linestyle="--",
                       linewidth=1.2, zorder=3, label="Decision threshold (50%)")

            AGRO_MARKERS = {
                "rain_3d": [(ONSET_THRESH, "#60A5FA", f"Onset ≥{ONSET_THRESH}mm")],
                "rain_30d": [(SEASONAL_THRESH, "#60A5FA", f"Seasonal ≥{SEASONAL_THRESH}mm")],
                "t_mean": [
                    (TEMP_MIN_SUIT, "#FBBF24", f"Min {TEMP_MIN_SUIT}°C"),
                    (TEMP_MAX_SUIT, "#FBBF24", f"Max {TEMP_MAX_SUIT}°C"),
                ],
            }
            for xv, col, lbl in AGRO_MARKERS.get(feat_key, []):
                ax.axvline(xv, color=col, linewidth=1.2,
                           linestyle=":", alpha=0.9, zorder=2)
                ax.text(xv + (v_max - v_min) * 0.012, 0.94, lbl,
                        color=col, fontsize=8.5, va="top", fontweight="600")

            ax.set_xlabel(sweep_label, fontsize=11, color="#CBD5E1", labelpad=8)
            ax.set_ylabel("P(Suitable for Planting)", fontsize=11,
                          color="#CBD5E1", labelpad=8)
            ax.set_title(
                f"Sensitivity: {sweep_label}  ·  Zone: {s_zone}"
                f"  ·  {s_date.strftime('%d %b %Y')}",
                fontsize=11, color="#F1F5F9", pad=12,
            )
            ax.set_ylim(-0.04, 1.08)
            ax.set_xlim(v_min, v_max)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.tick_params(colors="#94A3B8", labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#2D3748")
            ax.grid(linestyle="--", alpha=0.15, color="#94A3B8")
            ax.legend(fontsize=9, framealpha=0.25, fancybox=True,
                      labelcolor="#CBD5E1", facecolor="#1E2535",
                      edgecolor="#2D3748")
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()

            c1, c2, c3 = st.columns(3)
            c1.metric("Min P(Suitable)", f"{probas.min():.1%}")
            c2.metric("Max P(Suitable)", f"{probas.max():.1%}")
            c3.metric("Range", f"{probas.max() - probas.min():.1%}")

            crossing = None
            for i in range(len(probas) - 1):
                if probas[i] < 0.5 <= probas[i + 1]:
                    frac = (0.5 - probas[i]) / (probas[i + 1] - probas[i])
                    crossing = (
                            sweep_vals[i] + frac * (sweep_vals[i + 1] - sweep_vals[i])
                    )
                    break

            if crossing is not None:
                unit = sweep_label.split("(")[-1].rstrip(")")
                st.success(
                    f"**Tipping point:** suitability crosses 50% when "
                    f"**{sweep_label}** reaches **{crossing:.1f} {unit}**"
                )
            elif probas.min() >= 0.5:
                st.info(
                    f"Suitability is above 50% across the entire range — "
                    f"**{sweep_label}** is not the limiting factor."
                )
            else:
                st.warning(
                    "Suitability stays below 50% across the entire range — "
                    "another input is the limiting factor. Adjust the fixed inputs."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About this Project</h1>
        <p>CPS 371: Artificial Intelligence · University of The Gambia · 2025/2026</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            '<div class="card"><div class="card-title">🏫 Project Information</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
| | |
|:---|:---|
| **Course** | CPS 371: Artificial Intelligence |
| **Institution** | University of The Gambia |
| **Year** | 2025 / 2026 |
| **Option** | Option A — Enhancing an Existing AI Model |
| **Baseline** | Jeong et al. (2016) · RMSE 12.4% |
| **P1 (Model)** | Alassan Saine |
| **P2 (App)** | Baboucarr Sallah |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">🌐 Data Sources</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
| Source | Role |
|:---|:---|
| **NASA POWER** | Model training (1990–2023) |
| **Open-Meteo Archive** | Monthly weather fetch per zone |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div class="card"><div class="card-title">📐 Model Details</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
| | |
|:---|:---|
| **Algorithm** | Random Forest Classifier |
| **Trees** | 300 |
| **Class weight** | balanced |
| **CV F1** | 0.9998 |
| **Features** | 14 (incl. zone_encoded) |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">📚 References</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
- Jeong et al. (2016). Random forests for global and regional crop yield
  predictions. *PLOS ONE*, 11(6).
- Sultan & Gaetani (2016). Agriculture in West Africa in the twenty-first
  century. *Frontiers in Plant Science*, 7.
- NASA POWER: https://power.larc.nasa.gov
- Open-Meteo: https://open-meteo.com
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Gambia Crop Planting Advisor · "
        "Alassan Saine & Baboucarr Sallah · "
        "University of The Gambia · "
        "CPS 371 Artificial Intelligence 2025–2026"
    )