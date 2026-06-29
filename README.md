# crop-planting-prediction

Planting suitability prediction workflow for The Gambia using NASA POWER climate data and Random Forest methods.

## Project Overview

This repository includes:

- A baseline reproduction notebook based on Jeong et al. (2016) to validate the machine learning pipeline.
- A data collection and feature engineering notebook that builds a labeled planting-suitability dataset for The Gambia.

## Repository Contents

- baseline_jeong2016.ipynb: Baseline Random Forest regression pipeline (sanity-check workflow).
- data_collection.ipynb: NASA POWER API download, cleaning, feature engineering, and suitability label generation.
- requirements.txt: Python dependencies for the notebooks.

## Environment Setup

1. Create and activate a virtual environment.

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install dependencies.

powershell
pip install -r requirements.txt

3. Open the notebooks and select the .venv Python kernel.

## Notebook Workflow

1. Run data_collection.ipynb to:

- define agricultural zones,
- download climate data from NASA POWER,
- clean and engineer features,
- generate binary suitability labels,
- save final dataset artifacts.

2. Run baseline_jeong2016.ipynb to:

- validate the Random Forest pipeline,
- compute baseline metrics,
- produce baseline plots.

## Outputs

Common generated files include:

- gambia_climate_raw.csv
- gambia_labelled_dataset.csv
- baseline and EDA figure files (\*.png)

These outputs are generated locally from notebook runs.

## How to Use the Gambia Crop Planting Advisor

### Prerequisites

Make sure you have completed the model training step before running the app.
The following files must exist in the `models/` folder:

```text
models/
├── random_forest_model.pkl
├── feature_scaler.pkl
├── zone_encoder.pkl
└── feature_columns.pkl
```

If any of these are missing, open and run all cells in
`model_training.ipynb` first.

---

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/assanlaye/crop-planting-prediction.git
cd crop-planting-prediction
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Running the App

From the **project root** (not from inside `app/`), run:

```bash
streamlit run app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> ⚠️ Always run from the project root. Running from inside `app/` will
> cause import errors because `src/config.py` will not be found.

---

### Using the App

The app has three tabs.

---

#### Tab 1 — 🌱 Crop Planting Time

This is the main farmer-facing view. It requires no technical knowledge.

**Step 1 — Choose the month**

Select the current month from the dropdown. The app defaults to
the current calendar month.

**Step 2 — Choose your zone**

Select the agricultural zone where your farm is located:

| Zone | Location |
|---|---|
| Western | Greater Banjul / Brikama area |
| North Bank | North Bank Region |
| Lower River | Lower River Region |
| Central River | Central River Region |
| Upper River | Upper River / Basse area |

**Step 3 — Press the button**

Click **🌱 Can I plant now?**

The app will:
1. Automatically fetch last year's weather data for your zone and month
   from the Open-Meteo Archive API
2. If the API is unavailable, fall back to built-in historical averages
3. Run the Random Forest model on the weather data
4. Display the result

**Reading the result**

| Result | Meaning |
|---|---|
| ✅ **Yes — Plant Now** | The AI model predicts conditions are suitable for groundnut planting |
| ⏳ **Wait — Not Yet** | The AI model predicts conditions are not yet suitable |

Below the result banner you will see:

- **AI Probability** — how confident the model is (e.g. 87% means 87% chance of suitable conditions)
- **Confidence word** — Very confident / Fairly confident / Not very sure
- **Mean Temperature** — average of min and max for the month
- **Onset Flag** — whether 3-day rainfall reached the monsoon onset threshold
- **Weather tiles** — the actual climate data used for the prediction
- **Agronomic Threshold Reference** — shows which of the three training rules passed or failed (for reference only — the model weighs all 14 features together, not just these three)
- **Best Planting Months calendar** — historical suitability across all 12 months for your zone, with the current month highlighted in orange

> 💡 The threshold reference showing a ✗ FAIL does not necessarily mean
> the model will predict "Not Yet". The Random Forest learned weighted
> combinations of all 14 features — a strong seasonal moisture signal
> can compensate for a weak onset signal, for example.

---

#### Tab 2 — 📊 Sensitivity Analysis

This tab is for technical users, researchers, and presentations.

It answers the question: *"How does changing one climate variable affect
the model's prediction, while everything else stays fixed?"*

**Step 1 — Set the zone and date**

Choose a zone and a representative date for your analysis.

**Step 2 — Set the fixed climate inputs**

Set all sliders to realistic rainy-season values before running.
Recommended starting values:

| Input | Recommended value |
|---|---|
| 3-day rainfall | 25 mm |
| 30-day rainfall | 80 mm |
| 7-day rainfall | 50 mm |
| Min temperature | 23 °C |
| Max temperature | 33 °C |
| Humidity | 75 % |
| Solar radiation | 18 MJ/m² |
| Daily rainfall | 8 mm |

> ⚠️ If you leave all sliders at their minimum values the model will
> always predict Not Suitable regardless of which variable you sweep,
> because dry-season temperature and humidity conditions override
> everything else.

**Step 3 — Choose a variable to sweep**

Select one variable. The app will vary it from its minimum to maximum
while keeping all other inputs fixed.

**Step 4 — Click ▶ Run Analysis**

The chart shows:
- **Green curve** — P(Suitable) probability as the variable changes
- **Green shaded area** — zone where model predicts Suitable (above 50%)
- **Red shaded area** — zone where model predicts Not Suitable (below 50%)
- **Dashed line** — 50% decision threshold
- **Blue dotted line** — agronomic threshold marker (where applicable)

**Reading the tipping point**

After the chart, the app shows:

- **Tipping point** — the exact value where suitability crosses 50%
- If suitability is above 50% across the entire range — the variable you
  chose is not the limiting factor; try adjusting the fixed inputs
- If suitability stays below 50% across the entire range — another fixed
  input is limiting the prediction; increase 3-day rainfall or temperature

---

#### Tab 3 — ℹ️ About

Shows project information, data sources, model details, and references.

---

### Data Sources

| Source | What it provides | When it is used |
|---|---|---|
| **Open-Meteo Archive API** | Last year's actual daily weather for each zone | Primary source — fetched automatically |
| **Built-in historical averages** | Long-term monthly climate averages per zone | Fallback if API is unavailable |
| **NASA POWER** | Historical climate data 1990–2023 | Used to train the model (offline) |

The app caches the Open-Meteo response for 24 hours. If you run the app
multiple times on the same day for the same zone and month, it uses the
cached result and does not make a new API call.

---

### Troubleshooting

**"Model not found" error**
Run `model_training.ipynb` and make sure all four `.pkl` files are saved
to the `models/` folder.

**"Module not found: src.config" error**
You are running the app from inside the `app/` folder. Move to the
project root and run `streamlit run app/app.py` again.

**"Suitability stays below 50% across the entire range" in Sensitivity Analysis**
The fixed inputs are at dry-season values. Set 3-day rainfall to at least
25mm, temperature to 23–33°C, and humidity to 75% before running the
analysis.

**Open-Meteo data not loading**
The app automatically falls back to historical averages. Check your
internet connection and try again later. The fallback values are still
agronomically valid for the zone and month selected.



## Reference

Jeong, J. H., et al. (2016). Random forests for global and regional crop yield predictions. PLOS ONE, 11(6), e0156571.
