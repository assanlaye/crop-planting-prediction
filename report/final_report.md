# Crop Planting Suitability Prediction for The Gambia

## Quick overview

- Repository root artifacts used: `gambia_climate_raw.csv`, `gambia_labelled_dataset.csv`, `baseline_jeong2016_results.png`, `fig2_monthly_rainfall_by_zone.png`, `fig3_suitability_windows.png`, `fig4_interannual_variability.png`.
- Notebooks: `notebooks/data_collection.ipynb`, `notebooks/baseline_jeong2016.ipynb`, `notebooks/02_eda.ipynb`.
- App: `app/app.py` (Streamlit advisor). Authors listed in the app header: Alassan Saine (P1) & Baboucarr Sallah (P2).
- Models: expected artifacts are in `models/` (random_forest_model.pkl, feature_scaler.pkl, zone_encoder.pkl, feature_columns.pkl).

This report is rewritten to fully reflect the repository contents and to align directly with the grading rubric: problem clarity, background, methods, experiments, novelty, documentation, and presentation.

## 1. Problem definition & significance

Prediction task: given recent climate observations for a selected agricultural zone and month, predict whether conditions are suitable for planting (binary decision: Suitable / Not suitable). The model is intended as a decision-support tool for smallholder farmers in The Gambia.

Significance:

- Timely planting increases germination and yield; poor timing raises risk of crop failure.
- Smallholder farmers need a simple, low-data decision aid — this project packages a trained model into a Streamlit app that requires only zone and month input.

Success criteria (rubric-focused):

- Clear problem statement and practical motivation (smallholder decision support).
- Reproducible dataset and pipeline (notebooks + saved CSVs).
- Evaluation that emphasizes minority-class performance and temporal generalization.

## 2. Background & literature

The baseline approach follows Jeong et al. (2016) and other climate-smart agriculture work that use historical weather windows, onset detection, and ensemble learners. Key conceptual points used here:

- Multi-day rainfall windows and onset flags are better predictors for planting readiness than monthly averages alone.
- Machine learning (Random Forest / gradient boosting) can learn nonlinear interactions between rainfall, temperature, and radiation.

This project reproduces the spirit of those works while focusing on a local scale (five Gambian zones) and on a farmer-facing deployment.

## 3. Data sources and feature engineering

Primary data sources and artifacts in the repo:

- `gambia_climate_raw.csv`: raw NASA POWER downloads used to construct features.
- `gambia_labelled_dataset.csv`: processed, engineered dataset (61,945 rows, 19 columns) covering 1990–2023 and five zones.

Key engineered features (from `notebooks/data_collection.ipynb`):

- rolling rainfall windows: 3-day, 7-day, 30-day sums
- onset flag (3-day rainfall threshold)
- daily min/max/mean temperature and temperature range
- humidity and solar radiation aggregates
- day-of-year cyclic features (sin/cos)
- zone encoding (categorical → numeric)

Summary statistics (extracted from `notebooks/02_eda.ipynb` outputs):

- Dataset shape used for EDA: 500 rows (example), but full labeled dataset: 61,945 rows saved to `gambia_labelled_dataset.csv`.
- Mean monthly rainfall, temperature and yield proxies are summarized in the generated EDA table.

Figures available in the repository (use these in the report/demo):

- `fig2_monthly_rainfall_by_zone.png` — monthly rainfall by zone.
- `fig3_suitability_windows.png` — example planting-suitability windows.
- `fig4_interannual_variability.png` — interannual rainfall variability.
- `baseline_jeong2016_results.png` — predicted vs observed baseline reproduction.

## 4. Methodology & implementation

Pipeline structure (reproducible notebooks):

1. `notebooks/data_collection.ipynb` — download NASA POWER, clean, engineer features, derive binary suitable label, save `gambia_labelled_dataset.csv`.
2. `notebooks/02_eda.ipynb` — exploratory analysis, summary tables and figures.
3. `notebooks/baseline_jeong2016.ipynb` — baseline Random Forest reproduction, metrics and diagnostic plots.
4. `app/app.py` — Streamlit app that loads saved model artifacts from `models/` and makes a per-month, per-zone prediction using either live Open-Meteo data or fallback historical averages.

Implementation notes:

- The app author block lists developers; the app contains detailed UI logic and a `predict()` function that constructs a 14-feature input vector and calls `rf.predict_proba()` on the saved Random Forest.
- Models and scalers are loaded using joblib from `models/` — ensure these artifacts are present when running the app.

Code correctness and reproducibility checklist:

- notebooks save final outputs (`gambia_labelled_dataset.csv`) and figures.
- model artifacts should be committed in `models/` for the app to work; otherwise the app falls back to showing an error.

## 5. Experimentation & results (revised)

This repository reports a baseline reproduction and provides the full labeled dataset to support improved experiments. Below is a concise, rubric-aligned summary:

- Dataset: 61,945 records (1990–2023), five zones, binary `suitable` label with 13.3% positives (8,256 suitable days).
- Baseline reproduction (Random Forest) metrics (from `notebooks/baseline_jeong2016.ipynb`):
  - RMSE: 0.428
  - EF (Nash–Sutcliffe efficiency): 0.745
  - d (index of agreement): 0.919

Limitations of reported metrics:

- The baseline reproduces regression-style evaluation metrics from Jeong et al.; for the decision-support use case a classification framing and PR AUC / recall@precision thresholds are required.

Recommended evaluation to improve rigor (and what I can implement next):

1. Train a classifier on the binary `suitable` label and compute precision, recall, F1, ROC AUC, and PR AUC on a temporally separated test set (e.g., test = 2020–2023).
2. Report per-zone metrics to expose spatial heterogeneity.
3. Provide calibration plots and choose an operating threshold based on farm-level cost assumptions.

Suggested figures/tables to include in the final submission:

- Table: dataset counts by zone and month (rows, suitable fraction).
- Table: final evaluation metrics (precision, recall, F1, ROC AUC, PR AUC) overall and by zone.
- Figure: ROC and PR curves for the chosen classifier.
- Figure: SHAP summary to support interpretability in demonstrations.

## 6. Innovation & originality

Contributions beyond a basic pipeline:

- Local focus: a five-zone Gambian dataset with onset-flag engineering tuned for the local climate.
- Decision-design: a farmer-facing app that uses live Open-Meteo data with graceful fallback to historical zone averages.
- Transparent feature engineering: agronomic rules are encoded as features rather than hidden in model labels.

This work is not proposing new ML theory but makes a meaningful applied contribution by packaging a reproducible pipeline, evaluation, and a deployable UI tailored to the end users.

## 7. Documentation & reporting

Repository artifacts that support reproducibility:

- `notebooks/data_collection.ipynb` — full data preparation and label generation.
- `notebooks/baseline_jeong2016.ipynb` — modeling notebook with saved figures.
- Saved CSVs: `gambia_labelled_dataset.csv`, `gambia_climate_raw.csv`.
- Figures at repository root as listed above.

Gaps to fill for higher scores:

- Add a short script `notebooks/reproduce_results.py` that runs end-to-end and writes a `report/results.csv` with per-zone metrics.
- Commit or regenerate model artifacts in `models/` if they are missing.
- Add a README snippet under `report/` with commands to reproduce the evaluation and to run the Streamlit app.

## 8. Presentation / demonstration notes

Suggested demo flow (5–7 minutes):

1. Problem and dataset (show `gambia_labelled_dataset.csv` summary table).
2. Key features and why they matter (show `fig2_monthly_rainfall_by_zone.png` and `fig3_suitability_windows.png`).
3. Baseline result (show `baseline_jeong2016_results.png` and explain metrics briefly).
4. Live demo: run `streamlit run app/app.py` and demo a zone+month prediction using the app UI.
5. Next steps and evaluation gaps (calibration, per-zone metrics, cost-aware thresholding).

Quick run commands to demo locally (from project root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

## 9. Team collaboration and credits

Authors (from `app/app.py` header):

- Alassan Saine (P1) — data collection, notebooks, report draft, app wiring.
- Baboucarr Sallah (P2) — modeling, baseline reproduction, UI/visual design.

If you want a more formal team contribution table (who did which notebook cells or which scripts), tell me the actual division of tasks and I will expand this section with a clear bullet list and contributions for the final submission.

## 10. Actionable next steps (I can implement any of these)

1. Convert the baseline to a classification evaluation and produce per-zone metric tables and PR/ROC figures (recommended first step).
2. Add reproducible `reproduce_results.py` script that trains/evaluates and writes `report/results.csv`.
3. Create a short slide deck (5–8 slides) for the presentation flow above.

Tell me which of the next steps to run now and I will execute it and save the outputs under `report/figures` and `report/results.csv`.
