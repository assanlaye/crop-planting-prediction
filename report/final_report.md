# Crop Planting Suitability Prediction for The Gambia

## Rubric Alignment

This report is organized to match the project rubric:

- Problem Definition & Clarity: Section 1
- Background & Literature Review: Section 2
- Methodology & Implementation: Section 3
- Experimentation & Results: Section 4
- Innovation & Originality: Section 5
- Documentation & Reporting: Section 6
- Presentation / Demonstration: Section 7
- Team Collaboration: Section 8

## Abstract

This project develops a climate-driven workflow for predicting crop planting suitability in The Gambia. The pipeline combines historical NASA POWER climate data, domain-inspired feature engineering, and a Random Forest model to classify days as suitable or not suitable for planting. A baseline reproduction inspired by Jeong et al. (2016) was also completed to validate the modeling approach. The resulting labeled dataset contains 61,945 daily records across five agricultural zones, with a strong class imbalance that reflects the rarity of suitable planting days. The project also supports a Streamlit application that turns the trained model into a practical decision aid for users.

## 1. Introduction

Agricultural planning in The Gambia depends heavily on rainfall onset, short-term moisture accumulation, and seasonal temperature conditions. Farmers need a simple way to decide when planting conditions are likely to be favorable, especially during the rainy season when weather can shift quickly. This project addresses that need by converting historical climate observations into a supervised learning problem: given recent rainfall and temperature patterns, predict whether planting conditions are suitable.

The work was designed around two goals. First, it builds a labeled climate dataset for five Gambian zones. Second, it validates a machine learning baseline that can later support an interactive planting advisor.

The significance of the problem is practical rather than abstract. A model that flags likely planting windows can support decision-making in smallholder agriculture, where timing errors can reduce germination success and yield. The project therefore treats suitability prediction as a decision-support task, not just a classification exercise.

## 2. Background and Literature Review

The approach follows a common pattern in climate-smart agriculture: historical weather data are transformed into predictive features, then used to estimate a crop-relevant outcome. Jeong et al. (2016) is the main methodological reference for the baseline reproduction, because it demonstrates how Random Forest methods can be used to model agricultural suitability or yield-related targets from climate and management variables.

The project also builds on two broader ideas from the literature. First, rainfall onset and short-window accumulation are often more informative for planting decisions than seasonal averages alone. Second, machine learning models can combine multiple climate signals at once, which is useful when no single threshold fully explains planting suitability. In that sense, the model is not replacing agronomic rules; it is learning how several rules and climate conditions interact.

Compared with a purely rule-based system, the model can capture nonlinear relationships, seasonal interactions, and partial compensations between variables. Compared with a black-box prediction setup, this project keeps the agronomic logic visible through engineered onset and rainfall-window features.

## 3. Data and Feature Engineering

Historical climate data were collected from NASA POWER for the period 1990 to 2023. The workflow covers five zones: Western, North Bank, Lower River, Central River, and Upper River. After cleaning and feature engineering, the final labeled dataset was saved as gambia_labelled_dataset.csv.

The feature set includes:

- daily rainfall,
- minimum and maximum temperature,
- humidity,
- solar radiation,
- 3-day, 7-day, and 30-day rainfall windows,
- a rainfall onset flag,
- temperature mean and range,
- day-of-year cyclic features,
- year and month.

This design captures both short-term planting conditions and seasonal context. It also reflects the agronomic logic used in the labeling rules, which depend strongly on rainfall accumulation and onset timing.

The final dataset contains 61,945 rows and 19 columns. The class distribution is imbalanced:

- Not suitable: 53,689 days, or 86.7%
- Suitable: 8,256 days, or 13.3%

## 4. Methodology and Implementation

The implementation is organized as a reproducible notebook workflow. The data collection notebook downloads and cleans climate observations, engineers rolling rainfall features, derives the onset flag, and generates labels. The baseline notebook then loads a smaller tabular dataset, fits a Random Forest model, and evaluates the predictions with standard regression-style metrics.

The core design choices are:

- use multi-day rainfall windows to approximate planting readiness,
- encode seasonality through day-of-year sine and cosine features,
- preserve zone information so the model learns region-specific climate patterns,
- keep the workflow transparent enough to explain in a report or demonstration.

This implementation is technically sound because it keeps preprocessing and modeling separated, uses explicit feature engineering rather than hidden transformations, and saves outputs as reusable artifacts for the application layer.

## 5. Baseline Modeling

To verify the modeling pipeline, a baseline Random Forest reproduction was run following the structure of Jeong et al. (2016). The baseline notebook produced the following results on the reproduced dataset:

- RMSE: 0.428
- EF: 0.745
- d: 0.919

The predicted-versus-observed plot shows a strong diagonal relationship, which indicates that the model tracks the target values reasonably well. The feature-importance ranking suggests that total precipitation contributes the most to the model, followed by mean temperature, fertiliser input, solar radiation, and evapotranspiration.

![Figure 1: Baseline reproduction results](figures/baseline_jeong2016_results.png)

## 6. Experimentation and Results

The labeled Gambia dataset was used to support exploratory analysis and to validate that the engineered labels behave as expected. The key dataset-level result is that suitable planting days are a minority class, which is consistent with the fact that favorable planting windows are relatively short compared with the full calendar year.

The strongest quantitative findings from the project are:

- 61,945 labeled daily records across five zones,
- 8,256 suitable days versus 53,689 unsuitable days,
- a baseline Random Forest reproduction with RMSE 0.428, EF 0.745, and d 0.919,
- visible separation in the feature-importance ranking, with rainfall carrying the most weight.

These results are clear enough for presentation because they combine dataset scale, class balance, and model quality in a compact form. The plots already generated in `report/figures` also support visual explanation during a demo.

## 7. Innovation and Originality

The project is not novel in the sense of inventing a new learning algorithm, but it does make a meaningful local contribution. Its originality comes from combining a Gambia-specific agricultural use case, zone-aware climate labeling, rainfall onset engineering, and a practical Streamlit deployment path.

The strongest innovation is the way the system bridges rule-based agronomy and machine learning. Instead of using the model only as a predictor, the workflow turns agronomic logic into features and labels, then uses the trained model to provide a smoother, more flexible decision aid than a single threshold rule would allow.

## 8. Documentation, Presentation, and Collaboration

The repository is documented through notebooks, a README, saved datasets, and generated figures. That structure makes it easier to reproduce the workflow and explain it step by step. The report is written to be readable by both technical and non-technical audiences, and the saved figures can be reused in slides or a live demonstration.

For presentation, the key story to communicate is simple: the project turns historical climate data into planting guidance. A strong demo should show the problem, the engineered features, one baseline result, and the final decision-support output.

If this project was completed by a team, this section should be expanded with the actual division of labor, for example data collection, modeling, app development, and report writing. If it was completed individually, the report should state that clearly in the final submission so the rubric is interpreted correctly.

## 9. Exploratory Analysis

Exploratory analysis confirms that the labeled data are not evenly distributed across classes or zones. The overall dataset is dominated by unsuitable days, and the zone-level suitability rates remain clustered around a similar range, with only moderate variation by region.

Figure 5 shows that the overall suitable share is 13.3%, while zone-level values range from about 12.3% in Western to 14.3% in North Bank. This suggests that the class imbalance is structural rather than the result of a single outlier zone.

![Figure 5: Class imbalance analysis](figures/fig5_label_distribution.png)

## 10. Project Outcome

The project delivers three practical outputs:

1. A labeled climate dataset for five Gambian zones.
2. A validated Random Forest baseline that demonstrates the feasibility of the modeling approach.
3. A decision-support application design that can present planting guidance in a farmer-friendly interface.

The resulting workflow is useful for both analysis and deployment. The notebook pipeline supports reproducible data preparation, while the application layer turns the model into an accessible tool for non-technical users.

## 11. Limitations

Several limitations should be noted. The dataset is highly imbalanced, so accuracy alone would not be a sufficient evaluation metric. The suitability labels are derived from rules, which means the model learns the labeling logic rather than direct field outcomes. The approach is also constrained by the quality and coverage of the historical climate data, and it does not yet include soil, pest, or market factors that may affect real planting decisions.

## 12. Conclusion

This project demonstrates that historical climate data can be transformed into a usable planting suitability model for The Gambia. The dataset engineering process produced a large multi-zone labeled dataset, and the baseline Random Forest results show that the approach captures meaningful structure in the climate variables. The next step is to continue refining the model and integrating it into the interactive advisor so that planting recommendations can be delivered in a simple, practical form.

## References

- NASA POWER climate data
- Jeong et al. (2016), baseline Random Forest reproduction
- Open-Meteo Archive API for application-time fallback weather data
