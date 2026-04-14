# 🏠 Automated ML Pipeline: House Price Prediction

## 📌 Overview
This project implements a complete **end-to-end MLOps pipeline** for predicting house prices using a synthetic California-style dataset.

The pipeline is fully automated and includes:
- Data ingestion
- Data preprocessing & feature engineering
- Model training & selection
- Evaluation
- Data drift detection

---

## ⚙️ Pipeline Architecture


---

## 🚀 Features

### 1. Data Ingestion
- Generates synthetic housing dataset
- Injects real-world missing values (2%)

### 2. Preprocessing
- Median imputation
- Feature engineering:
  - rooms_per_person
  - bedrms_per_room
  - pop_density
  - income_location
- Outlier clipping (1–99 percentile)

### 3. Model Training
Models compared:
- Gradient Boosting Regressor
- Random Forest Regressor
- Ridge Regression

✔ Uses:
- Cross-validation (5-fold)
- Pipeline (Imputer + Scaler + Model)

---

### 4. Evaluation Metrics
- RMSE
- MAE
- R² Score
- MAPE

---

### 5. Drift Detection
Advanced monitoring using:
- Kolmogorov-Smirnov Test
- Wasserstein Distance
- Population Stability Index (PSI)

✔ Automatically flags:
- Drifted features
- Critical distribution shifts

---

## 📊 Outputs

### Model Artifact
- `artifacts/best_model.pkl`

### Reports
- `evaluation.json`
- `drift_report.json`
- `pipeline_summary.json`

---

## ▶️ How to Run

```bash
python pipeline.py
