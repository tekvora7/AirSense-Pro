# 🌫️ AirSense Pro — Real-Time Air Quality Forecasting System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![OpenAQ](https://img.shields.io/badge/OpenAQ-API-lightblue)

An end-to-end **Real-Time Air Quality Forecasting System** that predicts PM2.5 pollution levels using 132,728 records from 13 Indian monitoring stations fetched via the OpenAQ API.

---

## 📸 Dashboard Preview

| Tab | Description |
|-----|-------------|
| 📊 Overview | PM2.5 distribution + average by station |
| 📈 Trends | Monthly, hourly, seasonal patterns |
| 🔮 Live Prediction | Real-time PM2.5 prediction with sliders |
| 🏙️ City Comparison | Box plots + seasonal comparison |
| 🔍 Feature Analysis | Feature importance + correlation heatmap |

---

## 📊 Dataset

- **Source:** OpenAQ REST API
- **Stations:** 13 monitoring stations across Delhi, Mumbai, Kolkata
- **Records:** 132,728 clean records
- **Date Range:** 2016 – 2024
- **Target Variable:** PM2.5 (µg/m³)

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML Models:** Random Forest, XGBoost, Gradient Boosting
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Data Source:** OpenAQ API

---

## 🧠 ML Pipeline
[OpenAQ API] → [Data Collection] → [Feature Engineering] → [Model Training] → [PM2.5 Prediction]

(132K records)   (13 stations)      (14 features)           (RF/XGB/GB)        (µg/m³)

---

## 📈 Model Results

| Model | MAE | RMSE |
|-------|-----|------|
| 🌲 Random Forest | 18.00 | 33.00 |
| ⚡ XGBoost | 20.19 | 34.53 |
| 🚀 Gradient Boosting | 18.11 | **31.80** |

✅ **Best Model: Gradient Boosting** with RMSE of 31.80 and MAE of 18.11

---

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| hour | Hour of day |
| day_of_week | Day of week (0=Mon, 6=Sun) |
| month | Month of year |
| is_weekend | Weekend indicator |
| season_encoded | Season (Winter/Spring/Summer/Autumn) |
| city_encoded | Monitoring station |
| pm25_lag1 | PM2.5 value 1 hour ago |
| pm25_lag3 | PM2.5 value 3 hours ago |
| pm25_lag6 | PM2.5 value 6 hours ago |
| pm25_lag24 | PM2.5 value 24 hours ago |
| pm25_rolling_mean_6 | Rolling mean over 6 hours |
| pm25_rolling_mean_24 | Rolling mean over 24 hours |
| pm25_rolling_std_24 | Rolling std over 24 hours |

---

## 📁 Project Structure
AirSense-Pro/

├── data/               # Collected and processed CSV files

├── models/             # Saved .pkl models

├── notebooks/          # Jupyter notebooks

├── src/

│   └── app.py          # Streamlit dashboard

└── README.md

---

## 🏃 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost streamlit requests matplotlib seaborn

# Run dashboard
streamlit run src/app.py
```

---

## 📝 Resume Description

> **AirSense Pro — Real-Time Air Quality Forecasting System** | Python, Scikit-Learn, XGBoost, Streamlit
> - Built an end-to-end air quality forecasting pipeline ingesting 132,728 records from OpenAQ API across 13 Indian monitoring stations (2016–2024)
> - Engineered 14 time-series features including lag features, rolling statistics, hour of day, and seasonality indicators
> - Trained and compared Random Forest, XGBoost, and Gradient Boosting — Gradient Boosting achieved best RMSE of 31.80 and MAE of 18.11
> - Built an interactive dark-themed Streamlit dashboard (AirSense Pro) with 5 tabs for overview, trends, live PM2.5 prediction, city comparison, and feature analysis

---

## 🙏 Data Source

OpenAQ — Open Air Quality Data, https://openaq.org