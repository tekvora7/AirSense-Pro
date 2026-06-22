import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="AirSense Pro",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2d3d;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1117;
        border-bottom: 2px solid #1e2d3d;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        font-weight: 600;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        color: #00aaff !important;
        border-bottom: 2px solid #00aaff !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff !important; }
    p, label { color: #8b949e !important; }
    
    /* Buttons */
    .stButton > button {
        background-color: #00aaff;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton > button:hover {
        background-color: #0088cc;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { color: #00aaff; }

    /* Divider */
    hr { border-color: #1e2d3d; }

    /* Title styling */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00aaff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        margin-top: 0;
    }
    .aqi-card {
        background: linear-gradient(135deg, #0d1117, #1e2d3d);
        border: 1px solid #00aaff44;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown('<p class="hero-title">🌫️ AirSense Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Real-Time Air Quality Forecasting for Indian Cities | Powered by ML</p>', unsafe_allow_html=True)
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/air_quality_processed.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df

# Load Models
@st.cache_resource
def load_models():
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('models/gradient_boosting_model.pkl', 'rb') as f:
        gb = pickle.load(f)
    return rf, xgb, gb

df = load_data()
rf_model, xgb_model, gb_model = load_models()

# Sidebar
with st.sidebar:
    st.markdown("### 🌫️ AirSense Pro")
    st.markdown("---")
    st.markdown("**📊 Dataset Info**")
    st.info(f"📁 {len(df):,} total records")
    st.info(f"🏙️ {df['city'].nunique()} monitoring stations")
    st.info(f"📅 2016 – 2024")
    st.markdown("---")
    st.markdown("**🤖 Models**")
    st.success("✅ Random Forest")
    st.success("✅ XGBoost")
    st.success("✅ Gradient Boosting")
    st.markdown("---")
    st.markdown("**⚠️ AQI Scale**")
    st.markdown("🟢 Good: 0–30")
    st.markdown("🟡 Moderate: 31–60")
    st.markdown("🟠 Unhealthy: 61–150")
    st.markdown("🔴 Hazardous: 150+")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🔮 Live Prediction",
    "🏙️ City Comparison",
    "🔍 Feature Analysis"
]) 
# Tab 1 - Overview
with tab1:
    st.header("📊 Dashboard Overview")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg PM2.5", f"{df['pm25'].mean():.1f} µg/m³")
    with col3:
        st.metric("Max PM2.5", f"{df['pm25'].max():.1f} µg/m³")
    with col4:
        st.metric("Monitoring Stations", df['city'].nunique())

    st.markdown("---")

    # PM2.5 distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].hist(df['pm25'], bins=50, color='#00aaff', edgecolor='black')
    axes[0].set_title('PM2.5 Distribution')
    axes[0].set_xlabel('PM2.5 (µg/m³)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(x=15, color='green', linestyle='--', label='WHO Safe (15)')
    axes[0].axvline(x=60, color='orange', linestyle='--', label='Moderate (60)')
    axes[0].axvline(x=150, color='red', linestyle='--', label='Unhealthy (150)')
    axes[0].legend(labelcolor='white')

    city_avg = df.groupby('city')['pm25'].mean().sort_values(ascending=False)
    axes[1].barh(city_avg.index, city_avg.values, color='#00ff88')
    axes[1].set_title('Average PM2.5 by Station')
    axes[1].set_xlabel('Average PM2.5 (µg/m³)')

    st.pyplot(fig)
    # Tab 2 - Trends
with tab2:
    st.header("📈 PM2.5 Trends Analysis")

    # Monthly trend
    monthly_avg = df.groupby(['year', 'month'])['pm25'].mean().reset_index()
    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes.flatten():
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    # Monthly trend
    axes[0,0].plot(monthly_avg['date'], monthly_avg['pm25'], color='#00aaff', linewidth=2)
    axes[0,0].axhline(y=15, color='green', linestyle='--', label='WHO Safe (15)')
    axes[0,0].axhline(y=150, color='red', linestyle='--', label='Unhealthy (150)')
    axes[0,0].set_title('Monthly Average PM2.5 Over Time')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('PM2.5 (µg/m³)')
    axes[0,0].legend(labelcolor='white')
    axes[0,0].grid(True, alpha=0.2)

    # Hourly pattern
    hourly_avg = df.groupby('hour')['pm25'].mean()
    axes[0,1].plot(hourly_avg.index, hourly_avg.values, color='#00ff88', linewidth=2, marker='o', markersize=4)
    axes[0,1].set_title('Average PM2.5 by Hour of Day')
    axes[0,1].set_xlabel('Hour')
    axes[0,1].set_ylabel('PM2.5 (µg/m³)')
    axes[0,1].grid(True, alpha=0.2)

    # Monthly seasonality
    monthly_pattern = df.groupby('month')['pm25'].mean()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    axes[1,0].bar(months, monthly_pattern.values, color='#ff6600')
    axes[1,0].set_title('Average PM2.5 by Month')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('PM2.5 (µg/m³)')
    axes[1,0].grid(True, alpha=0.2)

    # Weekend vs Weekday
    weekend_avg = df.groupby('is_weekend')['pm25'].mean()
    axes[1,1].bar(['Weekday', 'Weekend'], weekend_avg.values, color=['#00aaff', '#ff0066'])
    axes[1,1].set_title('Weekday vs Weekend PM2.5')
    axes[1,1].set_xlabel('Day Type')
    axes[1,1].set_ylabel('Average PM2.5 (µg/m³)')
    axes[1,1].grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    # Tab 3 - Live Prediction
with tab3:
    st.header("🔮 Live PM2.5 Prediction")
    st.markdown("Adjust the parameters to predict PM2.5 levels!")

    col1, col2 = st.columns(2)

    with col1:
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
        month = st.slider("Month", 1, 12, 6)
        year = st.slider("Year", 2016, 2024, 2023)
        is_weekend = st.selectbox("Day Type", [0, 1], format_func=lambda x: "Weekday" if x == 0 else "Weekend")

    with col2:
        city_options = sorted(df['city'].unique().tolist())
        city = st.selectbox("Select Station", city_options)
        season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Autumn'])
        pm25_lag1 = st.slider("PM2.5 Last Hour", 0.0, 500.0, 60.0)
        pm25_lag3 = st.slider("PM2.5 3 Hours Ago", 0.0, 500.0, 55.0)
        pm25_lag6 = st.slider("PM2.5 6 Hours Ago", 0.0, 500.0, 50.0)
        pm25_lag24 = st.slider("PM2.5 24 Hours Ago", 0.0, 500.0, 65.0)

    # Encode inputs
    city_list = sorted(df['city'].unique().tolist())
    season_list = ['Autumn', 'Spring', 'Summer', 'Winter']
    city_encoded = city_list.index(city)
    season_encoded = season_list.index(season)

    roll_mean_6 = (pm25_lag1 + pm25_lag3 + pm25_lag6) / 3
    roll_mean_24 = (pm25_lag1 + pm25_lag3 + pm25_lag6 + pm25_lag24) / 4
    roll_std_24 = np.std([pm25_lag1, pm25_lag3, pm25_lag6, pm25_lag24])

    input_data = np.array([[
        hour, day_of_week, month, year, is_weekend,
        city_encoded, season_encoded,
        pm25_lag1, pm25_lag3, pm25_lag6, pm25_lag24,
        roll_mean_6, roll_mean_24, roll_std_24
    ]])

    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]
    gb_pred = gb_model.predict(input_data)[0]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌲 Random Forest", f"{max(0, rf_pred):.1f} µg/m³")
    with col2:
        st.metric("⚡ XGBoost", f"{max(0, xgb_pred):.1f} µg/m³")
    with col3:
        st.metric("🚀 Gradient Boosting", f"{max(0, gb_pred):.1f} µg/m³")

    # AQI Health indicator
    avg_pred = (rf_pred + xgb_pred + gb_pred) / 3
    st.markdown("---")
    if avg_pred <= 30:
        st.success(f"🟢 Air Quality: GOOD ({avg_pred:.1f} µg/m³) — Safe to go outside!")
    elif avg_pred <= 60:
        st.warning(f"🟡 Air Quality: MODERATE ({avg_pred:.1f} µg/m³) — Sensitive groups be careful!")
    elif avg_pred <= 150:
        st.error(f"🟠 Air Quality: UNHEALTHY ({avg_pred:.1f} µg/m³) — Avoid outdoor activities!")
    else:
        st.error(f"🔴 Air Quality: HAZARDOUS ({avg_pred:.1f} µg/m³) — Stay indoors!")
# Tab 4 - City Comparison
with tab4:
    st.header("🏙️ City & Station Comparison")

    # City stats table
    city_stats = df.groupby('city')['pm25'].agg(['mean', 'min', 'max', 'std']).round(2)
    city_stats.columns = ['Avg PM2.5', 'Min PM2.5', 'Max PM2.5', 'Std Dev']
    city_stats = city_stats.sort_values('Avg PM2.5', ascending=False)
    st.dataframe(city_stats, use_container_width=True)

    st.markdown("---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    colors = ['#00aaff', '#00ff88', '#ff6600', '#ff0066', '#ffcc00',
              '#aa00ff', '#00ffcc', '#ff4488', '#88ff44', '#0066ff',
              '#ff8844', '#44aaff', '#ffaa00']

    # Box plot
    cities = sorted(df['city'].unique())
    data_by_city = [df[df['city'] == c]['pm25'].values for c in cities]
    bp = axes[0].boxplot(data_by_city, tick_labels=[c[:10] for c in cities],
                          patch_artist=True, vert=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for whisker in bp['whiskers']:
        whisker.set_color('white')
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_color('white')
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    for flier in bp['fliers']:
        flier.set_markerfacecolor('white')
        flier.set_markersize(2)
    axes[0].set_title('PM2.5 Distribution by Station')
    axes[0].set_ylabel('PM2.5 (µg/m³)')
    axes[0].tick_params(axis='x', rotation=90)

    # Seasonal - only major cities
    df_major = df[df['city'].isin(['Kolkata', 'Mumbai', 'New Delhi AirNow', 'RK Puram Delhi', 'Punjabi Bagh Delhi'])]
    seasonal = df_major.groupby(['season', 'city'])['pm25'].mean().unstack()
    seasonal.plot(kind='bar', ax=axes[1],
                  color=['#00aaff', '#00ff88', '#ff6600', '#ff0066', '#ffcc00'])
    axes[1].set_title('Seasonal PM2.5 — Major Stations')
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Average PM2.5 (µg/m³)')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(fontsize=8, labelcolor='white')

    plt.tight_layout()
    st.pyplot(fig)
    # Tab 5 - Feature Analysis
with tab5:
    st.header("🔍 Feature Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Feature Importance - Random Forest
        features = [
            'hour', 'day_of_week', 'month', 'year', 'is_weekend',
            'city_encoded', 'season_encoded',
            'pm25_lag1', 'pm25_lag3', 'pm25_lag6', 'pm25_lag24',
            'pm25_rolling_mean_6', 'pm25_rolling_mean_24', 'pm25_rolling_std_24'
        ]

        importance = rf_model.feature_importances_

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        colors = ['#00ff88' if i == importance.argmax() else '#00aaff' for i in range(len(features))]
        ax.barh(features, importance, color=colors)
        ax.set_title('Feature Importance (Random Forest)')
        ax.set_xlabel('Importance Score')

        st.pyplot(fig)

    with col2:
        # Correlation heatmap
        numeric_cols = ['pm25', 'hour', 'month', 'is_weekend',
                        'pm25_lag1', 'pm25_lag3', 'pm25_lag6', 'pm25_lag24',
                        'pm25_rolling_mean_6', 'pm25_rolling_mean_24']

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        ax2.tick_params(colors='white')
        ax2.title.set_color('white')

        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax2, annot_kws={'size': 7})
        ax2.set_title('Feature Correlation Matrix')

        st.pyplot(fig2)

    st.markdown("---")

    # Model comparison
    st.subheader("📊 Model Performance Comparison")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌲 Random Forest MAE", "18.00")
        st.metric("🌲 Random Forest RMSE", "33.00")
    with col2:
        st.metric("⚡ XGBoost MAE", "20.19")
        st.metric("⚡ XGBoost RMSE", "34.53")
    with col3:
        st.metric("🚀 Gradient Boosting MAE", "18.11")
        st.metric("🚀 Gradient Boosting RMSE", "31.80")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor('#0d1117')
    ax3.set_facecolor('#0d1117')
    ax3.tick_params(colors='white')
    ax3.title.set_color('white')
    ax3.xaxis.label.set_color('white')
    ax3.yaxis.label.set_color('white')

    models = ['Random Forest', 'XGBoost', 'Gradient Boosting']
    mae_scores = [18.00, 20.19, 18.11]
    rmse_scores = [33.00, 34.53, 31.80]
    x = np.arange(len(models))
    width = 0.35

    ax3.bar(x - width/2, mae_scores, width, label='MAE', color='#00aaff')
    ax3.bar(x + width/2, rmse_scores, width, label='RMSE', color='#00ff88')
    ax3.set_title('Model Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend(labelcolor='white')
    ax3.grid(True, alpha=0.2)

    st.pyplot(fig3)