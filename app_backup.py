import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f7fb;
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 16px 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .main-title {
            color: #17365d;
            font-size: 2rem;
            font-weight: 700;
        }
        .subtitle {
            color: #64748b;
            font-size: 1rem;
        }
        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button[kind="primary"] {
            background-color: #2563eb;
            color: #ffffff;
            border-radius: 8px;
            border: 0;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. LOAD MODELS AND DATA
# =========================================================

@st.cache_resource
def load_assets():
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
    }

    models = {}
    for model_name, filename in model_files.items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[model_name] = joblib.load(model_path)

    scaler_path = MODEL_DIR / "scaler.pkl"
    imputer_path = MODEL_DIR / "imputer.pkl"
    columns_path = MODEL_DIR / "feature_columns.pkl"

    for required_path in (scaler_path, imputer_path, columns_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required preprocessing file not found: {required_path}")

    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    feature_columns = joblib.load(columns_path)
