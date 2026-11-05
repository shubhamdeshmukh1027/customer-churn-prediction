
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "telco_churn_cleaned.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl"
}

# --------------------------------------------------
# PROFESSIONAL DARK THEME CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #20232e;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(110deg, #102a43, #176b87);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    font-weight: 750;
}

.hero p {
    color: #e5f3fa !important;
    font-size: 17px;
}

[data-testid="stMetric"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    padding: 20px !important;
    border-radius: 14px !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 30px !important;
}

[data-testid="stMetricDelta"] {
    color: #4ade80 !important;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 20px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #1d4ed8;
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

p, label, span {
    color: #e2e8f0;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_assets():

    feature_columns = joblib.load(
        os.path.join(MODEL_DIR, "feature_columns.pkl")
    )

    imputer = joblib.load(
        os.path.join(MODEL_DIR, "imputer.pkl")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    models = {}

    for name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(path):
            models[name] = joblib.load(path)

    if not models:
        raise FileNotFoundError("No trained models found.")

    return feature_columns, imputer, scaler, models


# --------------------------------------------------
# CUSTOMER INPUT PREPROCESSING
# --------------------------------------------------

def prepare_input(customer, feature_columns, imputer, scaler):

    # Create one row with exact training feature names
    data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Numerical features
    numerical_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for col in numerical_columns:

        if col in feature_columns:
            data.loc[0, col] = customer[col]

    # Categorical feature mapping
    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    # Create the exact dummy columns expected by training
    for col in categorical_columns:

        selected_value = str(customer[col])

        dummy_name = f"{col}_{selected_value}"

        if dummy_name in feature_columns:
            data.loc[0, dummy_name] = 1

    # Ensure correct feature order
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Imputer expects all 30 training features
    imputed_data = imputer.transform(data)

    data = pd.DataFrame(
        imputed_data,
        columns=feature_columns
    )

    # Scaler expects only three numerical columns
    scaled_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    data.loc[:, scaled_columns] = scaler.transform(
        data[scaled_columns]
    )

    return data[feature_columns].to_numpy()


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def get_probability(model, X):

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0][1])

    return float(model.predict(X)[0])


# --------------------------------------------------
# LOAD PROJECT FILES
# --------------------------------------------------

if not os.path.exists(DATA_PATH):

    st.error("Dataset file not found.")
    st.code(DATA_PATH)
    st.stop()

try:

    df = load_dataset()

    feature_columns, imputer, scaler, models = load_assets()

except Exception as error:

    st.error("Unable to load project files.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# NORMALIZE CHURN COLUMN
# --------------------------------------------------

def convert_churn(value):

    value = str(value).strip().lower()

    if value in ["yes", "1", "true"]:
        return 1

    if value in ["no", "0", "false"]:
        return 0

    return np.nan


df["Churn_Flag"] = df["Churn"].apply(convert_churn)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📊 Churn Prediction")

st.sidebar.caption("Telecom Customer Analytics")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Customer Prediction",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Telecom Customer Churn Prediction"
)


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Telecom Customer Churn Dashboard</h1>
        <p>Analyze customer behaviour and identify churn patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI calculations
    total = len(df)

    churn_count = int(
        df["Churn_Flag"].sum()
    )

    churn_rate = (
        df["Churn_Flag"].mean() * 100
        if total else 0
    )

    avg_charges = df["MonthlyCharges"].mean()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{total:,}"
    )

    c2.metric(
        "Churned Customers",
        f"{churn_count:,}"
    )

    c3.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%"
    )

    c4.metric(
        "Average Monthly Charges",
        f"${avg_charges:.2f}"
    )

    st.divider()

    # Churn distribution and contract analysis
    left, right = st.columns(2)

    with left:

        st.subheader("Customer Churn Distribution")

        churn_data = pd.DataFrame({
            "Customer Status": [
                "Retained",
                "Churned"
            ],
            "Customers": [
                int((df["Churn_Flag"] == 0).sum()),
                int((df["Churn_Flag"] == 1).sum())
            ]
        })

        fig = px.pie(
            churn_data,
            names="Customer Status",
            values="Customers",
            hole=0.48,
            title="Churn vs Retained Customers",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Churn Rate by Contract")

        contract_data = (
            df.groupby("Contract", as_index=False)
            ["Churn_Flag"].mean()
        )

        contract_data["Churn Rate (%)"] = (
            contract_data["Churn_Flag"] * 100
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate (%)",
            color="Contract",
            title="Churn by Contract Type",
            template="plotly_dark",
            text_auto=".1f"
        )

        fig.update_layout(
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Distribution charts
    left, right = st.columns(2)

    with left:

        st.subheader("Customer Tenure Distribution")

        fig = px.histogram(
            df,
            x="tenure",
            nbins=18,
            title="Customers by Tenure",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Monthly Charges Distribution")

        fig = px.histogram(
            df,
            x="MonthlyCharges",
            nbins=25,
            title="Monthly Charges",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Dataset preview
    st.subheader("Customer Dataset Preview")

    st.dataframe(
        df.drop(columns=["Churn_Flag"]).head(10),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# CUSTOMER PREDICTION PAGE
# ==================================================

elif page == "Customer Prediction":

    st.markdown("""
    <div class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>Enter customer details and compare four trained ML models.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):

        x1, x2, x3 = st.columns(3)

        with x1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (months)",
                0,
                72,
                12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with x2:

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with x3:

            protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=500.0,
            value=70.0
        )

        submitted = st.form_submit_button(
            "Predict Churn",
            type="primary",
            use_container_width=True
        )

    # Run prediction
    if submitted:

        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly
        }

        try:

            X = prepare_input(
                customer,
                feature_columns,
                imputer,
                scaler
            )

            predictions = {
                name: get_probability(model, X)
                for name, model in models.items()
            }

            st.divider()

            st.subheader("Model Prediction Results")

            cols = st.columns(len(predictions))

            for col, (name, probability) in zip(
                cols,
                predictions.items()
            ):

                col.metric(
                    name,
                    f"{probability * 100:.1f}%"
                )

                if probability >= 0.5:
                    col.error("High churn risk")
                else:
                    col.success("Low churn risk")

            # Average probability
            average = float(
                np.mean(list(predictions.values()))
            )

            st.divider()

            st.metric(
                "Average Churn Probability",
                f"{average * 100:.1f}%"
            )

            if average >= 0.5:

                st.error(
                    "Customer is predicted to be at risk of churn."
                )

            else:

                st.success(
                    "Customer is more likely to stay."
                )

            # Prediction comparison chart
            chart_data = pd.DataFrame({
                "Model": list(predictions.keys()),
                "Churn Probability (%)": [
                    p * 100
                    for p in predictions.values()
                ]
            })

            fig = px.bar(
                chart_data,
                x="Model",
                y="Churn Probability (%)",
                title="Model-wise Churn Probability",
                range_y=[0, 100],
                template="plotly_dark",
                text_auto=".1f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Total Charges is estimated as tenure × monthly charges. "
                "Prediction reliability depends on matching the original "
                "model training preprocessing."
            )

        except Exception as error:

            st.error(
                "Prediction failed. Check preprocessing compatibility."
            )

            st.exception(error)


# ==================================================
# ABOUT PROJECT PAGE
# ==================================================

else:

    st.markdown("""
    <div class="hero">
        <h1>About This Project</h1>
        <p>Telecom Customer Churn Prediction using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Objective")

    st.write(
        "Identify customers who may discontinue telecom services "
        "so that retention actions can be taken proactively."
    )

    st.divider()

    st.subheader("Dataset Information")

    st.write(
        f"Dataset contains {len(df):,} customer records "
        f"and {df.shape[1] - 1} original columns."
    )

    st.divider()

    st.subheader("Machine Learning Models")

    for name in models:
        st.write("• " + name)

    st.divider()

    st.subheader("Technology Stack")

    st.write(
        "Python, Pandas, NumPy, Scikit-learn, "
        "XGBoost, LightGBM, Plotly and Streamlit."
    )

    st.divider()

    st.subheader("Business Value")

    st.write(
        "Churn insights help prioritize customer outreach, "
        "improve retention campaigns and reduce customer loss."
    )
