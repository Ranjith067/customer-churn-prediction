import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")

model = load_model()

# ── Title ────────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.markdown("Enter customer details below to predict whether they are likely to churn.")

# ── Input form ───────────────────────────────────────────────
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")
    Partner = st.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    Contract = st.selectbox("Contract Type",
                             ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    MonthlyCharges = st.number_input("Monthly Charges ($)", 10.0, 120.0, 65.0)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                    float(tenure * MonthlyCharges))
    MultipleLines = st.selectbox("Multiple Lines",
                                  ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service",
                                    ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security",
                                   ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup",
                                 ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection",
                                     ["No", "Yes", "No internet service"])

st.subheader("Additional Services")
col3, col4 = st.columns(2)
with col3:
    TechSupport = st.selectbox("Tech Support",
                                ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV",
                                ["No", "Yes", "No internet service"])
with col4:
    StreamingMovies = st.selectbox("Streaming Movies",
                                    ["No", "Yes", "No internet service"])

# ── Build feature row ─────────────────────────────────────────
def build_features():
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    addon_vals = [OnlineSecurity, OnlineBackup, DeviceProtection,
                  TechSupport, StreamingTV, StreamingMovies]
    num_services = sum(1 for v in addon_vals if v == "Yes")

    has_security_or_support = int(
        OnlineSecurity == "Yes" or TechSupport == "Yes"
    )
    auto_payment = int("automatic" in PaymentMethod)

    if tenure <= 12:
        tenure_group = "0-12"
    elif tenure <= 24:
        tenure_group = "13-24"
    elif tenure <= 48:
        tenure_group = "25-48"
    else:
        tenure_group = "49-72"

    return pd.DataFrame([{
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "num_services": num_services,
        "has_security_or_support": has_security_or_support,
        "auto_payment": auto_payment,
        "tenure_group": tenure_group,
    }])

# ── Predict button ────────────────────────────────────────────
if st.button("Predict Churn", type="primary"):
    features = build_features()
    proba = model.predict_proba(features)[0][1]
    prediction = int(proba >= 0.35)

    if proba >= 0.60:
        risk = "🔴 High Risk"
        color = "red"
    elif proba >= 0.30:
        risk = "🟡 Medium Risk"
        color = "orange"
    else:
        risk = "🟢 Low Risk"
        color = "green"

    st.divider()
    st.subheader("Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", "Will Churn" if prediction else "Will Stay")
    col_b.metric("Churn Probability", f"{proba * 100:.1f}%")
    col_c.metric("Risk Category", risk)

    st.divider()
    st.subheader("Top Factors")

    try:
        prep = model.named_steps["prep"]
        xgb_model = model.named_steps["model"]
        X_t = prep.transform(features)
        names = [n.replace("num__", "").replace("cat__", "")
                 for n in prep.get_feature_names_out()]
        X_df = pd.DataFrame(X_t, columns=names)

        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer(X_df)
        shap_df = pd.DataFrame({
            "Feature": names,
            "Impact": sv.values[0]
        }).sort_values("Impact", key=abs, ascending=False).head(5)

        for _, row in shap_df.iterrows():
            direction = "pushes toward churn" if row["Impact"] > 0 else "reduces churn risk"
            st.write(f"**{row['Feature']}** → {direction} "
                     f"(impact: {row['Impact']:+.2f})")
    except Exception:
        st.info("Feature explanation unavailable for this input.")

    st.divider()
    st.caption("Model: XGBoost tuned pipeline. Cutoff: 0.35. "
               "Results are probabilistic, not certain.")