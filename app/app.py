import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("../models/credit_model.pkl")

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1 style='text-align:center;'>Credit Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray;'>Input & Output</h4>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# -----------------------------
# INPUT SECTION
# -----------------------------
with col1:
    st.markdown("## 📥 INPUT DATA")

    amount = st.number_input("Transaction Amount", value=100.0)

    v_features = []
    for i in range(1, 29):
        val = st.slider(f"V{i}", -5.0, 5.0, 0.0)
        v_features.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Risk"):

    try:
        # Input data
        input_data = np.array([[amount] + v_features])

        # Prediction
        prob = model.predict_proba(input_data)[0][1]

        # Risk category
        if prob < 0.3:
            risk = "Low Risk"
            color = "green"
        elif prob < 0.7:
            risk = "Medium Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"

        # -----------------------------
        # SHAP EXPLANATION (FINAL FIX)
        # -----------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        feature_names = ["Amount"] + [f"V{i}" for i in range(1, 29)]

        # Handle SHAP output safely
        if isinstance(shap_values, list):
            impact_values = shap_values[1][0]
        else:
            impact_values = shap_values[0]

        # Convert to 1D
        impact_values = np.array(impact_values).flatten()

        # 🔥 FORCE SAME LENGTH (FINAL FIX)
        min_len = min(len(feature_names), len(impact_values))

        feature_names = feature_names[:min_len]
        impact_values = impact_values[:min_len]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": impact_values
        })

        top_features = shap_df.reindex(
            shap_df.Impact.abs().sort_values(ascending=False).index
        ).head(3)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        with col2:
            st.markdown("## 📊 PREDICTED OUTPUT")

            st.markdown("### Risk Score")
            st.markdown(f"<h1 style='color:{color}'>{prob:.2f}</h1>", unsafe_allow_html=True)
            st.progress(float(prob))

            st.markdown("### Risk Level")
            st.markdown(f"<h2 style='color:{color}'>{risk}</h2>", unsafe_allow_html=True)

            st.markdown("### 🔍 Key Factors")
            for feature in top_features["Feature"]:
                st.write(f"• {feature}")

    except Exception as e:
        st.error(f"Error: {e}")