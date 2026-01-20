import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Agriculture DSS",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------
# Custom Dark Theme Styling
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
h1, h2, h3, h4 {
    color: #ffffff;
}
p, label {
    color: #d1d5db;
}
div[data-testid="stMetric"] {
    background-color: #1f2937;
    padding: 16px;
    border-radius: 12px;
}
.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}
.stDataFrame {
    background-color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Models & Data
# --------------------------------------------------
reg_model = joblib.load("profit_model.pkl")
clf_model = joblib.load("profit_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

data = pd.read_csv("merged_data.csv")

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
# 🌾 Smart Agriculture Decision Support System
### Crop recommendation based on climate, area, profit, and risk
""")

# --------------------------------------------------
# Input Section (VERTICAL LAYOUT)
# --------------------------------------------------
st.markdown("## 📌 Enter Farm & Climate Details")

temp_input = st.number_input(
    "Temperature (°C)",
    min_value=10,
    max_value=45,
    value=22
)

rain_input = st.number_input(
    "Rainfall (mm)",
    min_value=100,
    max_value=3000,
    value=140
)

area_input = st.number_input(
    "Area (hectares)",
    min_value=1.0,
    max_value=20.0,
    value=9.0
)

risk_pref = st.selectbox(
    "Risk Preference",
    ["Low", "Medium", "High"]
)

recommend_button = st.button("✅ Recommend Crop")

# --------------------------------------------------
# DSS Helper Functions
# --------------------------------------------------
def temp_score(user_temp, crop_temp):
    return max(0, 1 - abs(user_temp - crop_temp) / 15)

def rain_score(user_rain, crop_rain):
    return max(0, 1 - abs(user_rain - crop_rain) / 2000)

def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

# --------------------------------------------------
# Recommendation Logic
# --------------------------------------------------
if recommend_button:

    # -------------------------------
    # ML Predictions
    # -------------------------------
    X = data[
        ["Temperature_C", "Rainfall_mm", "Yield", "Total_Cost_INR", "Area_Hectare"]
    ]

    data["Expected_Profit"] = reg_model.predict(X)
    data["Profit_Category"] = label_encoder.inverse_transform(
        clf_model.predict(X)
    )

    # -------------------------------
    # DSS Scoring
    # -------------------------------
    data["Temp_score"] = data["Temperature_C"].apply(
        lambda x: temp_score(temp_input, x)
    )
    data["Rain_score"] = data["Rainfall_mm"].apply(
        lambda x: rain_score(rain_input, x)
    )

    data["Profit_Score"] = normalize(data["Expected_Profit"])
    data["Yield_Score"] = normalize(data["Yield"])

    data["Suitability_Score"] = (
        0.25 * data["Temp_score"] +
        0.25 * data["Rain_score"] +
        0.25 * data["Yield_Score"] +
        0.25 * data["Profit_Score"]
    )

    # -------------------------------
    # Risk Preference Filtering
    # -------------------------------
    if risk_pref == "Low":
        data_filtered = data[data["Risk_Level"] == "Low"]
    elif risk_pref == "Medium":
        data_filtered = data[data["Risk_Level"].isin(["Low", "Medium"])]
    else:
        data_filtered = data.copy()

    # --------------------------------------------------
    # UNIQUE TOP CROPS (NO DUPLICATES)
    # --------------------------------------------------
    top_crops = (
        data_filtered
        .sort_values("Suitability_Score", ascending=False)
        .groupby("Crop", as_index=False)
        .first()
        .head(3)   # show at least 3 different crops
    )

    # --------------------------------------------------
    # Output Table
    # --------------------------------------------------
    st.markdown("## ✅ Top Recommended Crops")

    st.dataframe(
        top_crops[["Crop", "Expected_Profit", "Risk_Level"]]
        .style.format({"Expected_Profit": "{:,.2f}"})
    )

    # --------------------------------------------------
    # Decision Insight (UNCHANGED)
    # --------------------------------------------------
    st.markdown("## 📊 Decision Insight")

    best = top_crops.iloc[0]

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("🌾 Best Crop", best["Crop"])

    with colB:
        st.metric("💰 Expected Profit", f"₹{best['Expected_Profit']:,.0f}")

    with colC:
        st.metric("⚠️ Risk Level", best["Risk_Level"])
