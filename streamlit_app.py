import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('best_model.pkl')

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.markdown("Enter property details to predict its **Sale Price**.")

# Example input fields (adjust based on your dataset)
# Here we assume numerical features only (like LotArea, YearBuilt, OverallQual, GrLivArea, etc.)
# You MUST match these with the features your model was trained on

def get_user_input():
    LotArea = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=8000)
    YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
    OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GrLivArea = st.number_input("Above ground living area (sq ft)", min_value=300, max_value=5000, value=1500)
    GarageCars = st.slider("Garage Cars", 0, 4, 2)
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=800)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)

    data = {
        "LotArea": LotArea,
        "YearBuilt": YearBuilt,
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "FullBath": FullBath
    }

    return pd.DataFrame([data])

input_df = get_user_input()

# Predict and show result
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)
        st.success(f"💰 Predicted House Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")
