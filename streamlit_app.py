import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Load the feature list
features = joblib.load("features.pkl")

# Title
st.set_page_config(page_title="🏡 House Price Predictor")
st.title("🏡 House Price Prediction App")
st.markdown("Enter the house details below to predict the **Sale Price**.")

# Create input widgets for each feature
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", step=1.0)

# Create dataframe for prediction
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Sale Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
