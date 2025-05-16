import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and features
model = joblib.load("best_model.pkl")
features = joblib.load("features.pkl")

# App title
st.title("🏡 House Price Predictor")

# Create user inputs dynamically
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

# Predict button
if st.button("Predict"):
    if all(value == 0 for value in input_data.values()):
        st.warning("⚠️ Please enter meaningful (non-zero) values for prediction.")
    else:
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        # Reverse log transformation if model was trained on log(SalePrice)
        if prediction < 0:  # crude check – you can refine this
            st.warning("⚠️ Model might be predicting on a log scale. Applying exponential transformation.")
            prediction = np.expm1(prediction)

        if prediction < 0:
            st.error(f"❌ Predicted price is negative: ${prediction:,.2f}. Please check your inputs.")
        else:
            st.success(f"💰 Predicted Sale Price: ${prediction:,.2f}")
