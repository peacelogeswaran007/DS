import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below to predict its price.")

# Input fields
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=0)
GarageCars = st.selectbox("Garage Capacity (Number of Cars)", options=[0, 1, 2, 3, 4])
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[GrLivArea, GarageCars, TotalBsmtSF]])
    prediction = model.predict(input_data)
    predicted_price = np.expm1(prediction[0])  # Inverse of log1p if log was used
    st.success(f"🏷️ Estimated House Price: ${predicted_price:,.2f}")
