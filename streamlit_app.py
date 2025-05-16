import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the feature list
with open('features.pkl', 'rb') as f:
    model_features = pickle.load(f)

# Set up the Streamlit interface
st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Predictor")
st.write("Enter the values below to predict the house sale price.")

# Generate input fields for all required features
user_input = {}
for feature in model_features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)

# When the Predict button is clicked
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # Predict using the model
    predicted = model.predict(input_df)[0]

    # Handle log-transformed target variable
    if predicted < 100:  # common range for log-prices (e.g. log1p)
        predicted = np.expm1(predicted)  # reverse log1p

    # Show result
    if predicted < 0:
        st.error(f"❌ Predicted price is negative: ${predicted:,.2f}. Please check your inputs.")
    else:
        st.success(f"💰 Predicted Sale Price: ${predicted:,.2f}")
