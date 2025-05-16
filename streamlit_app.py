import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

# Define the feature list in the exact order the model was trained on
features = [
    '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch',
    'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GarageYrBlt',
    'GrLivArea', 'HalfBath', 'Id', 'KitchenAbvGr', 'LotArea', 'LotFrontage',
    'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'MoSold', 'MSSubClass',
    'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch',
    'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd',
    'YrSold'
]

st.set_page_config(page_title="🏡 House Price Predictor")
st.title("🏡 House Price Predictor")

st.write("Enter house details to estimate its price:")

# Create input fields
input_data = []
for feature in features:
    val = st.number_input(f"{feature}", step=1)
    input_data.append(val)

# When user clicks "Predict Price"
if st.button("Predict Price"):
    try:
        # Convert input into DataFrame with correct column names
        input_df = pd.DataFrame([input_data], columns=features)

        # Predict the price
        prediction = model.predict(input_df)[0]
        st.success(f"🏠 Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
