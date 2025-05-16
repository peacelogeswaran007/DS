import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("best_model.pkl")

# Page title
st.set_page_config(page_title="🏡 House Price Predictor")
st.title("🏡 House Price Predictor")

st.write("Fill in the details below to predict the house price:")

# Define all features your model expects
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

# Create input fields for each feature
input_data = {}
for feature in features:
    if feature in ['Id', 'MSSubClass', 'YrSold', 'MoSold', 'OverallQual', 'OverallCond', 'KitchenAbvGr', 'TotRmsAbvGrd', 'BedroomAbvGr',
                   'Fireplaces', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:
        input_data[feature] = st.number_input(feature, step=1)
    else:
        input_data[feature] = st.number_input(feature)

# Prediction
if st.button("Predict Price"):
    try:
        # Create DataFrame with single row in same order as features
        input_df = pd.DataFrame([input_data[f] for f in features]).T
        input_df.columns = features

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"🏠 Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
