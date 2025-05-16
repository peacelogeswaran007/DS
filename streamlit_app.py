import streamlit as st
import numpy as np
import joblib

# Load the trained model (ensure this matches your saved joblib file name)
model = joblib.load("best_model.joblib")

# Page title
st.title("🏠 House Price Prediction App")
st.write("Enter the features below to predict house price.")

# Input fields
features = {}

features['1stFlrSF'] = st.number_input("1st Floor SF", min_value=0.0)
features['2ndFlrSF'] = st.number_input("2nd Floor SF", min_value=0.0)
features['TotalBsmtSF'] = st.number_input("Total Basement SF", min_value=0.0)
features['GrLivArea'] = st.number_input("Ground Living Area", min_value=0.0)
features['GarageArea'] = st.number_input("Garage Area", min_value=0.0)
features['LotArea'] = st.number_input("Lot Area", min_value=0.0)
features['OverallQual'] = st.slider("Overall Quality (1-10)", 1, 10, 5)
features['FullBath'] = st.slider("Full Bathrooms", 0, 4, 1)
features['BedroomAbvGr'] = st.slider("Bedrooms Above Ground", 0, 10, 3)
features['YearBuilt'] = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
features['YearRemodAdd'] = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000)
features['MoSold'] = st.slider("Month Sold", 1, 12, 6)
features['YrSold'] = st.number_input("Year Sold", min_value=2006, max_value=2025, value=2010)

# Convert inputs to numpy array
input_array = np.array([list(features.values())])

# Predict
if st.button("Predict"):
    if np.all(input_array == 0):
        st.warning("⚠️ Please enter meaningful (non-zero) values for prediction.")
    else:
        try:
            # Predict and apply inverse transformation if necessary
            prediction = model.predict(input_array)[0]

            # If model was trained on log prices, apply exp
            if prediction < 0:
                st.error("❌ Predicted price is negative. Please check your inputs or model.")
            else:
                predicted_price = np.expm1(prediction) if prediction < 1000 else prediction
                st.success(f"🏡 Predicted House Price: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
