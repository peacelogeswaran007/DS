import streamlit as st
import pandas as pd
import joblib

# Title of the app
st.title("🏡 House Price Predictor")

# Load the trained model and the feature list
model = joblib.load("best_model.pkl")
feature_names = joblib.load("features.pkl")  # list of features used in training

# User input interface (minimal important features)
st.header("Enter Property Details")

lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=200000, value=8000)
year_built = st.slider("Year Built", min_value=1900, max_value=2025, value=2000)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1800)
full_bath = st.slider("Number of Full Bathrooms", 0, 4, 2)
garage_cars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)

# Create a dictionary with the inputs
input_data = {
    'LotArea': lot_area,
    'YearBuilt': year_built,
    'GrLivArea': gr_liv_area,
    'FullBath': full_bath,
    'GarageCars': garage_cars,
    'OverallQual': overall_qual,
}

# Fill in missing features with 0
for feature in feature_names:
    if feature not in input_data:
        input_data[feature] = 0

# Convert to DataFrame and reorder columns to match training data
input_df = pd.DataFrame([input_data])[feature_names]

# Make prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏠 Estimated House Price: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
