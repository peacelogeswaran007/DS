import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('best_model.joblib')

# Streamlit app UI
st.title("🏠 House Price Prediction App")

st.markdown("Enter the house features below to get a predicted sale price.")

# Input fields
first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=0)
second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=0)
garage_area = st.number_input("Garage Area (sq ft)", min_value=0)
lot_area = st.number_input("Lot Area (sq ft)", min_value=0)

overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
full_bath = st.slider("Number of Full Bathrooms", 0, 4, 1)
bedrooms = st.slider("Number of Bedrooms Above Ground", 0, 10, 3)

year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
year_remod = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000)
mo_sold = st.slider("Month Sold", 1, 12, 6)
yr_sold = st.number_input("Year Sold", min_value=2006, max_value=2025, value=2010)

# Prediction
if st.button("Predict"):
    try:
        input_features = np.array([
            first_flr_sf, second_flr_sf, total_bsmt_sf, gr_liv_area, garage_area,
            lot_area, overall_qual, full_bath, bedrooms,
            year_built, year_remod, mo_sold, yr_sold
        ]).reshape(1, -1)

        prediction = model.predict(input_features)[0]

        st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
