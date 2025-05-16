import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("best_model.pkl")

st.title("🏡 House Price Predictor")

# Input fields
st.slider("Overall Quality", 1, 10, 5)
GrLivArea = st.number_input("Above Grade Living Area (sqft)", 500, 5000, 1500)
GarageCars = st.slider("Garage Cars", 0, 4, 2)
TotalBsmtSF = st.number_input("Total Basement Area (sqft)", 0, 3000, 800)
FullBath = st.slider("Full Bathrooms", 0, 4, 2)

if st.button("Predict Price"):
    # Construct the input DataFrame based on training features
    input_df = pd.DataFrame([{
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "FullBath": FullBath,
        "OverallQual": 5  # you can also get this from slider above
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"🏠 Estimated House Price: ${prediction:,.0f}")
