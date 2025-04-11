import streamlit as st
import joblib
import numpy as np

# Load trained model and imputer
model = joblib.load("house_price_model.pkl")
imputer = joblib.load("imputer.pkl")

# Streamlit App UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

st.markdown("Enter house details below to get the estimated selling price.")

# Input fields
overall_qual = st.slider("Overall Quality (1–10)", min_value=1, max_value=10, value=5)
gr_liv_area = st.number_input("Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1500)
garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=400)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=1000)

# Prepare input
input_data = np.array([[overall_qual, gr_liv_area, garage_area, total_bsmt_sf]])
input_data = imputer.transform(input_data)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹{prediction:,.2f}")
