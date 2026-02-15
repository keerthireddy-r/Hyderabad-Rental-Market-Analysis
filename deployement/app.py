import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("models/rental_price_pipeline.pkl")

# Load data for dropdowns
df = pd.read_csv("data/clean_data.csv")

locality_list = sorted(df["Locality"].unique())
unit_type_list = sorted(df["Unit_Type"].unique())
property_category_list = sorted(df["Property_Category"].unique())
furnishing_status_list = sorted(df["Furnishing_Status"].unique())

st.set_page_config(page_title="Hyderabad House Rent Predictor", layout="centered")

st.title("🏠 Hyderabad House Rent Prediction App")
st.write("Predict monthly house rent using ML")

st.markdown("---")

# Inputs
locality = st.selectbox("Locality", locality_list)
area_sqft = st.number_input("Area (sqft)", 200, 10000, 1000)
no_of_rooms = st.selectbox("Number of Rooms", [1, 2, 3, 4, 5])

unit_type = st.selectbox("Unit Type", unit_type_list)
property_category = st.selectbox("Property Category", property_category_list)
furnishing_status = st.selectbox("Furnishing Status", furnishing_status_list)

has_pool = st.selectbox("Has Pool", ["Yes", "No"])
has_gym = st.selectbox("Has Gym", ["Yes", "No"])
has_parking = st.selectbox("Has Parking", ["Yes", "No"])
has_lift = st.selectbox("Has Lift", ["Yes", "No"])
power_backup = st.selectbox("Power Backup", ["Yes", "No"])
kids_play_area = st.selectbox("Kids Play Area", ["Yes", "No"])
close_to_hospital = st.selectbox("Close to Hospital", ["Yes", "No"])

price_per_sqft = st.number_input("Price per sqft", 10, 500, 50)

# Predict
if st.button("Predict Rent 💰"):

    input_data = pd.DataFrame({
        "No_of_Rooms": [float(no_of_rooms)],
        "Unit_Type": [unit_type],
        "Property_Category": [property_category],
        "Locality": [locality],
        "Area_sqft": [area_sqft],
        "Furnishing_Status": [furnishing_status],
        "Has_Pool": [has_pool],
        "Has_Gym": [has_gym],
        "Has_Parking": [has_parking],
        "Has_Lift": [has_lift],
        "Close_to_Hospital": [close_to_hospital],
        "Power_Backup": [power_backup],
        "Kids_Play_Area": [kids_play_area],
        "Price_per_sqft": [price_per_sqft]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"🏷️ Estimated Monthly Rent: ₹ {int(prediction):,}")



