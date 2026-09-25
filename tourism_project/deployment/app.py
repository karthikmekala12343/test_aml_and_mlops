import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🧳")
st.title("🧳 Test Wellness Tourism Package — Purchase Predictor")

# Dynamically find the folder where this app.py script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

st.markdown("Enter customer details to predict purchase likelihood.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    number_of_trips = st.number_input("Number of Trips (avg/year)", 1, 30, 3)

with col2:
    passport = st.selectbox("Passport", [0, 1])
    own_car = st.selectbox("Owns Car", [0, 1])
    number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 10, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 10000, 200000, 30000, step=1000)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    number_of_followups = st.number_input("Number of Followups", 1, 10, 3)
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", 5, 120, 15)

# Added CustomerID just in case the model expects it
input_df = pd.DataFrame([{
    "Unnamed: 0": 0, 
    "CustomerID": 1,
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch,
}])

if st.button("🔮 Predict"):
    try:
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.success(f"✅ Likely to purchase (probability: {proba:.2%})")
        else:
            st.warning(f"❌ Unlikely to purchase (probability: {proba:.2%})")
        st.subheader("Input Summary")
        st.dataframe(input_df.T)
        
    except ValueError as e:
        # This will print the EXACT missing column name on the screen
        st.error(f"VALUE ERROR: {e}")
