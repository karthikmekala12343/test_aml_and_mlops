import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration & Custom CSS ---
st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🧳", layout="wide")

# Inject custom CSS for a better look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧳 Wellness Tourism Package Predictor")
st.markdown("Enter the customer's details below to predict the likelihood of them purchasing the package.")

# --- Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"Model file not found at: {MODEL_PATH}. Please ensure the GitHub Action has completed successfully.")
    st.stop()

# --- UI Layout ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Customer Profile")
    age = st.number_input("Age", 18, 100, 35)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    monthly_income = st.number_input("Monthly Income ($)", 10000, 200000, 30000, step=1000)

with col2:
    st.subheader("🛫 Travel & Trip Details")
    type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_trips = st.number_input("Number of Trips (avg/year)", 1, 30, 3)
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 10, 1)

with col3:
    st.subheader("📊 Pitch & Designation")
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    number_of_followups = st.number_input("Number of Followups", 1, 10, 3)
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", 5, 120, 15)

st.markdown("---")

# --- Prediction Logic ---
if st.button("🔮 Predict Purchase Likelihood"):
    input_df = pd.DataFrame([{
        "Unnamed: 0": 0,  # Kept for compatibility with the trained pipeline
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

    try:
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        # Display metrics and progress
        m1, m2 = st.columns([1, 2])

        with m1:
            st.metric(label="Prediction Result", value="Will Purchase ✅" if pred == 1 else "Will Not Purchase ❌")

        with m2:
            st.write("**Purchase Probability:**")
            st.progress(int(proba * 100))
            st.write(f"**{proba:.2%}** confidence based on model inputs.")

        with st.expander("View Input Data Summary"):
            st.dataframe(input_df.T, use_container_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
