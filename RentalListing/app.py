import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Keep the model file in the same folder as this app.py file.
MODEL_PATH = Path(__file__).parent / "rental_final_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="RentHop Interest Predictor", page_icon="🏠")

st.title("🏠 Rental Listing Interest Predictor")
st.caption("Educational demonstration using the RentHop rental-listing dataset.")

bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
price = st.number_input("Monthly Price", min_value=0, max_value=100000, value=3000, step=100)
latitude = st.number_input("Latitude", value=40.75, format="%.6f")
longitude = st.number_input("Longitude", value=-73.98, format="%.6f")
feature_count = st.number_input("Number of listed features", min_value=0, max_value=50, value=8, step=1)
photo_count = st.number_input("Number of photos", min_value=0, max_value=100, value=6, step=1)
listing_hour = st.number_input("Listing hour (0–23)", min_value=0, max_value=23, value=12, step=1)
listing_dayofweek = st.number_input("Day of week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2, step=1)

description = st.text_area(
    "Rental description",
    "Beautiful renovated apartment with hardwood floors, elevator, laundry and subway access."
)

feature_text = st.text_area(
    "Rental features",
    "Doorman, elevator, hardwood floors, laundry"
)

if st.button("Predict Interest Level"):
    # IMPORTANT: The trained model expects the column name 'features_text'
    # (plural), not 'feature_text' (singular).
    row = pd.DataFrame([{
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "price": price,
        "latitude": latitude,
        "longitude": longitude,
        "feature_count": feature_count,
        "photo_count": photo_count,
        "listing_hour": listing_hour,
        "listing_dayofweek": listing_dayofweek,
        "features_text": feature_text
    }])

    try:
        prediction = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]

        st.success(f"Predicted interest level: {str(prediction).upper()}")

        probability_df = pd.DataFrame({
            "Interest Level": model.classes_,
            "Probability": [f"{p:.2%}" for p in probabilities]
        })
        st.table(probability_df)

    except Exception as e:
        st.error("The model could not make a prediction with these inputs.")
        st.exception(e)

st.info(
    "Educational demonstration only. Predictions are estimates and should not "
    "be treated as guaranteed market outcomes."
)
