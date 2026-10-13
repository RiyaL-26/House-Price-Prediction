
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

# ---------------- LOAD SAVED MODEL ----------------

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


st.title("🏠 Bengaluru House Price Predictor")
st.write(
    "Estimate a property's price using your trained "
    "machine-learning model."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Keep app.py and "
        "house_price_model.joblib in the same folder."
    )
    st.stop()

try:
    artifact = load_artifact()
    model = artifact["model"]
    model_name = artifact["model_name"]
    locations = artifact["locations"]
    area_types = artifact["area_types"]
    features = artifact["features"]
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()

expected_features = [
    "location", "area_type", "bhk",
    "total_sqft", "bath", "balcony"
]

if features != expected_features:
    st.error(
        f"Unexpected model features: {features}. "
        "Check your training script."
    )
    st.stop()

st.success(f"Model loaded successfully: {model_name}")

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("About this project")
    st.write("Model:", model_name)
    st.write("Target: House price")
    st.write("Unit: Indian lakh rupees")
    st.caption(
        "Predictions are estimates, not official "
        "property valuations."
    )

# ---------------- INPUT FORM ----------------

st.subheader("Enter property details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox(
            "Location",
            options=locations,
            index=0
        )

        area_type = st.selectbox(
            "Area type",
            options=area_types,
            index=0
        )

        bhk = st.number_input(
            "Bedrooms (BHK)",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

    with col2:
        total_sqft = st.number_input(
            "Total area (sq. ft.)",
            min_value=101.0,
            max_value=19999.0,
            value=1200.0,
            step=50.0
        )

        bath = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        balcony = st.number_input(
            "Balconies",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if submitted:
    input_data = pd.DataFrame(
        [{
            "location": location,
            "area_type": area_type,
            "bhk": int(bhk),
            "total_sqft": float(total_sqft),
            "bath": int(bath),
            "balcony": int(balcony)
        }],
        columns=features
    )

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction result")

        if prediction <= 0:
            st.warning(
                f"The model predicted ₹{prediction:.2f} lakh, "
                "which is not a meaningful positive price. "
                "Try different property details."
            )
        else:
            st.metric(
                label="Estimated house price",
                value=f"₹ {prediction:,.2f} lakh",
                help="1 lakh = ₹1,00,000"
            )

            st.write(
                f"Approximate price in rupees: "
                f"₹ {prediction * 100000:,.0f}"
            )

            st.caption(
                "This is a machine-learning estimate based on "
                "historical Bengaluru housing data. Actual prices "
                "can differ substantially."
            )

        with st.expander("View the submitted details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error("The prediction could not be completed.")
        st.exception(error)
