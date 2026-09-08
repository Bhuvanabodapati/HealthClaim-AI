from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

BEST_THRESHOLD = 0.35


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Health Claim Prediction",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #666;
    margin-bottom: 30px;
}

.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏥 Health Claim Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the claim details below to predict the claim outcome.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📋 Claim Details</div>',
    unsafe_allow_html=True
)


patient_age = st.number_input(
    "Patient Age",
    min_value=0,
    max_value=120,
    value=35
)


claim_amount = st.number_input(
    "Claim Amount",
    min_value=0.0,
    value=50000.0
)


previous_claims = st.number_input(
    "Previous Claims",
    min_value=0,
    value=1
)


previous_rejections = st.number_input(
    "Previous Rejections",
    min_value=0,
    value=0
)


missing_documents = st.selectbox(
    "Missing Documents",
    ["Yes", "No"]
)


provider_risk = st.selectbox(
    "Provider Risk",
    ["Low", "Medium", "High"]
)


diagnosis = st.selectbox(
    "Diagnosis",
    [
        "Cardiac",
        "Diabetes",
        "Orthopedic",
        "General",
        "Respiratory"
    ]
)


procedure = st.selectbox(
    "Procedure",
    [
        "Diagnostic",
        "Consultation",
        "Hospitalization",
        "Surgery",
        "Therapy"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")


if st.button("🔍 Predict Claim", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "patient_age": [patient_age],
        "claim_amount": [claim_amount],
        "previous_claims": [previous_claims],
        "previous_rejections": [previous_rejections],
        "missing_documents": [missing_documents],
        "provider_risk": [provider_risk],
        "diagnosis": [diagnosis],
        "procedure": [procedure]
    })


    # Apply the same preprocessing used during model training
    input_processed = preprocessor.transform(input_data)

    # Get probability of Class 1
    probability = model.predict_proba(input_processed)[0][1]

    # Apply selected threshold
    prediction = int(probability >= BEST_THRESHOLD)

   


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.success("✅ Claim Approved")

    else:

        st.error("❌ Claim Rejected")


    st.write(
        f"### Prediction Probability: {probability * 100:.2f}%"
    )


    # Progress bar
    st.progress(float(probability))


    # Explanation
    if prediction == 1:

        st.info(
            "The model predicts that this claim is likely to be approved."
        )

    else:

        st.warning(
            "The model predicts that this claim is likely to be rejected."
        )