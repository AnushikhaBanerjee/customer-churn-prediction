import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* Hide Streamlit top header */
[data-testid="stHeader"] {
    display: none;
}

/* Hide toolbar/menu */
[data-testid="stToolbar"] {
    display: none;
}

/* Remove extra top space */
.block-container {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

    /* -------------------------------
       MAIN PAGE
    ------------------------------- */

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Hide Streamlit top header */
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    /* -------------------------------
       TITLE
    ------------------------------- */

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* -------------------------------
       COLUMN CONTAINERS
    ------------------------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }

    /* -------------------------------
       HEADINGS
    ------------------------------- */

    h3 {
        color: #172554 !important;
        font-weight: 750 !important;
    }

    h4 {
        color: #4f46e5 !important;
    }

    /* -------------------------------
       INPUTS
    ------------------------------- */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-color: #cbd5e1 !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="input"] input {
        color: #1e293b !important;
    }

    /* -------------------------------
       BUTTON
    ------------------------------- */

    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        border: none;
        background-color: #4f46e5;
        color: white;
        font-size: 16px;
        font-weight: 700;
        margin-top: 12px;
    }

    .stButton > button:hover {
        background-color: #4338ca;
        color: white;
    }

    /* -------------------------------
       RESULT AREA
    ------------------------------- */

    .ready-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 35px 20px;
        text-align: center;
        margin-top: 15px;
    }

    .ready-icon {
        font-size: 50px;
    }

    .ready-title {
        font-size: 27px;
        font-weight: 800;
        color: #16a34a;
        margin-top: 10px;
    }

    .ready-text {
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
        margin-top: 8px;
    }

    /* -------------------------------
       INFO BOX
    ------------------------------- */

    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 20px;
        margin-top: 18px;
        color: #1e3a8a;
        line-height: 1.6;
    }

    /* -------------------------------
       RESULT CARDS
    ------------------------------- */

    .prediction-card {
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        margin-top: 15px;
    }

    .churn-card {
        background-color: #fff1f2;
        border: 1px solid #fecdd3;
    }

    .safe-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    .prediction-icon {
        font-size: 52px;
    }

    .churn-title {
        color: #dc2626;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
    }

    .safe-title {
        color: #16a34a;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
    }

    .prediction-text {
        color: #64748b;
        margin-top: 8px;
        line-height: 1.6;
    }

    /* -------------------------------
       METRIC CARDS
    ------------------------------- */

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-top: 10px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
    }

    .metric-value {
        color: #172554;
        font-size: 18px;
        font-weight: 750;
        margin-top: 4px;
    }

    /* -------------------------------
       SECTION DIVIDER
    ------------------------------- */

    .section-line {
        border-bottom: 1px solid #e2e8f0;
        margin: 8px 0 15px 0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter customer details to predict whether the customer is likely to churn.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TWO COLUMNS
# =========================================================

left_col, right_col = st.columns(
    [1.05, 0.95],
    gap="large"
)


# =========================================================
# LEFT COLUMN
# =========================================================

with left_col:

    with st.container(border=True):

        st.markdown("### 👤 Customer Details")

        st.caption("Fill in the customer information below")

        # ---------------------------------------------
        # PERSONAL INFORMATION
        # ---------------------------------------------

        st.markdown("#### 👤 Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col2:
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        col1, col2 = st.columns(2)

        with col1:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        with col2:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        # ---------------------------------------------
        # SERVICES
        # ---------------------------------------------

        st.markdown("#### 💻 Services")

        col1, col2 = st.columns(2)

        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

        with col2:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        col1, col2 = st.columns(2)

        with col1:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with col2:
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

        col1, col2 = st.columns(2)

        with col1:
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        col1, col2 = st.columns(2)

        with col1:
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        # ---------------------------------------------
        # ACCOUNT INFORMATION
        # ---------------------------------------------

        st.markdown("#### 💳 Account Information")

        col1, col2 = st.columns(2)

        with col1:
            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

        with col2:
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
                "Bank transfer (automatic)"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

        with col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=1000.0,
                step=10.0
            )

        # ---------------------------------------------
        # PREDICT BUTTON
        # ---------------------------------------------

        predict_button = st.button(
            "🔍 Predict Churn",
            use_container_width=True
        )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right_col:

    with st.container(border=True):

        st.markdown("### 🎯 Prediction Result")

        st.caption(
            "The machine learning model prediction is shown below"
        )

        # =================================================
        # BEFORE PREDICTION
        # =================================================

        if not predict_button:

            st.markdown(
                '<div class="ready-box">'
                '<div class="ready-icon">🎯</div>'
                '<div class="ready-title">Ready for Prediction</div>'
                '<div class="ready-text">'
                'Enter the customer details on the left and click '
                '<b>Predict Churn</b> to see the machine learning result.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="info-box">'
                '<b>🤖 How it works</b>'
                '<br><br>'
                'The trained machine learning model analyzes the '
                'customer demographic, service, contract, and billing '
                'information to predict the likelihood of churn.'
                '</div>',
                unsafe_allow_html=True
            )


        # =================================================
        # AFTER PREDICTION
        # =================================================

        else:

            # ---------------------------------------------
            # CREATE INPUT DATA
            # ---------------------------------------------

            input_data = pd.DataFrame({
                "SeniorCitizen": [senior_citizen],
                "tenure": [tenure],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
                "gender": [gender],
                "Partner": [partner],
                "Dependents": [dependents],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method]
            })

            # ---------------------------------------------
            # ONE HOT ENCODING
            # ---------------------------------------------

            input_data = pd.get_dummies(
                input_data,
                drop_first=True
            )

            # ---------------------------------------------
            # GET EXACT TRAINING FEATURES
            # ---------------------------------------------

            if hasattr(scaler, "feature_names_in_"):

                feature_names = list(
                    scaler.feature_names_in_
                )

            else:

                feature_names = [
                    "SeniorCitizen",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "gender_Female",
                    "gender_M",
                    "gender_Male",
                    "Partner_Yes",
                    "Dependents_Yes",
                    "PhoneService_Yes",
                    "MultipleLines_No phone service",
                    "MultipleLines_Yes",
                    "InternetService_Fiber optic",
                    "InternetService_No",
                    "OnlineSecurity_No internet service",
                    "OnlineSecurity_Yes",
                    "OnlineBackup_No internet service",
                    "OnlineBackup_Yes",
                    "DeviceProtection_No internet service",
                    "DeviceProtection_Yes",
                    "TechSupport_No internet service",
                    "TechSupport_Yes",
                    "StreamingTV_No internet service",
                    "StreamingTV_Yes",
                    "StreamingMovies_No internet service",
                    "StreamingMovies_Yes",
                    "Contract_One year",
                    "Contract_Two year",
                    "PaperlessBilling_Yes",
                    "PaymentMethod_Credit card (automatic)",
                    "PaymentMethod_Electronic check",
                    "PaymentMethod_Mailed check"
                ]

            # ---------------------------------------------
            # MATCH TRAINING COLUMNS
            # ---------------------------------------------

            input_data = input_data.reindex(
                columns=feature_names,
                fill_value=0
            )

            # ---------------------------------------------
            # SCALE
            # ---------------------------------------------

            input_scaled = scaler.transform(input_data)

            # ---------------------------------------------
            # PREDICT
            # ---------------------------------------------

            prediction = model.predict(
                input_scaled
            )[0]

            # ---------------------------------------------
            # PROBABILITY
            # ---------------------------------------------

            probability = model.predict_proba(
                input_scaled
            )[0]

            no_churn_probability = probability[0] * 100
            churn_probability = probability[1] * 100

            # =================================================
            # RESULT
            # =================================================

            if prediction == 1:

                st.markdown(
                    '<div class="prediction-card churn-card">'
                    '<div class="prediction-icon">⚠️</div>'
                    '<div class="churn-title">'
                    'Customer is likely to CHURN'
                    '</div>'
                    '<div class="prediction-text">'
                    'This customer shows characteristics associated '
                    'with a higher chance of churn.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                prediction_text = "CHURN"
                confidence = churn_probability

            else:

                st.markdown(
                    '<div class="prediction-card safe-card">'
                    '<div class="prediction-icon">✅</div>'
                    '<div class="safe-title">'
                    'Customer is unlikely to CHURN'
                    '</div>'
                    '<div class="prediction-text">'
                    'This customer shows characteristics associated '
                    'with a lower chance of churn.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                prediction_text = "NO CHURN"
                confidence = no_churn_probability

            # =================================================
            # SUMMARY
            # =================================================

            st.markdown("#### 📋 Prediction Summary")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="metric-card">'
                    '<div class="metric-label">Prediction</div>'
                    f'<div class="metric-value">{prediction_text}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    '<div class="metric-card">'
                    '<div class="metric-label">Confidence</div>'
                    f'<div class="metric-value">{confidence:.1f}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="metric-card">'
                    '<div class="metric-label">Tenure</div>'
                    f'<div class="metric-value">{tenure} Months</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    '<div class="metric-card">'
                    '<div class="metric-label">Monthly Charges</div>'
                    f'<div class="metric-value">${monthly_charges:.2f}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            # =================================================
            # PROBABILITY
            # =================================================

            st.markdown("#### 📊 Churn Probability")

            st.progress(
                int(churn_probability)
            )

            st.write(
                f"**Churn:** {churn_probability:.1f}%"
            )

            st.write(
                f"**No Churn:** {no_churn_probability:.1f}%"
            )

            # =================================================
            # EXPLANATION
            # =================================================

            if prediction == 1:

                st.info(
                    "💡 **Retention suggestion:** "
                    "This customer may benefit from personalized "
                    "offers, better support, or suitable discounts "
                    "to reduce the risk of churn."
                )

            else:

                st.success(
                    "💡 **Customer status:** "
                    "The model predicts that this customer is likely "
                    "to continue their service."
                )