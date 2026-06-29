import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# -------------------- Load Model & Encoders --------------------

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('standard_scaler.pkl', 'rb') as file:
        standard_scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, standard_scaler


model, label_encoder_gender, onehot_encoder_geo, standard_scaler = load_artifacts()

# -------------------- Title --------------------

st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the bank.")

# -------------------- Input Fields --------------------

col1, col2 = st.columns(2)

with col1:

    geography = st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "Age",
        18,
        92,
        35
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=600
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

with col2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

    num_of_products = st.slider(
        "Number of Products",
        1,
        4,
        2
    )

    has_cr_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    is_active_member = st.selectbox(
        "Is Active Member",
        ["Yes", "No"]
    )

has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# -------------------- Prediction --------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Fixes the OneHotEncoder warning
    geo_df = pd.DataFrame({'Geography': [geography]})

    geo_encoded = onehot_encoder_geo.transform(geo_df)

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    input_scaled = standard_scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    if probability > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")