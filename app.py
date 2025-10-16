import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('rf.pkl', 'rb'))

st.title("Telco Customer Churn Prediction App")
st.write("Enter customer details to predict if they are likely to churn.")

# Input fields (must match training feature order)
gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", [0, 1])
dependents = st.selectbox("Has Dependents", [0, 1])
tenure = st.number_input("Tenure (months)", min_value=0)
phone_service = st.selectbox("Has Phone Service", [0, 1])
paperless_billing = st.selectbox("Paperless Billing", [0, 1])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):
    # Arrange features in the same order as during training
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
