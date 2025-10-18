import streamlit as st
import pandas as pd
import pickle

with open('rf.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on key details.")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
dependents = st.selectbox("Dependents", [0, 1])
phone_service = st.selectbox("Phone Service", [0, 1])
paperless_billing = st.selectbox("Paperless Billing", [0, 1])

input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'PaperlessBilling': [paperless_billing]
})

for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model.feature_names_in_]

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.write(f"The customer is likely to churn. Probability: {probability:.2f}")
    else:
        st.write(f"The customer is likely to stay. Probability: {probability:.2f}")
