import streamlit as st
import pandas as pd
import pickle

with open('random_forest_model.pkl', 'rb') as file:
    rf = pickle.load(file)

st.title("Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn based on input features.")

st.sidebar.header("Enter Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("SeniorCitizen", [0, 1])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.sidebar.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=1000.0)
    internet_service = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.sidebar.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService': internet_service,
        'Contract': contract,
        'PaymentMethod': payment_method
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_processed = pd.get_dummies(input_df)
input_processed = input_processed.reindex(columns=rf.feature_names_in_, fill_value=0)

if st.button("Predict Churn"):
    prediction = rf.predict(input_processed)[0]
    prediction_proba = rf.predict_proba(input_processed)[0][1]
    if prediction == 1:
        st.write(f"The customer is likely to churn. Probability: {prediction_proba:.2f}")
    else:
        st.write(f"The customer is likely to stay. Probability: {prediction_proba:.2f}")
