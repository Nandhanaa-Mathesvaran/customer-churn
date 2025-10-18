import streamlit as st
import pandas as pd
import pickle

with open('rf.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction App")
st.write("This app predicts whether a customer will churn based on input features.")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(input_df.head())

    if st.button("Predict Churn"):
        predictions = model.predict(input_df)
        input_df["Predicted_Churn"] = predictions
        st.write("Prediction Results:")
        st.dataframe(input_df[["Predicted_Churn"]])
        churn_count = sum(predictions)
        st.write(f"Total Customers Predicted to Churn: {churn_count}")
else:
    st.info("Please upload a CSV file to make predictions.")
