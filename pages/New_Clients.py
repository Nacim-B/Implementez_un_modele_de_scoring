import streamlit as st
import requests

API_URL = "https://implementez-un-modele-de-scoring.onrender.com/predict_new_client/"

new_client_data = {'loan_request_amount': st.number_input("Amount Loan Request:", min_value=0,
                                                          max_value=1000000000, step=5000),
                   'annual_salary': st.number_input("Annual Income :", min_value=0, max_value=50000000, step=5000),
                   'annual_annuity': st.number_input("Amount Annuity:", min_value=0, max_value=300000, step=1000),
                   'age': st.number_input("Age :", min_value=18, max_value=150, step=1) * (-365)
                   }
threshold_default = 20
if st.button("Predict"):
    try:
        response = requests.post(API_URL, json=new_client_data)

        if response.status_code == 200:
            result = response.json()
            if result['probability'] * 100 <= threshold_default:
                st.success(f"Result for new client : Loan Accepted")
            else:
                st.error(f"Result for new client : Loan Declined")

            st.write(f"Certainty of default payment: {round(result['probability'] * 100, 2)}%")
            st.write(f"Maximum allowed is set at : {threshold_default} %")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown Problem')}")
    except requests.exceptions.RequestException as e:
        st.error(e)

