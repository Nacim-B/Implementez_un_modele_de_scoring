import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict/"

# Titre du tableau de bord
st.title("Prediction Credit Default")

id_client = st.number_input("Enter client id :", min_value=0, step=1)

if st.button("Predict"):
    if id_client:
        # Appel de l'API
        try:
            response = requests.post(API_URL, json={"id_client": id_client})

            if response.status_code == 200:
                result = response.json()
                if result['prediction'] == 1:
                    label_result = "Loan Declined"
                else:
                    label_result = "Loan Accepted"
                st.success(f"Prediction for client number: {result['id_client']} : {label_result}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown Problem')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Please enter a valid ID")
    else:
        st.warning("Please enter a valid ID")
