import streamlit as st
import requests

API_URL = "https://implementez-un-modele-de-scoring.onrender.com/predict/"

id_client = st.number_input("Enter client id :", min_value=0, step=1)

threshold_default = 20
if st.button("Predict"):
    if id_client:
        # Appel de l'API
        try:
            response = requests.post(API_URL, json={"id_client": id_client})

            if response.status_code == 200:
                result = response.json()
                if result['probability'] * 100 <= threshold_default:
                    st.success(f"Result for client number: {result['id_client']} : Loan Accepted")
                else:
                    st.error(f"Result for client number: {result['id_client']} : Loan Declined")

                st.write(f"Certainty of default payment: {round(result['probability'] * 100, 2)}%")
                st.write(f"Maximum allowed is set at : {threshold_default} %")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown Problem')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Please enter a valid ID")
    else:
        st.warning("Please enter a valid ID")
