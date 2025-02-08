import streamlit as st
import requests

API_URL = "http://localhost:8000/predict_new_client"

st.title("New Client Loan Application")

# Input fields for new client
amt_goods_price = st.number_input("Goods Price (AMT_GOODS_PRICE):", min_value=0.0, value=500000.0)
income_per_person = st.number_input("Income per Person (INCOME_PER_PERSON):", min_value=0.0, value=100000.0)
amt_annuity = st.number_input("Loan Annuity Amount (AMT_ANNUITY):", min_value=0.0, value=24000.0)
age_years = st.number_input("Age (years):", min_value=18, max_value=100, value=35)
days_birth = -age_years * 365  # Convert age to negative days

if st.button("Get Prediction"):
    try:
        payload = {
            "amt_goods_price": amt_goods_price,
            "income_per_person": income_per_person,
            "amt_annuity": amt_annuity,
            "days_birth": days_birth
        }
        
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            probability = result["prediction_probability"]
            st.write(f"Prediction Probability: {probability:.2%}")
            
            if probability < 0.4:
                st.success("Accept the loan")
            elif 0.4 <= probability < 0.5:
                st.warning("Check with 'un conseiller' for further info")
            else:
                st.error("Decline the loan")
            
            # Display input features used
            st.subheader("Input Features Used:")
            st.json(result["input_features"])
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown Problem')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
