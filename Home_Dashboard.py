# Import necessary libraries
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import pickle

# Constants
API_URL = "https://implementez-un-modele-de-scoring.onrender.com"
DATASET_PATH = "./data_test_for_dashboard.csv"
MODEL_PATH = "./old_client_model.pkl"


# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(DATASET_PATH)
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return data, pipeline


data, pipeline = load_data()
model_lgbm = pipeline.named_steps['classifier']


# Function to fetch prediction from API
def fetch_prediction(client_id, amt_goods_price=None, amt_annuity=None):
    payload = {"id_client": client_id}
    if amt_goods_price is not None:
        payload["amt_goods_price"] = amt_goods_price
    if amt_annuity is not None:
        payload["amt_annuity"] = amt_annuity
    response = requests.post(f"{API_URL}/predict", json=payload, verify=False)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching data from API")
        return None


# Function to display a message based on the probability score
def display_probability_message(probability):
    if probability < 0.4:
        st.success("Accept the loan")
    elif 0.4 <= probability < 0.5:
        st.warning("Check with 'un conseiller' for further info")
    else:
        st.error("Decline the loan")


# Function to compute global feature importance
@st.cache_data
def global_feature_importance():
    explainer = shap.TreeExplainer(model_lgbm, shap.maskers.Independent(data.drop('SK_ID_CURR', axis=1)))
    return explainer(data.drop('SK_ID_CURR', axis=1)), explainer


shap_globalvalues, feature_explainer = global_feature_importance()


# Function to plot local SHAP values
def plot_local_waterfall_by_id(sk_id_curr, explainer):
    client_data = data[data['SK_ID_CURR'] == sk_id_curr]
    if client_data.empty:
        st.write("Client ID not found.")
        return
    features = client_data.drop(columns=['SK_ID_CURR'])
    shap_values = explainer(features)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False, ax=ax)
    st.pyplot(fig)


# Function to plot income distribution with client marker
def plot_income_distribution_with_marker(data, client_id, income_col):
    fig_income_log = px.histogram(data, x=income_col, title=f'Distribution of {income_col} (Log Scale)', log_x=True)
    client_income = data[data['SK_ID_CURR'] == client_id][income_col].values
    if client_income.size > 0:
        fig_income_log.add_shape(
            type='line',
            x0=client_income[0], x1=client_income[0],
            y0=0, y1=1,
            line=dict(color='Red', width=3),
            xref='x', yref='paper'
        )
    st.plotly_chart(fig_income_log)


# Function to plot horizontal box plot of EXT_SOURCE
def plot_ext_source_box(data, client_id, ext_source):
    fig_ext_source_box = px.box(data, x=ext_source, title=f'Horizontal Box Plot of {ext_source}')
    client_ext_source = data[data['SK_ID_CURR'] == client_id][ext_source].values
    if client_ext_source.size > 0:
        fig_ext_source_box.add_shape(
            type='line',
            x0=client_ext_source[0], x1=client_ext_source[0],
            y0=0, y1=1,
            line=dict(color='Red', width=3),
            xref='x', yref='paper'
        )
    st.plotly_chart(fig_ext_source_box)


# Function to filter data based on selected options
def filter_data(data, client_id, filter_option, data_age_category, education_columns):
    filtered_data = data.copy()
    if filter_option.startswith('All'):
        return filtered_data
    elif filter_option.startswith('Age Category'):
        age_category_option = data_age_category[data_age_category['SK_ID_CURR'] == client_id]['AGE_CATEGORY'].values
        if age_category_option.size > 0:
            filtered_data = data_age_category[data_age_category['AGE_CATEGORY'] == age_category_option[0]]
    elif filter_option.startswith('Gender'):
        client_gender = data[data['SK_ID_CURR'] == client_id]['CODE_GENDER'].values
        if client_gender.size > 0:
            filtered_data = data[data['CODE_GENDER'] == client_gender[0]]
    elif filter_option.startswith('Education Type'):
        selected_education = None
        for col in education_columns:
            if data.loc[data['SK_ID_CURR'] == client_id, col].values[0] == 1:
                selected_education = col
                break
        if selected_education:
            filtered_data = data[data[selected_education] == 1]
    elif filter_option.startswith('Car Owner'):
        client_car_ownership = data[data['SK_ID_CURR'] == client_id]['FLAG_OWN_CAR'].values
        if client_car_ownership.size > 0:
            filtered_data = data[data['FLAG_OWN_CAR'] == client_car_ownership[0]]
    return filtered_data


# Main App
st.title("Prediction Credit Default")

# Calculate age and create age categories without modifying the original data
data_age_category = data.copy()
data_age_category['AGE'] = (-data['DAYS_BIRTH'] / 365).astype(int)
data_age_category['AGE_CATEGORY'] = pd.cut(data_age_category['AGE'], bins=[0, 25, 35, 45, 55, float('inf')],
                                           labels=['<25', '25-35', '35-45', '45-55', '55+'])

# Sidebar for client ID selection
client_id_list = data['SK_ID_CURR'].unique()
client_id = st.sidebar.selectbox("Select Client ID", options=client_id_list)

# Convert client_id to int for JSON serialization
client_id = int(client_id)

# Fetch client data
client_data = data[data['SK_ID_CURR'] == client_id]

# Display and adjust AMT_GOODS_PRICE
st.subheader("Goods Price Adjustment")
current_goods_price = data[data['SK_ID_CURR'] == client_id]['AMT_GOODS_PRICE'].values[0]
st.write(f"Current Goods Price: ${current_goods_price}")
new_goods_price = st.slider("Adjust Goods Price:", min_value=45000, max_value=3000000, value=int(current_goods_price))

# Display and adjust AMT_ANNUITY
st.subheader("Annuity Adjustment")
current_annuity = data[data['SK_ID_CURR'] == client_id]['AMT_ANNUITY'].values[0]
st.write(f"Current Annuity: ${current_annuity}")
new_annuity = st.slider("Adjust Annuity:", min_value=1000, max_value=1000000, value=int(current_annuity))

# Recalculate prediction with updated goods price and annuity
prediction_data = fetch_prediction(client_id, new_goods_price, new_annuity)
print(f"API response for client {client_id}: {prediction_data}")

# Display updated prediction
st.subheader("Updated Prediction")
if prediction_data:
    # Visualize credit score
    st.subheader("Credit Score")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_data['probability'],
        title={'text': "Credit Score"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 0.4], 'color': "lightgreen"},
                   {'range': [0.4, 0.5], 'color': "yellow"},
                   {'range': [0.5, 1], 'color': "red"}],
               'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.45}}))
    st.plotly_chart(fig)

    # Display probability message
    display_probability_message(prediction_data['probability'])

    # Create a SHAP plot
    fig, ax = plt.subplots()
    shap.plots.bar(shap_globalvalues, max_display=10, show=False, ax=ax)
    st.pyplot(fig)

    plot_local_waterfall_by_id(client_id, feature_explainer)

# Determine the client's age category, gender, and education type
age_category_option = data_age_category[data_age_category['SK_ID_CURR'] == client_id]['AGE_CATEGORY'].values
client_gender = data[data['SK_ID_CURR'] == client_id]['CODE_GENDER'].values
education_columns = [col for col in data.columns if col.startswith('NAME_EDUCATION_TYPE_')]
selected_education = None
for col in education_columns:
    if data.loc[data['SK_ID_CURR'] == client_id, col].values[0] == 1:
        selected_education = col.replace('NAME_EDUCATION_TYPE_', '').replace('_', ' ')
        break

# Convert gender to 'F' or 'M'
client_gender_display = 'F' if client_gender[0] == 1 else 'M'

# Update filter options with client-specific details
client_car_ownership = data[data['SK_ID_CURR'] == client_id]['FLAG_OWN_CAR'].values
car_ownership_display = 'Yes' if client_car_ownership[0] == 1 else 'No'

filter_options = [
    'All',
    f'Age Category ({age_category_option[0]})' if age_category_option.size > 0 else 'Age Category',
    f'Gender ({client_gender_display})' if client_gender.size > 0 else 'Gender',
    f'Education Type ({selected_education})' if selected_education else 'Education Type',
    f'Car Owner ({car_ownership_display})'
]

# Select box to filter data by age category, gender, or education type
filter_option = st.selectbox("Filter data by:", options=filter_options)

# Apply the filter to data
filtered_data = filter_data(data, client_id, filter_option, data_age_category, education_columns)

# Select box for income distribution graphs
income_distribution_options = ['AMT_INCOME_TOTAL', 'INCOME_PER_PERSON']
income_graph_option = st.selectbox("Select income distribution graph to display:", options=income_distribution_options)

# Display the selected income distribution graph with client marker
plot_income_distribution_with_marker(filtered_data, client_id, income_graph_option)

# Select box for box plots
boxplot_options = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
boxplot_graph_option = st.selectbox("Select box plot to display:", options=boxplot_options)

# Display the selected box plot
if boxplot_graph_option:
    plot_ext_source_box(filtered_data, client_id, boxplot_graph_option)
