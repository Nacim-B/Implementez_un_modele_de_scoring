# Import necessary libraries
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import pickle
import numpy as np

# Constants
API_URL = "http://127.0.0.1:8080"
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

# Sidebar for client ID selection
client_id_list = data['SK_ID_CURR'].unique()
client_id = st.sidebar.selectbox("Select Client ID", options=client_id_list)

# Convert client_id to int for JSON serialization
client_id = int(client_id)

# Fetch client data
client_data = data[data['SK_ID_CURR'] == client_id]

# Create a container for client information at the top
client_info = st.container()
with client_info:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Client ID")
        st.markdown(f"<h2 style='text-align: center;'>{client_id}</h2>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("👤 Client Profile")
        gender = "Female" if client_data['CODE_GENDER'].values[0] == 1 else "Male"
        age = int(-client_data['DAYS_BIRTH'].values[0] / 365)
        st.write(f"Gender: {gender}")
        st.write(f"Age: {age} years")
    
    with col3:
        st.subheader("💰 Financial Info")
        income = client_data['AMT_INCOME_TOTAL'].values[0]
        st.write(f"Income Total: ${income:,.2f}")
        st.write(f"Income per Person in Family: ${client_data['INCOME_PER_PERSON'].values[0]:,.2f}")

st.markdown("---")  # Add a separator line

# Calculate age and create age categories without modifying the original data
data_age_category = data.copy()
data_age_category['AGE'] = (-data['DAYS_BIRTH'] / 365).astype(int)
data_age_category['AGE_CATEGORY'] = pd.cut(data_age_category['AGE'], bins=[0, 25, 35, 45, 55, float('inf')],
                                           labels=['<25', '25-35', '35-45', '45-55', '55+'])

# Get initial prediction with current values
prediction_data = fetch_prediction(client_id, 
                                 client_data['AMT_GOODS_PRICE'].values[0],
                                 client_data['AMT_ANNUITY'].values[0])

# Display prediction section
prediction_container = st.container()
with prediction_container:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualize credit score
        if prediction_data:
            st.subheader("🎯 Credit Score Prediction")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_data['probability'],
                title={'text': "Default Risk Score"},
                number={'suffix': "%", 'valueformat': ".2f", 'font': {'size': 24}},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.4], 'color': "lightgreen"},
                           {'range': [0.4, 0.5], 'color': "yellow"},
                           {'range': [0.5, 1], 'color': "red"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.45}}))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Decision")
        if prediction_data:
            probability = prediction_data['probability']
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if probability < 0.4:
                st.success("ACCEPT THE LOAN", icon="✅")
            elif 0.4 <= probability < 0.5:
                st.warning("NEEDS REVIEW", icon="⚠️")
            else:
                st.error("DECLINE THE LOAN", icon="❌")

st.markdown("---")  # Add a separator line

# Adjustable parameters section
st.subheader("📝 Adjustable Parameters")
col1, col2 = st.columns(2)

with col1:
    # Display and adjust AMT_GOODS_PRICE
    current_goods_price = data[data['SK_ID_CURR'] == client_id]['AMT_GOODS_PRICE'].values[0]
    st.write("Current Goods Price:", f"${current_goods_price:,.2f}")
    new_goods_price = st.slider("Adjust Goods Price:", min_value=45000, max_value=3000000, value=int(current_goods_price))

with col2:
    # Display and adjust AMT_ANNUITY
    current_annuity = data[data['SK_ID_CURR'] == client_id]['AMT_ANNUITY'].values[0]
    st.write("Current Annuity:", f"${current_annuity:,.2f}")
    new_annuity = st.slider("Adjust Annuity:", min_value=1000, max_value=1000000, value=int(current_annuity))

# Only show recalculation if values have changed
if new_goods_price != current_goods_price or new_annuity != current_annuity:
    
    # Recalculate prediction with updated goods price and annuity
    prediction_data = fetch_prediction(client_id, new_goods_price, new_annuity)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction_data:
            st.subheader("🎯 Updated Credit Score")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_data['probability'],
                title={'text': "Updated Default Risk Score"},
                number={'suffix': "%", 'valueformat': ".2f", 'font': {'size': 24}},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.4], 'color': "lightgreen"},
                           {'range': [0.4, 0.5], 'color': "yellow"},
                           {'range': [0.5, 1], 'color': "red"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.45}}))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if prediction_data:
            st.subheader("📊 Updated Decision")
            probability = prediction_data['probability']
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if probability < 0.4:
                st.success("ACCEPT THE LOAN", icon="✅")
            elif 0.4 <= probability < 0.5:
                st.warning("NEEDS REVIEW", icon="⚠️")
            else:
                st.error("DECLINE THE LOAN", icon="❌")

st.markdown("---")  # Add a separator line

# Feature importance and analysis section
st.subheader("🔍 Feature Analysis")

# Create tabs for individual and global feature importance
tab1, tab2 = st.tabs(["Individual Feature Importance", "Global Feature Importance"])

with tab1:
    # Get SHAP values for the current client
    client_data_for_shap = data[data['SK_ID_CURR'] == client_id].drop('SK_ID_CURR', axis=1)
    shap_values = feature_explainer(client_data_for_shap)

    # Create a more user-friendly feature importance table
    feature_importance = pd.DataFrame({
        'Feature': client_data_for_shap.columns,
        'Impact': shap_values.values[0],
        'Direction': np.where(shap_values.values[0] > 0, '🔺 Increases Risk', '🔽 Decreases Risk')
    })

    # Sort by absolute impact value and get top 10 features
    feature_importance['AbsImpact'] = abs(feature_importance['Impact'])
    feature_importance = feature_importance.sort_values('AbsImpact', ascending=False).head(10)
    feature_importance = feature_importance.drop('AbsImpact', axis=1)

    # Create two columns for the feature importance display
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a styled dataframe
        styled_df = pd.DataFrame({
            'Feature': feature_importance['Feature'].apply(lambda x: x.replace('_', ' ').title()),
            'Current Value': feature_importance['Impact'],
            'Impact': feature_importance['Direction']
        }).reset_index(drop=True)
        
        # Apply custom styling
        st.dataframe(
            styled_df,
            column_config={
                "Feature": "Feature Name",
                "Current Value": "Importance",
                "Impact": "Risk Impact"
            },
            height=400
        )

    with col2:
        st.subheader("Understanding the Analysis")
        st.write("""
        This table shows the top 10 factors that influence the loan decision.
        """)

with tab2:
    # Calculate global feature importance using mean absolute SHAP values
    global_shap_values = shap_globalvalues.abs.mean(0)
    
    # Create DataFrame for global importance
    global_importance = pd.DataFrame({
        'Feature': client_data_for_shap.columns,
        'Global Impact': global_shap_values.values
    })
    
    # Sort by global impact and get top 10
    global_importance = global_importance.sort_values('Global Impact', ascending=False).head(10)
    
    # Create two columns for the global importance display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a styled dataframe for global importance
        styled_global_df = pd.DataFrame({
            'Feature': global_importance['Feature'].apply(lambda x: x.replace('_', ' ').title()),
            'Global Importance': global_importance['Global Impact']
        }).reset_index(drop=True)
        
        # Apply custom styling
        st.dataframe(
            styled_global_df,
            column_config={
                "Feature": "Feature Name",
                "Global Importance": st.column_config.NumberColumn(
                    "Global Importance",
                    format="%.4f"
                )
            },
            height=400
        )
    
    with col2:
        st.subheader("Understanding Global Impact")
        st.write("""
        This table shows the overall importance of each feature across all clients.
        """)

st.markdown("---")  # Add a separator line

# Comparison section
st.subheader("📊 Comparative Analysis")

# Determine the client's characteristics
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

# Create filter options
filter_options = [
    'All',
    f'Age Category ({age_category_option[0]})' if age_category_option.size > 0 else 'Age Category',
    f'Gender ({client_gender_display})' if client_gender.size > 0 else 'Gender',
    f'Education Type ({selected_education})' if selected_education else 'Education Type',
    f'Car Owner ({car_ownership_display})'
]

# Filter selection
st.write("Compare with clients who match:")
filter_option = st.selectbox("", options=filter_options)

# Apply the filter to data
filtered_data = filter_data(data, client_id, filter_option, data_age_category, education_columns)

# List of features to compare
comparison_features = [
    'AMT_INCOME_TOTAL', 
    'INCOME_PER_PERSON', 
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3'
]

# Get current client's values
client_values = data[data['SK_ID_CURR'] == client_id][comparison_features].iloc[0]

# Create comparison plots
for feature in comparison_features:
    fig = go.Figure()
    
    # Add box plot for filtered clients
    fig.add_trace(go.Box(
        x=filtered_data[feature],
        name='Filtered Clients',
        boxpoints=False,
        line_color='lightgray',
        fillcolor='lightgray',
        showlegend=False
    ))
    
    # Add client marker
    client_value = client_values[feature]
    fig.add_trace(go.Scatter(
        x=[client_value],
        y=['Filtered Clients'],
        mode='markers',
        name='Current Client',
        marker=dict(
            symbol='star',
            size=15,
            color='red',
        ),
        showlegend=False
    ))
    
    # Calculate percentile of client within filtered group
    percentile = (filtered_data[feature] <= client_value).mean() * 100
    
    # Format feature name for display
    display_name = feature
    if feature == 'AMT_INCOME_TOTAL':
        display_name = 'Total Income'
    elif feature == 'INCOME_PER_PERSON':
        display_name = 'Income per Person'
    elif feature.startswith('EXT_SOURCE'):
        display_name = f'External Source {feature[-1]}'
    
    # Add annotations for context
    fig.update_layout(
        title=f"{display_name}<br><sup>Client is at {percentile:.1f}th percentile {f'among {filter_option} clients' if filter_option != 'All' else ''}</sup>",
        height=150,
        margin=dict(l=0, r=0, t=60, b=0),
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(
            zeroline=False,
            showgrid=True,
            showline=True,
            showticklabels=True,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")  # Add a separator line

# Additional Analysis Expander
with st.expander("🔍 More Information - Bivariate Analysis"):
    st.write("This analysis shows the relationship between Income and Credit Amount, colored by default risk:")
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter plot for all clients
    fig.add_trace(go.Scatter(
        x=data['AMT_INCOME_TOTAL'],
        y=data['AMT_CREDIT'],
        mode='markers',
        marker=dict(
            size=8,
            color=data['TARGET'],  # Color by target/probability
            colorscale='RdYlGn_r',  # Red for high risk, green for low risk
            showscale=True,
            colorbar=dict(
                title='Default Risk',
                tickformat='.0%'
            )
        ),
        name='All Clients',
        opacity=0.6
    ))
    
    # Add current client as a star
    client_data = data[data['SK_ID_CURR'] == client_id]
    fig.add_trace(go.Scatter(
        x=client_data['AMT_INCOME_TOTAL'],
        y=client_data['AMT_CREDIT'],
        mode='markers',
        marker=dict(
            symbol='star',
            size=20,
            color='red',
            line=dict(
                color='black',
                width=2
            )
        ),
        name='Current Client',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title='Income vs Credit Amount Analysis',
        xaxis_title='Total Income',
        yaxis_title='Credit Amount',
        height=500,
        xaxis=dict(
            tickformat='$,.0f',
            showgrid=True
        ),
        yaxis=dict(
            tickformat='$,.0f',
            showgrid=True
        ),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **Understanding this plot:**
    - Each point represents a client
    - X-axis shows total income
    - Y-axis shows credit amount requested
    - Color indicates default risk (red = higher risk, green = lower risk)
    - Your position is marked with a red star ⭐
    
    This visualization helps understand how income and credit amount relate to default risk.
    """)
