import pickle
import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load test data and model
dataset_path = "./data_test_for_dashboard.csv"
model_path = "./old_client_model.pkl"

data = pd.read_csv(dataset_path)

with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)

# Access the model (LightGBM) from the pipeline
model = pipeline.named_steps['classifier']


class PredictionRequest(BaseModel):
    id_client: int


@app.get("/")
def read_root():
    return {"message": "API is working"}


@app.post("/predict/")
def predict(request: PredictionRequest):
    # Verify if client exists
    if request.id_client not in data["SK_ID_CURR"].values:
        raise HTTPException(status_code=404, detail="Client ID not found")

    # Retrieve client data
    client_data = data[data["SK_ID_CURR"] == request.id_client].drop(columns=["SK_ID_CURR"])

    # Prediction
    prediction = pipeline.predict(client_data)[0]
    probability_default = pipeline.predict_proba(client_data)[0][1]

    # Create SHAP explainer
    explainer = shap.LinearExplainer(model.named_steps['classifier'], shap.maskers.Independent(X1_train))
    # Compute SHAP values for the entire dataset
    shap_values = explainer(X1_train)

    # Convert SHAP values to dictionary for the specific client
    client_shap_values = dict(zip(client_data.columns, shap_values[0].tolist()))

    return {
        "id_client": request.id_client,
        "prediction": prediction,
        "probability": probability_default,
        "global_feature_importance": feature_importance,
        "client_feature_importance": client_shap_values
    }
