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
    # Retrieve client data using id_client
    client_data = data[data['SK_ID_CURR'] == request.id_client]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client ID not found")

    # Drop the id_client column for prediction
    features = client_data.drop(columns=['SK_ID_CURR'])

    # Make prediction
    prediction = pipeline.predict(features)
    probability = pipeline.predict_proba(features)[:, 1]

    # Compute SHAP values
    explainer = shap.Explainer(model, features)
    shap_values = explainer(features)

    # Prepare feature importances
    feature_importances = shap_values.values.mean(axis=0)
    feature_importance_list = list(zip(features.columns, feature_importances))
    feature_importance_list.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "id_client": request.id_client,
        "classification": int(prediction[0]),
        "probability": float(probability[0]),
        "feature_importances": feature_importance_list
    }