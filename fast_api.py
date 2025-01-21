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
    amt_goods_price: float = None
    amt_annuity: float = None


@app.get("/")
def read_root():
    return {"message": "API is working"}


@app.post("/predict/")
def predict(request: PredictionRequest):
    # Retrieve client data using id_client
    client_data = data[data['SK_ID_CURR'] == request.id_client]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client ID not found")

    # Update AMT_GOODS_PRICE if provided
    if request.amt_goods_price is not None:
        client_data['AMT_GOODS_PRICE'] = request.amt_goods_price

    # Update AMT_ANNUITY if provided
    if request.amt_annuity is not None:
        client_data['AMT_ANNUITY'] = request.amt_annuity

    # Drop the id_client column for prediction
    features = client_data.drop(columns=['SK_ID_CURR'])

    # Make prediction
    prediction = pipeline.predict(features)
    probability = pipeline.predict_proba(features)[:, 1]

    return {
        "id_client": request.id_client,
        "classification": int(prediction[0]),
        "probability": float(probability[0])
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
