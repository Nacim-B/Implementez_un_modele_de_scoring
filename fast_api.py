import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# load test data and model
dataset_path = "./data_test_for_dashboard.csv"
model_path = "./old_client_model.pkl"
model_new_client_path = "./new_client_model.pkl"

data = pd.read_csv(dataset_path)

data_new_client = data[['AMT_GOODS_PRICE', 'INCOME_PER_PERSON', 'AMT_ANNUITY', 'DAYS_BIRTH']]

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(model_new_client_path, 'rb') as f:
    model_new_client = pickle.load(f)


class PredictionRequest(BaseModel):
    id_client: int


class PredictionNewClientRequest(BaseModel):
    loan_request_amount: int
    annual_salary: int
    annual_annuity: int
    age: int


@app.get("/")
def read_root():
    return {"message": "API is working"}


@app.post("/predict/")
def predict(request: PredictionRequest):
    # verify if client exists
    if request.id_client not in data["SK_ID_CURR"].values:
        raise HTTPException(status_code=404, detail="Client ID non trouvé")

    # retrieve client data
    client_data = data[data["SK_ID_CURR"] == request.id_client].drop(columns=["SK_ID_CURR"])

    # prediction
    prediction = model.predict(client_data)[0]
    probability_default = model.predict_proba(client_data)[0][1]

    return {"id_client": request.id_client, "prediction": prediction, "probability": probability_default}


@app.post("/predict_new_client/")
def predict_new_client(request: PredictionNewClientRequest):
    input_data = np.array([
        request.loan_request_amount,
        request.annual_salary,
        request.annual_annuity,
        request.age
    ]).reshape(1, -1)

    # Perform prediction
    prediction = model_new_client.predict(input_data)[0]
    probability_default = model_new_client.predict_proba(input_data)[0][1]  # Probability of Default

    return {"prediction": prediction, "probability": probability_default}
