from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/churn_model.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine Learning API for predicting customer churn",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    data = pd.DataFrame([{
        "gender": customer.gender,
        "SeniorCitizen": customer.SeniorCitizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "tenure": customer.tenure,
        "PhoneService": customer.PhoneService,
        "MultipleLines": customer.MultipleLines,
        "InternetService": customer.InternetService,
        "OnlineSecurity": customer.OnlineSecurity,
        "OnlineBackup": customer.OnlineBackup,
        "DeviceProtection": customer.DeviceProtection,
        "TechSupport": customer.TechSupport,
        "StreamingTV": customer.StreamingTV,
        "StreamingMovies": customer.StreamingMovies,
        "Contract": customer.Contract,
        "PaperlessBilling": customer.PaperlessBilling,
        "PaymentMethod": customer.PaymentMethod,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges
    }])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Yes"
    else:
        result = "No"

    return {
        "churn_prediction": result
    }