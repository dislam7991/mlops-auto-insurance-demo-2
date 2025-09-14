from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

import autoinsurance

app = FastAPI(title="Auto Insurance Claim Prediction API")

class Claim(BaeModel):
    features: Dict[str, Any]

# add routes
@app.get("/health")
def health_health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Claim):
    model = joblib.load("models/model.pkl")
    X = pd.DataFrame([payload.features])
    yhat = model.predict(X)[0]
    proba = getattr(model, "predict_proba", lambda x: [[None, None]])(X)[0]
    return {"prediction": int(yhat), "proba": proba[1] if proba and len(proba) > 1 else None}

# To run this app, use the command:
# uvicorn autoinsurance.serving.app:app --host 0.0.0.0 --port 8000