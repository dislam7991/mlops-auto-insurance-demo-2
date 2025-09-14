from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os

app = FastAPI(title="Auto Insurance Claim Prediction API")

class Claim(BaseModel):
    features: Dict[str, Any]


@app.on_event("startup")
def _load_model():
    """Load model once on startup and store in app.state."""
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    try:
        app.state.model = joblib.load(model_path)
        app.state.model_path = model_path
    except Exception as e:
        # Defer raising until first request so app can still start for /health
        app.state.model = None
        app.state.model_load_error = str(e)

# add routes
@app.get("/")
def root():
    return {"name": app.title, "version": "1.0.0"}

@app.get("/health")
def health():
    status = "ok" if getattr(app.state, "model", None) is not None else "degraded"
    return {"status": status, "model_path": getattr(app.state, "model_path", None)}

@app.post("/predict")
def predict(payload: Claim):
    if getattr(app.state, "model", None) is None:
        # Try (one-time) lazy load if startup failed
        try:
            app.state.model = joblib.load(getattr(app.state, "model_path", "models/model.pkl"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {getattr(app.state, 'model_load_error', str(e))}")

    model = app.state.model
    X = pd.DataFrame([payload.features])
    yhat = model.predict(X)[0]
    proba = getattr(model, "predict_proba", lambda x: [[None, None]])(X)[0]
    p1 = None
    if proba is not None:
        try:
            p1 = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception:
            p1 = None
    return {"prediction": int(yhat), "proba": p1}

if __name__ == "__main__":
    # Run with: python -m autoinsurance.serving.app
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("autoinsurance.serving.app:app", host=host, port=port, reload=reload)