from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.model_loader import load_model
import numpy as np
app = FastAPI(title="ML Lifecycle API", description="Scikit-learn model served via FastAPI", version="1.0.0")
model = None
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = np.array(request.features).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction])
    return PredictResponse(prediction=prediction, confidence=round(confidence, 4), model_version="ChurnClassifier/latest")
