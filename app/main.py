from fastapi import FastAPI, HTTPException
from app.schemas import TitanicInput, PredictionResponse
from app.model import predict_survival

app = FastAPI(title="Titanic Survival API", version="2.0")


@app.get("/")
def home():
    return {
        "message": "API Titanic funcionando",
        "project": "Trabajo Práctico 7 - MLOps",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "v2.0",
        "service": "titanic-api"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TitanicInput):
    try:
        prediction = predict_survival(data.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))