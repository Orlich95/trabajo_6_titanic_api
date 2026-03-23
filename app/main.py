from fastapi import FastAPI, HTTPException
from app.schemas import TitanicInput, PredictionResponse
from app.model import predict_survival

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API Titanic funcionando"}

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: TitanicInput):
    try:
        prediction = predict_survival(data.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))