import os
import joblib
import pandas as pd

MODEL_PATH = os.getenv("MODEL_LOCAL_PATH", "models/titanic_model.joblib")

model = joblib.load(MODEL_PATH)


def predict_survival(data: dict) -> int:
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return int(prediction)