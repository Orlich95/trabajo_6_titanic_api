from pydantic import BaseModel

class TitanicInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

class PredictionResponse(BaseModel):
    prediction: int