import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_DIR = "models"


def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/titanic_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    le_sex = joblib.load(f"{MODEL_DIR}/le_sex.pkl")
    le_embarked = joblib.load(f"{MODEL_DIR}/le_embarked.pkl")
    return model, scaler, le_sex, le_embarked


model, scaler, le_sex, le_embarked = load_artifacts()


class PredictRequest(BaseModel):
    pclass: int
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0)
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    fare: float = Field(..., ge=0)
    embarked: Literal["C", "Q", "S"]


@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictRequest):
    family_size = request.sibsp + request.parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = request.fare / family_size

    age_bin = 0
    if request.age > 12:
        age_bin = 1
    if request.age > 18:
        age_bin = 2
    if request.age > 35:
        age_bin = 3
    if request.age > 60:
        age_bin = 4

    sample = pd.DataFrame(
        {
            "Pclass": [request.pclass],
            "Sex_encoded": [le_sex.transform([request.sex])[0]],
            "Age": [request.age],
            "SibSp": [request.sibsp],
            "Parch": [request.parch],
            "Fare": [request.fare],
            "Embarked_encoded": [le_embarked.transform([request.embarked])[0]],
            "FamilySize": [family_size],
            "IsAlone": [is_alone],
            "FarePerPerson": [fare_per_person],
            "AgeBin": [age_bin],
        }
    )

    sample_scaled = scaler.transform(sample)
    survival = int(model.predict(sample_scaled)[0])
    probability = float(model.predict_proba(sample_scaled)[0][1])

    return {"survival": survival, "probability": round(probability, 4)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
