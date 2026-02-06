import joblib
import pandas as pd

MODEL_DIR = "models"


def load_artifacts():
    """Load trained model and preprocessing artifacts.

    Returns:
        tuple: (model, scaler, features, le_sex, le_embarked)
    """
    model = joblib.load(f"{MODEL_DIR}/titanic_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    features = joblib.load(f"{MODEL_DIR}/features.pkl")
    le_sex = joblib.load(f"{MODEL_DIR}/le_sex.pkl")
    le_embarked = joblib.load(f"{MODEL_DIR}/le_embarked.pkl")
    return model, scaler, features, le_sex, le_embarked


def predict(pclass, sex, age, sibsp, parch, fare, embarked):
    """Predict Titanic survival probability for a passenger.

    Args:
        pclass: Passenger class (1, 2, or 3)
        sex: Gender ("male" or "female")
        age: Passenger age
        sibsp: Number of siblings/spouses aboard
        parch: Number of parents/children aboard
        fare: Ticket fare
        embarked: Port of embarkation ("C", "Q", or "S")

    Returns:
        dict: {"survival": int, "probability": float}
    """
    model, scaler, features, le_sex, le_embarked = load_artifacts()

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size

    age_bin = 0
    if age > 12:
        age_bin = 1
    if age > 18:
        age_bin = 2
    if age > 35:
        age_bin = 3
    if age > 60:
        age_bin = 4

    sample = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Sex_encoded": [le_sex.transform([sex])[0]],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked_encoded": [le_embarked.transform([embarked])[0]],
            "FamilySize": [family_size],
            "IsAlone": [is_alone],
            "FarePerPerson": [fare_per_person],
            "AgeBin": [age_bin],
        }
    )

    sample_scaled = scaler.transform(sample)
    survival = int(model.predict(sample_scaled)[0])
    probability = float(model.predict_proba(sample_scaled)[0][1])

    return {"survival": survival, "probability": probability}


if __name__ == "__main__":
    result = predict(pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.25, embarked="S")
    print(f"Result: {result}")
