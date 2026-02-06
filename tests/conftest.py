import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import joblib
import os

MODEL_DIR = "models"


@pytest.fixture
def sample_passenger_data():
    return {
        "pclass": 3,
        "sex": "male",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    }


@pytest.fixture
def mock_artifacts():
    model = MagicMock()
    model.predict.return_value = [0]
    model.predict_proba.return_value = [[0.9, 0.1]]

    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 11))

    features = [
        "Pclass",
        "Sex_encoded",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_encoded",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "AgeBin",
    ]

    le_sex = MagicMock()
    le_sex.transform.return_value = [0]

    le_embarked = MagicMock()
    le_embarked.transform.return_value = [0]

    return model, scaler, features, le_sex, le_embarked


@pytest.fixture
def mock_model_files(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model = MagicMock()
    model.predict.return_value = [1]
    model.predict_proba.return_value = [[0.2, 0.8]]

    scaler = MagicMock()
    scaler.transform.return_value = np.ones((1, 11))

    features = [
        "Pclass",
        "Sex_encoded",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_encoded",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "AgeBin",
    ]

    le_sex = MagicMock()
    le_sex.transform.return_value = [0]

    le_embarked = MagicMock()
    le_embarked.transform.return_value = [0]

    joblib.dump(model, model_dir / "titanic_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(features, model_dir / "features.pkl")
    joblib.dump(le_sex, model_dir / "le_sex.pkl")
    joblib.dump(le_embarked, model_dir / "le_embarked.pkl")

    monkeypatch.chdir(tmp_path)

    return model_dir
