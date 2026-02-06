import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import joblib


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


@patch("inference.load_artifacts")
def test_predict_returns_dict(mock_load, mock_artifacts):
    mock_load.return_value = mock_artifacts

    from inference import predict

    result = predict(
        pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.25, embarked="S"
    )

    assert isinstance(result, dict)
    assert "survival" in result
    assert "probability" in result
    assert result["survival"] == 0
    assert 0 <= result["probability"] <= 1


@patch("inference.load_artifacts")
def test_predict_survival_prediction(mock_load, mock_artifacts):
    model, scaler, features, le_sex, le_embarked = mock_artifacts
    model.predict.return_value = [1]
    model.predict_proba.return_value = [[0.3, 0.7]]
    mock_load.return_value = (model, scaler, features, le_sex, le_embarked)

    from inference import predict

    result = predict(
        pclass=1, sex="female", age=30, sibsp=1, parch=1, fare=100, embarked="C"
    )

    assert result["survival"] == 1
    assert result["probability"] == 0.7


@patch("inference.load_artifacts")
def test_feature_engineering_values(mock_load, mock_artifacts):
    model, scaler, features, le_sex, le_embarked = mock_artifacts
    model.predict.return_value = [0]
    model.predict_proba.return_value = [[0.95, 0.05]]
    mock_load.return_value = (model, scaler, features, le_sex, le_embarked)

    from inference import predict

    result = predict(
        pclass=3, sex="male", age=7, sibsp=3, parch=2, fare=20, embarked="Q"
    )

    assert result["survival"] == 0
    assert result["probability"] == 0.05
