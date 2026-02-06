import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys


MODEL_DIR = "models"


def are_model_files_available():
    required_files = [
        "titanic_model.pkl",
        "scaler.pkl",
        "features.pkl",
        "le_sex.pkl",
        "le_embarked.pkl",
    ]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files)


@pytest.mark.skipif(
    not are_model_files_available(),
    reason="Model files not available for e2e testing",
)
class TestEndToEnd:
    def test_load_artifacts_success(self):
        from inference import load_artifacts

        model, scaler, features, le_sex, le_embarked = load_artifacts()

        assert model is not None
        assert scaler is not None
        assert features is not None
        assert le_sex is not None
        assert le_embarked is not None

    def test_full_prediction_flow_male_single(self):
        from inference import predict

        result = predict(
            pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.25, embarked="S"
        )

        assert "survival" in result
        assert "probability" in result
        assert result["survival"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_full_prediction_flow_female_first_class(self):
        from inference import predict

        result = predict(
            pclass=1, sex="female", age=30, sibsp=1, parch=1, fare=100, embarked="C"
        )

        assert "survival" in result
        assert "probability" in result
        assert result["survival"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_full_prediction_flow_family_travel(self):
        from inference import predict

        result = predict(
            pclass=2, sex="male", age=45, sibsp=2, parch=2, fare=50, embarked="Q"
        )

        assert "survival" in result
        assert "probability" in result
        assert result["survival"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_full_prediction_flow_child_traveling(self):
        from inference import predict

        result = predict(
            pclass=3, sex="female", age=10, sibsp=3, parch=1, fare=15, embarked="S"
        )

        assert "survival" in result
        assert "probability" in result
        assert result["survival"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_full_prediction_flow_elderly_passenger(self):
        from inference import predict

        result = predict(
            pclass=1, sex="male", age=70, sibsp=0, parch=0, fare=200, embarked="C"
        )

        assert "survival" in result
        assert "probability" in result
        assert result["survival"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_full_prediction_batch_consistency(self):
        from inference import predict

        test_params = {
            "pclass": 1,
            "sex": "female",
            "age": 25,
            "sibsp": 0,
            "parch": 0,
            "fare": 50,
            "embarked": "C",
        }

        results = [predict(**test_params) for _ in range(5)]

        for result in results:
            assert result["survival"] == results[0]["survival"]
            assert result["probability"] == results[0]["probability"]

    def test_scaler_input_dimensions(self):
        from inference import load_artifacts
        import numpy as np

        model, scaler, features, le_sex, le_embarked = load_artifacts()

        sample = pd.DataFrame(
            {
                "Pclass": [3],
                "Sex_encoded": [0],
                "Age": [25],
                "SibSp": [0],
                "Parch": [0],
                "Fare": [7.25],
                "Embarked_encoded": [0],
                "FamilySize": [1],
                "IsAlone": [1],
                "FarePerPerson": [7.25],
                "AgeBin": [2],
            }
        )

        sample_scaled = scaler.transform(sample)

        assert sample_scaled.shape[1] == len(features)

    def test_model_prediction_returns_valid_output(self):
        from inference import load_artifacts, predict
        import numpy as np

        model, scaler, features, le_sex, le_embarked = load_artifacts()

        sample = pd.DataFrame(
            {
                "Pclass": [1],
                "Sex_encoded": [1],
                "Age": [30],
                "SibSp": [1],
                "Parch": [1],
                "Fare": [100],
                "Embarked_encoded": [0],
                "FamilySize": [3],
                "IsAlone": [0],
                "FarePerPerson": [33.33],
                "AgeBin": [2],
            }
        )

        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        probabilities = model.predict_proba(sample_scaled)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
        assert len(probabilities[0]) == 2
        assert probabilities[0].sum() == pytest.approx(1.0)


@pytest.mark.skipif(
    not are_model_files_available(),
    reason="Model files not available for e2e testing",
)
class TestCLIInterface:
    def test_main_module_execution(self, capsys):
        import subprocess

        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
        )

        assert "Result:" in result.stdout or result.stdout != ""


class TestFeatureEngineering:
    def test_family_size_calculation(self):
        from inference import predict
        from unittest.mock import patch, MagicMock

        model = MagicMock()
        model.predict.return_value = [0]
        model.predict_proba.return_value = [[0.5, 0.5]]

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

        with patch("inference.load_artifacts") as mock_load:
            mock_load.return_value = (model, scaler, features, le_sex, le_embarked)

            result_solo = predict(
                pclass=3, sex="male", age=30, sibsp=0, parch=0, fare=30, embarked="S"
            )
            result_family = predict(
                pclass=3, sex="male", age=30, sibsp=2, parch=2, fare=120, embarked="S"
            )

            assert result_solo is not None
            assert result_family is not None
