import joblib
import os
import pandas as pd
import numpy as np

try:
    from modelscope import Model, PipelineMixin
except ImportError:
    PipelineMixin = object


class Lifeboat(PipelineMixin):
    MODEL_DIR = os.path.dirname(__file__)

    def __init__(self, model_dir=None, **kwargs):
        if model_dir is not None:
            self.MODEL_DIR = model_dir
        self.model = None
        self.scaler = None
        self.features = None
        self.le_sex = None
        self.le_embarked = None
        super().__init__(**kwargs)

    def load(self):
        self.model = joblib.load(f"{self.MODEL_DIR}/titanic_model.pkl")
        self.scaler = joblib.load(f"{self.MODEL_DIR}/scaler.pkl")
        self.features = joblib.load(f"{self.MODEL_DIR}/features.pkl")
        self.le_sex = joblib.load(f"{self.MODEL_DIR}/le_sex.pkl")
        self.le_embarked = joblib.load(f"{self.MODEL_DIR}/le_embarked.pkl")
        return self

    def preprocess(self, input):
        pclass = input.get("pclass")
        sex = input.get("sex")
        age = input.get("age")
        sibsp = input.get("sibsp")
        parch = input.get("parch")
        fare = input.get("fare")
        embarked = input.get("embarked")

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
                "Sex_encoded": [self.le_sex.transform([sex])[0]],
                "Age": [age],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare],
                "Embarked_encoded": [self.le_embarked.transform([embarked])[0]],
                "FamilySize": [family_size],
                "IsAlone": [is_alone],
                "FarePerPerson": [fare_per_person],
                "AgeBin": [age_bin],
            }
        )
        return sample

    def forward(self, sample):
        return self.model.predict_proba(sample)

    def postprocess(self, output):
        probability = float(output[0][1])
        survival = int(probability > 0.5)
        return {"survival": survival, "probability": probability}

    def predict(self, **kwargs):
        if self.model is None:
            self.load()
        sample = self.preprocess(kwargs)
        sample_scaled = self.scaler.transform(sample)
        output = self.forward(sample_scaled)
        return self.postprocess(output)

    def __call__(self, **kwargs):
        return self.predict(**kwargs)
