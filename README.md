---
title: Lifeboat
subTitle: Titanic Survival Prediction
author: bniladridas
tags: tabular-classification,scikit-learn,gradient-boosting,binary-classification
license: MIT
task: tabular-classification
language: en
libraries: scikit-learn,xgboost,lightgbm
associated_dataset: Titanic (Kaggle)
foundation_model: null
description: A gradient boosting model trained on the Titanic dataset to predict passenger survival based on demographic and ticket features.
---

# Lifeboat

Binary classification model for Titanic passenger survival prediction.

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 81.01% |
| AUC-ROC | 0.8088 |
| CV Mean | 80.21% |

## Features

- Pclass, Sex, Age, SibSp, Parch, Fare
- Embarked, FamilySize, IsAlone, FarePerPerson, AgeBin

## Test

```python
from inference import predict

result = predict(
    pclass=3, sex="male", age=25,
    sibsp=0, parch=0, fare=7.25, embarked="S"
)
print(result)  # {"survival": 0, "probability": 0.01}
```

## Install

```bash
pip install -r requirements.txt
```

## Use as SDK

```bash
pip install lifeboat-sdk
```

```python
from lifeboat_sdk import Lifeboat

model = Lifeboat()
model.load()
model.predict(pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.25, embarked="S")
```

## Use with ModelScope

```python
from modelscope import pipeline

p = pipeline("tabular-classification", "bniladridas/lifeboat")
p(pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.25, embarked="S")
```
# test
# test update Sun Feb  8 03:34:57 IST 2026
