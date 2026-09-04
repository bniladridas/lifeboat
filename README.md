<p align="center">
  <img src="https://raw.githubusercontent.com/basebin/lifeboat/main/.github/assets/thumbnail.png" alt="lifeboat" width="100%">
</p>

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

## Foundation Model

This model can be used as a foundation model for fine-tuning on other survival prediction tasks or similar tabular classification problems.

```yaml
# If fine-tuning this model, set in your model card:
foundation_model: bniladridas/lifeboat
```
