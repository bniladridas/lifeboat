---
title: Lifeboat
subTitle: Titanic Survival Prediction
author: bniladridas
tags: tabular-classification,scikit-learn,gradient-boosting
license: MIT
---
# Lifeboat

Binary classification model for predicting Titanic passenger survival using Gradient Boosting.

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | GradientBoostingClassifier |
| Task | tabular-classification |
| Framework | scikit-learn |
| Test Accuracy | 81% |
| Test AUC-ROC | 0.81 |

## Training Data

- Dataset: Titanic (Kaggle)
- Samples: 891
- Train/Test Split: 80/20

## Usage

```python
from inference import predict

result = predict(
    pclass=3,
    sex="male",
    age=25,
    sibsp=0,
    parch=0,
    fare=7.25,
    embarked="S"
)
# Output: {"survival": 0, "probability": 0.15}
```
