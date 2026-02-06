# Lifeboat - Titanic Survival Prediction

A machine learning model that predicts Titanic passenger survival probability.

## Quick Start

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
print(result)
# {"survival": 0, "probability": 0.1}
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Training

```bash
python train.py
```

## Project Structure

```
lifeboat/
├── inference.py      # Prediction module
├── train.py          # Training script
├── upload.py         # Model upload
├── models/           # Saved model files
├── tests/            # Test suite
│   ├── test_inference.py
│   └── test_e2e.py
├── docs/             # Documentation
│   └── api.md
└── requirements.txt
```

## Models

The model uses ensemble of:
- XGBoost
- LightGBM
- Gradient Boosting

## License

MIT
