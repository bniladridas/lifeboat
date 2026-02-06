# API Documentation

## Inference Module

### `predict(pclass, sex, age, sibsp, parch, fare, embarked)`

Predict Titanic survival probability for a passenger.

**Parameters:**
- `pclass` (int): Passenger class (1, 2, or 3)
- `sex` (str): Gender ("male" or "female")
- `age` (int): Passenger age
- `sibsp` (int): Number of siblings/spouses aboard
- `parch` (int): Number of parents/children aboard
- `fare` (float): Ticket fare
- `embarked` (str): Port of embarkation ("C", "Q", or "S")

**Returns:**
- `dict`:
  - `survival` (int): 0 = not survived, 1 = survived
  - `probability` (float): Probability of survival (0-1)

**Example:**
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
# Returns: {"survival": 0, "probability": 0.1}
```

---

### `load_artifacts()`

Load model artifacts from the `models/` directory.

**Returns:**
- `tuple`: (model, scaler, features, le_sex, le_embarked)

**Raises:**
- `FileNotFoundError`: If model files are missing

---

## Training Module

### `train()`

Train Titanic survival models using multiple algorithms.

**Returns:**
- `dict`: Training results with model names and CV accuracy scores

**Models trained:**
- XGBoost
- LightGBM
- Gradient Boosting
- Random Forest
- Logistic Regression
- Decision Tree
- Dummy Classifier

---

## Upload Module

### `upload()`

Upload trained model to ModelScope.

**Requirements:**
- Model files must exist in `models/`
- ModelScope credentials configured

---

## Feature Engineering

### Engineered Features

| Feature | Description |
|---------|-------------|
| `FamilySize` | sibsp + parch + 1 |
| `IsAlone` | 1 if FamilySize == 1, else 0 |
| `FarePerPerson` | fare / FamilySize |
| `AgeBin` | Age category (0-4) |

### Age Bin Categories

| Bin | Age Range |
|-----|-----------|
| 0 | 0-12 (child) |
| 1 | 13-18 (teen) |
| 2 | 19-35 (young adult) |
| 3 | 36-60 (adult) |
| 4 | 60+ (senior) |
