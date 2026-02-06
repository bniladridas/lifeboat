# Lifeboat Project Context

This is a Titanic survival prediction ML project with the following structure:

## Core Files
- **inference.py** - Main prediction module with `predict()` function and `load_artifacts()`
- **train.py** - Model training script with XGBoost, LightGBM, and sklearn models
- **upload.py** - Model upload functionality

## Models
- Saved in `models/` directory: `titanic_model.pkl`, `scaler.pkl`, `features.pkl`, `le_sex.pkl`, `le_embarked.pkl`

## Testing
- **tests/test_inference.py** - Unit tests with mocked artifacts
- **tests/test_e2e.py** - End-to-end tests with real model files
- **test_inference.py** - Original test file (legacy)

## Key Features
- Feature engineering: FamilySize, IsAlone, FarePerPerson, AgeBin
- Supports male/female, embarked (S/C/Q)
- Returns survival prediction and probability

## Commands
- `pytest` - Run all tests
- `python inference.py` - Run inference example
