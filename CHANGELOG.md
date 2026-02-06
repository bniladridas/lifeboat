# Changelog

All notable changes to the Lifeboat project are documented here.

## [Unreleased]

## [1.0.0] - 2026-02-06

### Added
- **PR #3**: Complete lifeboat notebook with EDA, training, and inference
  - Added `lifeboat.ipynb` - comprehensive notebook covering data exploration, model training, and prediction
  - Added matplotlib and seaborn to requirements for visualizations
  - GradientBoosting model trained (81.01% accuracy, 0.8088 AUC-ROC)
  - Matches ModelScope deployment configuration

- **PR #2**: Test and e2e workflow
  - Added test infrastructure
  - Added end-to-end workflow for CI/CD

### Changed
- Updated requirements.txt with visualization dependencies

## [0.1.0] - Initial Release

### Added
- Titanic survival prediction model
- Python SDK (`lifeboat_sdk/`)
- ModelScope upload configuration
- Inference API
- Training pipeline (`train.py`)

---

**Model Performance:**
- Algorithm: GradientBoostingClassifier
- Test Accuracy: 81.01%
- Test AUC-ROC: 0.8088

**Features Used:**
- Pclass, Sex, Age, SibSp, Parch, Fare
- Embarked, FamilySize, IsAlone, FarePerPerson, AgeBin
