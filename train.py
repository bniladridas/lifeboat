import os

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = "titanic.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load Titanic dataset from online source.

    Returns:
        pd.DataFrame: Raw Titanic dataset with passenger information
    """
    df = pd.read_csv(TITANIC_URL)
    print(f"Loaded {len(df)} samples")
    return df


def preprocess(df):
    """Preprocess Titanic data and engineer features.

    Args:
        df: Raw Titanic DataFrame

    Returns:
        tuple: (X, y, features) - Features, target, and feature names
    """
    df = df.copy()

    df.loc[:, "Age"] = df["Age"].fillna(df["Age"].median())
    df.loc[:, "Fare"] = df["Fare"].fillna(df["Fare"].median())
    df.loc[:, "Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df.loc[:, "AgeBin"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]
    ).astype(float)

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
    df["Embarked_encoded"] = le_embarked.fit_transform(df["Embarked"])

    joblib.dump(le_sex, f"{MODEL_DIR}/le_sex.pkl")
    joblib.dump(le_embarked, f"{MODEL_DIR}/le_embarked.pkl")

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

    X = df[features]
    y = df["Survived"]

    return X, y, features


def train_models(X_train, y_train):
    """Train multiple classification models.

    Args:
        X_train: Scaled training features
        y_train: Training labels

    Returns:
        dict: Model results with trained model, CV mean, and CV std
    """
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        print(f"{name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


def evaluate_and_save(models, X_test, y_test):
    """Evaluate models and save the best one.

    Args:
        models: Dictionary of trained models
        X_test: Scaled test features
        y_test: Test labels

    Returns:
        tuple: (best_model_name, best_model)
    """
    best_model = None
    best_score = 0

    for name, result in models.items():
        model = result["model"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n{name}:")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test AUC-ROC: {auc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = (name, model)

    print(f"\nBest model: {best_model[0]} with accuracy {best_score:.4f}")
    joblib.dump(best_model[1], f"{MODEL_DIR}/titanic_model.pkl")

    return best_model[0], best_model[1]


def main():
    print("=" * 50)
    print("Titanic Survival Prediction Model Training")
    print("=" * 50)

    df = load_data()
    X, y, features = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    models = train_models(X_train_scaled, y_train)
    best_name, best_model = evaluate_and_save(models, X_test_scaled, y_test)

    joblib.dump(features, f"{MODEL_DIR}/features.pkl")

    print(f"\n{'=' * 50}")
    print("Training complete!")
    print(f"Model saved to: {MODEL_DIR}/titanic_model.pkl")
    print(f"Features: {features}")
    print("=" * 50)


if __name__ == "__main__":
    main()
    import version

    version.bump_version()
