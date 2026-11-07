"""
Train five classifiers on the Online Shoppers Purchasing Intention dataset.

Run this once (inside the BITS Virtual Lab) to produce:
  - model/*.pkl          -> five fitted sklearn pipelines (preprocessing + estimator)
  - model/metadata.json  -> target name + feature columns
  - model/metrics.json   -> evaluation metrics for every model
  - test_data.csv        -> the held-out test split that is uploaded to the app

The app (app.py) only LOADS these pipelines; it never retrains.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TARGET = "Revenue"
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def load_dataset() -> pd.DataFrame:
    """Fetch the Online Shoppers Purchasing Intention dataset from the UCI repo."""
    repo = fetch_ucirepo(id=468)
    X = repo.data.features.copy()
    y = repo.data.targets.copy()
    # targets is a single-column frame; normalise its name to TARGET
    y.columns = [TARGET]
    frame = pd.concat([X, y], axis=1)
    # Revenue is boolean in the source -> store as 0/1 for consistent metrics
    frame[TARGET] = frame[TARGET].astype(int)
    frame[CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].astype(str)
    return frame


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode the categorical ones."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def make_models() -> dict:
    """The five required classifiers, keyed by display name -> file stem."""
    return {
        "Logistic Regression": (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": (
            "decision_tree",
            DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        ),
        "KNN": ("knn", KNeighborsClassifier(n_neighbors=15)),
        "Naive Bayes": ("naive_bayes", GaussianNB()),
        "Support Vector Machine": (
            "svm",
            CalibratedClassifierCV(
                estimator=SVC(random_state=RANDOM_STATE),
                method="sigmoid",
                cv=5,
                ensemble=False,
            ),
        ),
    }


def compute_metrics(y_true, y_pred, y_prob, classes) -> dict:
    """Six metrics. Handles binary and multi-class targets."""
    is_binary = len(classes) == 2
    if is_binary:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        average = "binary"
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        average = "weighted"
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(auc, 4),
        "Precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main() -> None:
    frame = load_dataset()
    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    classes = np.unique(y)
    metrics = {}
    trained_models = {}

    for name, (stem, estimator) in make_models().items():
        pipe = Pipeline(
            steps=[("preprocess", build_preprocessor(X)), ("model", estimator)]
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        metrics[name] = compute_metrics(y_test, y_pred, y_prob, classes)
        trained_models[stem] = pipe
        print(f"saved {stem}.pkl -> {metrics[name]}")

    for stem, pipe in trained_models.items():
        joblib.dump(pipe, HERE / f"{stem}.pkl")

    # Save the held-out test set (features + target) for the Streamlit app.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    # Metadata the app needs to line up columns and model files.
    metadata = {
        "target": TARGET,
        "features": X.columns.tolist(),
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": [int(c) for c in classes],
        "model_files": {name: f"{stem}.pkl" for name, (stem, _) in make_models().items()},
    }
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nWrote test_data.csv, metadata.json, metrics.json")


if __name__ == "__main__":
    main()
