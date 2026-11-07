"""
Online Shoppers Purchasing Intention - model comparison dashboard.

The app LOADS the five pre-trained pipelines from model/ and evaluates them on
an uploaded test CSV. It never retrains. Upload test_data.csv (features + the
Revenue target column) to reproduce the metrics reported in the README.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(
    page_title="Online Shoppers Intention - Classifier Comparison",
    page_icon="🛒",
    layout="wide",
)


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text())


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


def coerce_target(series: pd.Series) -> pd.Series:
    """Accept Revenue as 0/1, True/False, or 'True'/'False' text."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
    return series


def evaluate(model, X: pd.DataFrame, y_true: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    classes = np.unique(y_true)
    is_binary = len(classes) == 2
    average = "binary" if is_binary else "weighted"

    y_prob = model.predict_proba(X)
    if is_binary:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred, confusion_matrix(y_true, y_pred)


meta = load_metadata()
TARGET = meta["target"]
FEATURES = meta["features"]
MODEL_FILES = meta["model_files"]

# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("🛒 Online Shoppers Intention")
st.sidebar.markdown(
    "Predict whether a browsing session ends in a **purchase** (`Revenue`). "
    "Five classifiers are pre-trained; this dashboard only evaluates them on "
    "your uploaded test data."
)
st.sidebar.markdown("---")
st.sidebar.subheader("1. Upload test CSV")
uploaded = st.sidebar.file_uploader(
    "Must contain the 17 feature columns + the `Revenue` target.", type="csv"
)

sample_path = Path(__file__).parent / "test_data.csv"
if sample_path.exists():
    st.sidebar.download_button(
        "⬇️ Download sample test_data.csv",
        data=sample_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

st.sidebar.subheader("2. Select a model")
choice = st.sidebar.selectbox("Classifier", list(MODEL_FILES.keys()))

# ------------------------------------------------------------------ main ------
st.title("Classifier Comparison Dashboard")
st.caption(
    "Dataset: UCI Online Shoppers Purchasing Intention — 12,330 sessions, "
    "17 features, binary target (Revenue)."
)

if uploaded is None:
    st.info(
        "👈 Upload a test CSV (or download the bundled `test_data.csv`) to see "
        "evaluation metrics, a confusion matrix, and a side-by-side comparison "
        "of all five models."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no rows. Upload a non-empty test dataset.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

X = df[FEATURES].copy()
for column in meta.get("categorical_features", []):
    X[column] = X[column].astype(str)
y = coerce_target(df[TARGET])

if y.isna().any() or not set(y.unique()).issubset({0, 1}):
    st.error("`Revenue` must contain only 0/1 or True/False values; no values may be blank.")
    st.stop()

if y.nunique() != 2:
    st.error("The uploaded test data must contain both target classes (0 and 1) to calculate AUC.")
    st.stop()

missing_values = X.isna().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    st.error(f"Feature values are missing: {missing_values.to_dict()}. Clean these before evaluation.")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head(), width="stretch")
st.write(f"Rows: **{len(df)}** · Positive (purchase) rate: **{y.mean():.1%}**")

tab_single, tab_compare = st.tabs(["🔍 Selected model", "📊 Compare all models"])

with tab_single:
    st.subheader(f"{choice}")
    model = load_model(MODEL_FILES[choice])
    metrics, y_pred, cm = evaluate(model, X, y)

    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No buy", "Buy"], yticklabels=["No buy", "Buy"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

with tab_compare:
    st.subheader("All five models on the uploaded test data")
    rows = {}
    for name, filename in MODEL_FILES.items():
        m, _, _ = evaluate(load_model(filename), X, y)
        rows[name] = m
    table = pd.DataFrame(rows).transpose().round(4)
    st.dataframe(
        table.style.highlight_max(axis=0, color="#c6f6d5"),
        width="stretch",
    )
    st.bar_chart(table[["F1", "MCC", "AUC"]])
