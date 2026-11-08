"""Streamlit educational reference dashboard. Models are loaded, never retrained."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classification Comparison Lab", page_icon="📊", layout="wide")
st.title("Classification Model Comparison Lab")
st.caption("Educational reference • UCI Wisconsin Diagnostic Breast Cancer • binary classification")
st.write("Upload a held-out test CSV with all 30 input features and a `target` column. "
         "Target coding: 0 = benign, 1 = malignant. The trained models are loaded from disk.")

if not (MODEL_DIR / "metadata.json").exists():
    st.error("Model artifacts missing. Run `python train_models.py` in the project folder first.")
    st.stop()

metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]
model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
upload = st.file_uploader("Upload test CSV", type=["csv"])
with st.expander("Required CSV columns and example"):
    st.code(", ".join(feature_names + ["target"]))
    st.write("Use the `test_data.csv` produced by the training script for your first validation.")

if upload is None:
    st.info("Choose a model and upload `test_data.csv` to display its test-set results.")
    st.stop()

try:
    frame = pd.read_csv(upload)
except Exception as exc:
    st.error(f"The CSV cannot be read: {exc}")
    st.stop()

expected = feature_names + ["target"]
missing = sorted(set(expected) - set(frame.columns))
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()
if frame.empty:
    st.error("Uploaded CSV has no rows.")
    st.stop()

try:
    X_test = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(frame["target"], errors="raise").astype(int)
except (ValueError, TypeError) as exc:
    st.error(f"Numeric data validation failed: {exc}")
    st.stop()
if X_test.isna().any().any() or y_test.isna().any() or not y_test.isin([0, 1]).all():
    st.error("Input values must be numeric, finite and nonmissing; target must contain only 0 or 1.")
    st.stop()

@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)

model = load_model(MODEL_FILES[model_name])
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "AUC": roc_auc_score(y_test, prob) if y_test.nunique() == 2 else None,
    "Precision": precision_score(y_test, pred, zero_division=0),
    "Recall": recall_score(y_test, pred, zero_division=0),
    "F1": f1_score(y_test, pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, pred),
}
st.subheader(f"Evaluation — {model_name}")
metric_cols = st.columns(3)
for idx, (label, value) in enumerate(metrics.items()):
    metric_cols[idx % 3].metric(label, "N/A" if value is None else f"{value:.4f}")
st.caption("AUC is defined only when both classes occur in the uploaded test file.")

cm = confusion_matrix(y_test, pred, labels=[0, 1])
st.subheader("Confusion matrix")
st.dataframe(pd.DataFrame(cm,
                          index=["Actual benign", "Actual malignant"],
                          columns=["Predicted benign", "Predicted malignant"]),
             use_container_width=True)
st.subheader("Classification report")
report = classification_report(y_test, pred, labels=[0, 1],
                               target_names=["Benign", "Malignant"],
                               output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

st.subheader("Compare all five models on this uploaded CSV")
all_results = []
for name, filename in MODEL_FILES.items():
    loaded = load_model(filename)
    labels = loaded.predict(X_test)
    scores = loaded.predict_proba(X_test)[:, 1]
    all_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, labels),
        "AUC": roc_auc_score(y_test, scores) if y_test.nunique() == 2 else None,
        "Precision": precision_score(y_test, labels, zero_division=0),
        "Recall": recall_score(y_test, labels, zero_division=0),
        "F1": f1_score(y_test, labels, zero_division=0),
        "MCC": matthews_corrcoef(y_test, labels),
    })
st.dataframe(pd.DataFrame(all_results).set_index("Model").style.format(precision=4),
             use_container_width=True)
st.caption("All results use previously trained models. No training occurs in this app.")
