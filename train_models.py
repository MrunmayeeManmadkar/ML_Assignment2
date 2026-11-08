"""Educational reference: fit five classifiers on the UCI WDBC dataset.

Run this on your BITS Virtual Lab and document your own experiments.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

# sklearn's load_breast_cancer is the Wisconsin Diagnostic Breast Cancer dataset
# originally provided by the UCI Machine Learning Repository.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data.copy()
# Positive class is malignant (1), negative class benign (0).
y = (dataset.target == 0).astype(int)
y.name = "target"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipelines bundle preprocessing with models, preventing leakage and
# ensuring consistent transformations during inference.
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ]),
    "KNN": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("model", GaussianNB()),
    ]),
    "Support Vector Machine": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

results = []
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    artifact_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, OUT / artifact_name)
    print(f"{name}: confusion matrix = {confusion_matrix(y_test, pred).tolist()}")

# Preserve feature and class schema for validation and reproducibility.
metadata = {
    "feature_names": X.columns.tolist(),
    "target_name": "target",
    "class_labels": {"0": "Benign", "1": "Malignant"},
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "split_seed": 42,
    "dataset_source": "UCI Breast Cancer Wisconsin (Diagnostic)",
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
test_data = X_test.copy()
test_data["target"] = y_test
(test_data).to_csv(ROOT / "test_data.csv", index=False)

report = pd.DataFrame(results).sort_values("F1", ascending=False)
report.to_csv(ROOT / "comparison_metrics.csv", index=False)
print("\nComparison table (positive class: malignant):")
print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nSaved test_data.csv, comparison_metrics.csv, model/*.joblib, model/metadata.json")
