"""ML Model Studio — polished Streamlit classification dashboard.

Uses pre-trained pipelines; uploaded test data is NEVER used for fitting.
Run `python train_models.py` once before launching this app.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MODEL_NOTES = {
    "Logistic Regression": "Linear decision boundary · standardized features",
    "Decision Tree": "Interpretable feature splits · depth-limited tree",
    "KNN": "Distance-based classifier · k = 5",
    "Gaussian Naive Bayes": "Probabilistic baseline · Gaussian feature assumption",
    "Support Vector Machine": "Nonlinear RBF kernel · standardized features",
}

st.set_page_config(page_title="ML Model Studio | Classification Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS is only visual: core inputs, charts, and metrics use native Streamlit.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --navy:#101b32; --teal:#0e9488; --muted:#6b7890; }
html,body,[class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background: #f6f8fc;color:var(--navy);}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:2.1rem; padding-bottom:3rem;max-width:1440px;}
[data-testid="stSidebar"] {background:#101b32;}
[data-testid="stSidebar"] * {color:#e6edf7;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {background:#1b2b47;border:1px dashed #6a809b;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {color:#11233c !important;}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {background:#1e304f; border-color:#425674;}
[data-testid="stSidebar"] hr {border-color:#35465d;}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;color:#14223a;}
h1 {font-weight:800 !important;}
h2,h3 {font-weight:750 !important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e6ebf2;border-radius:16px;padding:18px 20px;box-shadow:0 5px 18px rgba(25,42,77,.035);}
[data-testid="stMetricLabel"] {font-weight:700;color:#728097;font-size:.88rem;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;color:#182b45;font-size:1.8rem;font-weight:800;}
[data-testid="stTabs"] button {font-weight:700;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color:#0e9488;}
[data-testid="stTabs"] [aria-selected="true"] {color:#087a72;}
[data-testid="stDataFrame"] {border:1px solid #e6ebf2;border-radius:14px;overflow:hidden;}
.hero {padding:28px 34px;border-radius:22px;background:linear-gradient(120deg,#14253f 0%,#1b3e55 65%,#146c6b 140%);color:white;margin-bottom:18px;box-shadow:0 12px 28px rgba(15,34,57,.12);}
.hero .eyebrow {font-weight:800;font-size:11px;letter-spacing:2.1px;color:#91e2d7;text-transform:uppercase;margin-bottom:12px;}
.hero h1 {color:#fff;font-size:2.15rem;letter-spacing:-.05em;margin:0 0 9px 0;}
.hero p {color:#c9d9e7;font-size:1rem;margin:0;max-width:740px;line-height:1.65;}
.hero .hero-tag {display:inline-block;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);border-radius:30px;padding:5px 11px;margin-top:19px;margin-right:9px;color:#dff6f2;font-weight:600;font-size:12px;}
.section-label {text-transform:uppercase;letter-spacing:1.5px;font-size:11px;font-weight:800;color:#7a8ba5;margin:19px 0 9px 0;}
.kicker {font-size:12px;font-weight:800;color:#0e9488;letter-spacing:1.3px;text-transform:uppercase;}
.panel {background:#fff;border:1px solid #e6ebf2;border-radius:17px;padding:20px 23px;margin:12px 0;}
.panel h3 {margin:0 0 8px 0;font-size:1.05rem;}
.panel p {color:#64758c;line-height:1.6;margin:0;}
.status-pill {display:inline-block;border-radius:30px;background:#dff8f0;color:#117a67;font-weight:800;font-size:12px;padding:6px 11px;}
.side-brand {font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;color:#fff;margin:5px 0 4px;}
.side-sub {color:#99acc5;font-size:.84rem;margin-bottom:24px;line-height:1.5;}
.side-detail {padding:12px 13px;border:1px solid #394d6a;border-radius:12px;background:#1b2c47;color:#c9d7e8;font-size:13px;line-height:1.6;}
.footer {text-align:center;font-size:12px;color:#8491a7;margin-top:32px;padding-top:16px;border-top:1px solid #e2e8f0;}
@media (max-width:650px) {.hero{padding:20px}.hero h1{font-size:1.65rem}.block-container{padding-top:1rem}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def evaluate(model, X, y):
    pred = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, scores) if y.nunique() == 2 else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred


# Sidebar remains useful even before loading data.
with st.sidebar:
    st.markdown('<div class="side-brand">◈ &nbsp; ML Model Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Classification experiments<br>Interactive evaluation workspace</div>', unsafe_allow_html=True)
    st.markdown('#### 01 &nbsp; Select a model')
    chosen = st.selectbox("Classification algorithm", MODEL_FILES.keys(), index=4, label_visibility="collapsed")
    st.caption(MODEL_NOTES[chosen])
    st.divider()
    st.markdown('#### 02 &nbsp; Upload test dataset')
    uploaded = st.file_uploader("Upload a CSV file", type="csv", help="Test data must include 30 input features and a 'target' column.")
    if (ROOT / "test_data.csv").exists():
        st.download_button("↓ Download sample test CSV", (ROOT / "test_data.csv").read_bytes(),
                           file_name="test_data.csv", mime="text/csv", use_container_width=True)
    st.divider()
    st.markdown('<div class="side-detail"><strong>DATASET</strong><br>Wisconsin Diagnostic Breast Cancer<br>569 samples · 30 features · 2 classes<br><br><strong>LABELS</strong><br>0 — Benign<br>1 — Malignant (positive)</div>', unsafe_allow_html=True)
    st.caption("Pre-trained models only · No retraining on uploads")

st.markdown('''<div class="hero"><div class="eyebrow">Machine Learning · Assignment 02</div>
<h1>Classification Model Studio</h1>
<p>Explore five machine learning algorithms, compare their predictive performance, and inspect their decisions on a held-out test dataset.</p>
<span class="hero-tag">05 trained models</span><span class="hero-tag">06 evaluation metrics</span><span class="hero-tag">Interactive analysis</span>
</div>''', unsafe_allow_html=True)

if not (MODEL_DIR / "metadata.json").exists():
    st.error("Saved models are missing. Run `python train_models.py` in this folder, then restart the application.")
    st.stop()

metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
features = metadata["feature_names"]

if uploaded is None:
    st.markdown('<div class="section-label">Getting started</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="panel"><h3>01 · Select</h3><p>Pick one of five trained classifiers using the sidebar selector.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><h3>02 · Upload</h3><p>Use the sample test CSV in the sidebar or your own compatible held-out data.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><h3>03 · Explore</h3><p>View model scores, a confusion matrix, and a five-model benchmark.</p></div>', unsafe_allow_html=True)
    st.info("To begin, upload `test_data.csv` using the left sidebar. The dataset must include all 30 feature columns and `target`.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Couldn't read the CSV: {exc}")
    st.stop()

missing = [c for c in features + ["target"] if c not in df.columns]
if missing:
    st.error("Missing required CSV columns: " + ", ".join(missing))
    st.stop()
if df.empty:
    st.error("Your CSV is empty. Upload a file with at least one labeled example.")
    st.stop()
try:
    X = df[features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df["target"], errors="raise")
except (ValueError, TypeError) as exc:
    st.error(f"All feature and target values must be numeric: {exc}")
    st.stop()
if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
    st.error("Feature and target values must be finite and nonmissing.")
    st.stop()
if not y.isin([0, 1]).all():
    st.error("Target values must be only 0 (benign) or 1 (malignant).")
    st.stop()
y = y.astype(int)

try:
    comparisons = []
    predictions = {}
    for model_name, filename in MODEL_FILES.items():
        result, pred = evaluate(load_model(filename), X, y)
        predictions[model_name] = pred
        comparisons.append({"Model": model_name, **result})
except Exception as exc:
    st.error(f"Unable to evaluate the saved models: {exc}")
    st.stop()

comparison = pd.DataFrame(comparisons).sort_values("F1", ascending=False).reset_index(drop=True)
selected = next(row for row in comparisons if row["Model"] == chosen)
pred = predictions[chosen]
best = comparison.iloc[0]["Model"]

st.markdown('<div class="section-label">Live evaluation</div>', unsafe_allow_html=True)
left, right = st.columns([3, 1])
with left:
    st.subheader(chosen)
    st.caption(MODEL_NOTES[chosen])
with right:
    st.markdown('<div style="text-align:right;padding-top:12px"><span class="status-pill">● Evaluation complete</span></div>', unsafe_allow_html=True)

metric_cols = st.columns(3)
for i, metric in enumerate(["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = selected[metric]
    metric_cols[i % 3].metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

st.caption(f"{len(df)} test rows · {len(features)} input features · {int((y == 1).sum())} malignant / {int((y == 0).sum())} benign. AUC requires both target classes.")

tab1, tab2, tab3 = st.tabs(["◈  Model insights", "▦  All-model comparison", "▤  Test data"])

with tab1:
    st.markdown('<div class="section-label">Prediction diagnostics</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1.15, 0.85], gap="large")
    with c1:
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        names = ["Benign", "Malignant"]
        cells = pd.DataFrame([
            {"Actual":names[r], "Predicted":names[c], "Count":int(cm[r,c]),
             "Cell": f"{int(cm[r,c])}"}
            for r in range(2) for c in range(2)
        ])
        heatmap = alt.Chart(cells).mark_rect(cornerRadius=7).encode(
            x=alt.X("Predicted:N", sort=names, title="Predicted class", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", sort=names, title="Actual class"),
            color=alt.Color("Count:Q", scale=alt.Scale(range=["#e3f6f2", "#0d8e84"]), legend=None),
            tooltip=["Actual", "Predicted", "Count"],
        ).properties(height=265)
        labels = alt.Chart(cells).mark_text(fontSize=23, fontWeight="bold").encode(
            x=alt.X("Predicted:N", sort=names),y=alt.Y("Actual:N", sort=names),
            text="Cell:N", color=alt.value("#12364b"),
        )
        st.altair_chart(heatmap + labels, use_container_width=True)
    with c2:
        st.markdown("#### What the results mean")
        tn,fp,fn,tp = cm.ravel()
        st.metric("Correct predictions", f"{tn + tp} / {len(y)}")
        st.metric("False negatives", f"{fn}", help="Malignant samples predicted benign")
        st.metric("False positives", f"{fp}", help="Benign samples predicted malignant")
        st.caption("For medical classification, false negatives are important to inspect. This educational project is not a clinical diagnostic tool.")
    st.markdown("#### Classification report")
    report = classification_report(y, pred, labels=[0, 1], target_names=names,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with tab2:
    st.markdown('<div class="section-label">Leaderboard</div>', unsafe_allow_html=True)
    st.markdown("#### Side-by-side model performance")
    st.caption(f"Ranked by F1 score on this uploaded dataset. Top performer: **{best}**.")
    st.dataframe(comparison.style.format({m:"{:.4f}" for m in ["Accuracy","AUC","Precision","Recall","F1","MCC"]}, na_rep="N/A"),
                 use_container_width=True, hide_index=True)
    metric_choice = st.selectbox("Visualize metric", ["F1", "Accuracy", "AUC", "Precision", "Recall", "MCC"], index=0)
    chart_df = comparison[["Model", metric_choice]].dropna().copy()
    bars = alt.Chart(chart_df).mark_bar(size=29, cornerRadiusEnd=6).encode(
        x=alt.X(f"{metric_choice}:Q", title=metric_choice, scale=alt.Scale(zero=True)),
        y=alt.Y("Model:N", sort=alt.SortField(field=metric_choice, order="descending"), title=None),
        color=alt.condition(alt.datum.Model == chosen, alt.value("#0e9488"), alt.value("#8099b1")),
        tooltip=["Model", alt.Tooltip(f"{metric_choice}:Q", format=".4f")],
    ).properties(height=275)
    st.altair_chart(bars, use_container_width=True)
    st.download_button("↓ Export current comparison (CSV)", comparison.to_csv(index=False).encode("utf-8"),
                       "model_comparison.csv", "text/csv")

with tab3:
    st.markdown('<div class="section-label">Dataset explorer</div>', unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    col1.metric("Test samples", len(df))
    col2.metric("Input features", len(features))
    col3.metric("Target classes", y.nunique())
    st.markdown("#### Uploaded data preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    with st.expander("See all required input columns"):
        st.code(", ".join(features + ["target"]))
    st.caption("Only uploaded test records are evaluated. No model is retrained or fitted in this application.")

st.markdown('<div class="footer">ML Model Studio · UCI Wisconsin Diagnostic Breast Cancer · Educational demonstration only</div>', unsafe_allow_html=True)
