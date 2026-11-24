"""Credit Card Default Risk Predictor - Streamlit app.

Run:  streamlit run app.py     ->  http://localhost:8501
"""
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.data_utils import (BILL_COLS, EDU_MAP, FEATURES, MARRIAGE_MAP, MODELS_DIR, NUMERIC, PAY_COLS,
                            PAYAMT_COLS, REPORTS_DIR, SEX_MAP, TARGET, clean, load_raw)

st.set_page_config(page_title="Credit Default Risk Predictor", page_icon="💳", layout="wide")

MODEL_FILES = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Decision Tree": MODELS_DIR / "decision_tree.joblib",
}


@st.cache_resource
def load_models():
    return {n: joblib.load(p) for n, p in MODEL_FILES.items() if p.exists()}


@st.cache_data
def load_metrics():
    p = REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_data():
    try:
        raw = load_raw()
    except (FileNotFoundError, ValueError) as e:
        return None, None, str(e)
    return raw, clean(raw), None


def show_split(df, **kw):
    """Show a tall table as two side-by-side halves so it fits in one screenshot."""
    half = (len(df) + 1) // 2
    c1, c2 = st.columns(2)
    c1.dataframe(df.iloc[:half], hide_index=kw.get("hide_index", True), width="stretch")
    c2.dataframe(df.iloc[half:], hide_index=kw.get("hide_index", True), width="stretch")


def risk_band(p):
    if p < 0.25:
        return "Low risk", "green"
    if p < 0.50:
        return "Moderate risk", "orange"
    return "High risk", "red"


models = load_models()
metrics = load_metrics()
raw_df, df, data_error = load_data()

st.title("💳 Credit Card Default Risk Predictor")
st.caption("Logistic Regression & Decision Tree trained on the Default of Credit Card Clients dataset")

if not models:
    st.error("No trained models found. Run these commands in the project folder, then refresh:")
    st.code("python -m src.train", language="bash")
    st.stop()

tab_predict, tab_data, tab_eda, tab_perf, tab_about = st.tabs(
    ["🔮 Predict", "🗂️ Dataset & preprocessing", "🔎 EDA", "📊 Model performance", "ℹ️ About"])

# ------------------------------------------------------------------ Predict
with tab_predict:
    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Lower = catch more defaulters (higher recall) but more false alarms.")
        st.caption("Lowering the threshold trades precision for recall.")

    st.subheader("Customer profile")
    c1, c2, c3, c4, c5 = st.columns(5)
    limit_bal = c1.number_input("Credit limit (LIMIT_BAL)", 10_000, 1_000_000, 100_000, 10_000)
    age = c2.number_input("Age", 18, 100, 35)
    sex = c3.selectbox("Sex", list(SEX_MAP), format_func=SEX_MAP.get)
    edu = c4.selectbox("Education", list(EDU_MAP), index=1, format_func=EDU_MAP.get)
    mar = c5.selectbox("Marital status", list(MARRIAGE_MAP), index=1, format_func=MARRIAGE_MAP.get)

    st.subheader("Repayment history")
    st.caption("-2 = no consumption, -1 = paid in full, 0 = revolving credit, 1-8 = months of payment delay "
               "(PAY_0 is the most recent month, PAY_6 the oldest)")
    pay_cols = st.columns(6)
    pay = {c: pay_cols[i].number_input(c, -2, 8, 0, key=c) for i, c in enumerate(PAY_COLS)}

    with st.expander("Bill statements (BILL_AMT1 = most recent)", expanded=False):
        cols = st.columns(6)
        bill = {c: cols[i].number_input(c, value=50_000, step=1_000, key=c) for i, c in enumerate(BILL_COLS)}
    with st.expander("Previous payments (PAY_AMT1 = most recent)", expanded=False):
        cols = st.columns(6)
        pamt = {c: cols[i].number_input(c, 0, 1_000_000, 2_000, 500, key=c) for i, c in enumerate(PAYAMT_COLS)}

    row = pd.DataFrame([{"LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar,
                         "AGE": age, **pay, **bill, **pamt}])[FEATURES]

    if st.button("Assess default risk", type="primary"):
        cols = st.columns(len(models))
        for col, (name, model) in zip(cols, models.items()):
            p = float(model.predict_proba(row)[0, 1])
            label, color = risk_band(p)
            with col:
                st.markdown(f"#### {name}")
                st.metric("Probability of default", f"{p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))
                st.markdown(f"**Risk band:** :{color}[{label}]")
                verdict = "⚠️ Likely to DEFAULT" if p >= threshold else "✅ Likely to PAY"
                st.markdown(f"**Decision @ {threshold:.2f}:** {verdict}")

# ------------------------------------------------------- Dataset & preprocessing
with tab_data:
    if data_error:
        st.warning(data_error)
    else:
        st.subheader("1. Dataset dimensions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (raw)", f"{len(raw_df):,}")
        c2.metric("Rows (after cleaning)", f"{len(df):,}", delta=f"{len(df) - len(raw_df):,} duplicates removed", delta_color="off")
        c3.metric("Columns (features + target)", f"{df.shape[1]}")
        c4.metric("Default rate", f"{df[TARGET].mean():.1%}")
        st.caption(f"Shape after cleaning: {df.shape}  |  features: {len(FEATURES)}  |  target: {TARGET}")

        st.subheader("2. Data types and missing-value check")
        total_missing = int(raw_df.isna().sum().sum())
        if total_missing == 0:
            st.success("Missing-value check: 0 missing values in any column.")
        else:
            st.warning(f"Missing-value check: {total_missing:,} missing values found.")
        info = pd.DataFrame({
            "Column": raw_df.columns,
            "Data type": raw_df.dtypes.astype(str).values,
            "Non-null": raw_df.notna().sum().values,
            "Missing": raw_df.isna().sum().values,
            "Unique": raw_df.nunique().values,
        })
        show_split(info)

        st.subheader("3. Preprocessing applied")
        st.markdown(
            "- Dropped the `ID` column and removed duplicate rows\n"
            "- Recoded undocumented codes: `EDUCATION` 0/5/6 to 4 (Other), `MARRIAGE` 0 to 3 (Other)\n"
            "- `StandardScaler` on the 20 numeric features (mean 0, standard deviation 1)\n"
            "- One-hot encoding for `SEX`, `EDUCATION`, `MARRIAGE`\n"
            "- Stratified 80/20 train/test split (scaler is fitted inside the training pipeline only)")

        st.subheader("4. Scaled-feature summary")
        scaled = pd.DataFrame(StandardScaler().fit_transform(df[NUMERIC]), columns=NUMERIC)
        sc = scaled.describe().T[["mean", "std", "min", "max"]].round(3)
        sc.insert(0, "Feature", sc.index)
        show_split(sc)
        st.caption("After scaling, every numeric feature has mean ~ 0 and standard deviation ~ 1.")

        st.subheader("5. Sample rows (first 5)")
        st.dataframe(df.head(), width="stretch")

# -------------------------------------------------------------- Performance
with tab_perf:
    if metrics:
        st.subheader("Test-set metrics")
        table = pd.DataFrame({n: m["metrics"] for n, m in metrics["models"].items()}).T
        st.dataframe(table.style.format("{:.3f}"))
        st.caption(f"Data: {metrics['n_rows']:,} rows | default rate {metrics['default_rate']:.1%} | "
                   f"class_weight = {metrics['class_weight']} | best params: "
                   + "; ".join(f"{n}: {m['best_params']}" for n, m in metrics["models"].items()))
        for img, cap in [("metric_comparison.png", "Metric comparison"),
                         ("confusion_matrices.png", "Confusion matrices"),
                         ("roc_curves.png", "ROC curves")]:
            if (REPORTS_DIR / img).exists():
                st.image(str(REPORTS_DIR / img), caption=cap)
        c1, c2 = st.columns(2)
        if (REPORTS_DIR / "lr_coefficients.png").exists():
            c1.image(str(REPORTS_DIR / "lr_coefficients.png"))
        if (REPORTS_DIR / "dt_feature_importance.png").exists():
            c2.image(str(REPORTS_DIR / "dt_feature_importance.png"))
        if (REPORTS_DIR / "decision_tree.png").exists():
            st.image(str(REPORTS_DIR / "decision_tree.png"), caption="Decision tree (first 3 levels)")
    else:
        st.info("Run `python -m src.train` to generate performance reports.")

# ---------------------------------------------------------------------- EDA
with tab_eda:
    if df is not None:
        st.subheader("Descriptive statistics")
        desc = df.describe().T.round(2)
        desc.insert(0, "Feature", desc.index)
        st.dataframe(desc, hide_index=True, width="stretch", height=(len(desc) + 1) * 35 + 3)
        st.subheader("Diagnostic plots")
    imgs = sorted((REPORTS_DIR / "eda").glob("*.png"))
    if imgs:
        for p in imgs:
            st.image(str(p), caption=p.stem.split("_", 1)[1].replace("_", " ").title())
    else:
        st.info("Run `python -m src.eda` to generate EDA charts.")

# -------------------------------------------------------------------- About
with tab_about:
    st.markdown("""
**Why precision, recall and F1 instead of accuracy?**
Only about 22% of customers in the real dataset default, so a model that predicts "no default" for everyone is
already ~78% accurate while being useless.

- **Precision** - of the customers flagged as defaulters, how many truly default (cost of false alarms).
- **Recall** - of the customers who truly default, how many were caught (cost of missed defaults).
- **F1** - harmonic mean of the two.

Use the *decision threshold* slider in the sidebar to move along the precision/recall trade-off.

*This is an educational project, not a production credit-decisioning system.*
""")
