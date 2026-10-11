"""BugScope – Software Defect Prediction Dashboard."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "software_metrics.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="BugScope | Defect Prediction",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background: #0f0f14; color: #f4f4f5; }
  h1, h2, h3 { color: #e4e4e7 !important; letter-spacing: -0.02em; }
  div[data-testid="stMetricValue"] { color: #a78bfa; }
  .stTabs [aria-selected="true"] { background: #7c3aed !important; color: #fff !important; }
  footer { visibility: hidden; }
  .block-container { padding-top: 1.25rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    arts = {
        "model": joblib.load(MODEL_DIR / "defect_model.joblib"),
        "encoders": joblib.load(MODEL_DIR / "encoders.joblib"),
        "scaler": joblib.load(MODEL_DIR / "scaler.joblib"),
        "selector": joblib.load(MODEL_DIR / "selector.joblib"),
        "feature_cols": joblib.load(MODEL_DIR / "feature_cols.joblib"),
        "selected_features": joblib.load(MODEL_DIR / "selected_features.joblib"),
        "num_cols": joblib.load(MODEL_DIR / "num_cols.joblib"),
        "cat_cols": joblib.load(MODEL_DIR / "cat_cols.joblib"),
    }
    with open(MODEL_DIR / "metrics.json") as f:
        arts["metrics"] = json.load(f)
    return arts


def style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d4d4d8", title_font_color="#f4f4f5",
    )
    return fig


def preprocess_row(row, arts):
    df = pd.DataFrame([row])
    for c in arts["cat_cols"]:
        le = arts["encoders"][c]
        known = set(le.classes_)
        val = str(df[c].iloc[0])
        if val not in known:
            val = le.classes_[0]
        df[c] = le.transform([val])
    df[arts["num_cols"]] = arts["scaler"].transform(df[arts["num_cols"]])
    X = df[arts["feature_cols"]].values
    return arts["selector"].transform(X)


# Sidebar
with st.sidebar:
    st.markdown("## BugScope")
    st.caption("Software defect prediction")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Explore", "Pipeline", "Predict", "Data", "API"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    try:
        arts = load_models()
        m = arts["metrics"]
        st.success(f"Model online · {m['best_model']}")
        st.caption(f"F1 {m['results'][m['best_model']]['f1']:.3f} · AUC {m['results'][m['best_model']]['roc_auc']:.3f}")
    except Exception:
        arts = None
        st.error("Models unavailable")
    st.caption("Quality intelligence for engineering teams")

df = load_data()

# ---- Overview ----
if page == "Overview":
    st.title("BugScope")
    st.subheader("Software defect prediction platform")
    st.markdown("Identify high-risk modules using static metrics and process signals.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules", f"{len(df):,}")
    c2.metric("Defect rate", f"{df['defective'].mean()*100:.1f}%")
    c3.metric("Avg complexity", f"{df['cyclomatic_complexity'].mean():.1f}")
    c4.metric("Avg LOC", f"{df['loc'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            df, names=df["defective"].map({0: "Clean", 1: "Defective"}),
            title="Module status", color=df["defective"].map({0: "Clean", 1: "Defective"}),
            color_discrete_map={"Clean": "#34d399", "Defective": "#f87171"}, hole=0.45,
        )
        st.plotly_chart(style(fig), use_container_width=True)
    with col2:
        sev = df[df["defective"] == 1]["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.bar(sev, x="severity", y="count", title="Severity among defective modules",
                     color="count", color_continuous_scale="Purples")
        st.plotly_chart(style(fig), use_container_width=True)

    st.subheader("Complexity vs defect status")
    sample = df.sample(min(2500, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="cyclomatic_complexity", y="loc", color=sample["defective"].map({0: "Clean", 1: "Defective"}),
        size="code_churn", color_discrete_map={"Clean": "#34d399", "Defective": "#f87171"},
        title="Complexity × size (bubble = churn)", hover_data=["language", "module_type"],
        opacity=0.7,
    )
    st.plotly_chart(style(fig), use_container_width=True)

# ---- Explore ----
elif page == "Explore":
    st.title("Explore")
    tabs = st.tabs(["Summary", "Distributions", "Correlations", "By language"])

    with tabs[0]:
        st.write("Shape:", df.shape)
        st.dataframe(df.describe().T, use_container_width=True)
        st.dataframe(df.head(12), use_container_width=True)

    with tabs[1]:
        cols = st.multiselect(
            "Numeric features",
            ["loc", "cyclomatic_complexity", "halstead_effort", "branch_count",
             "code_churn", "review_coverage", "test_coverage", "team_experience"],
            default=["cyclomatic_complexity", "loc", "code_churn"],
        )
        if cols:
            fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
            for i, c in enumerate(cols):
                fig.add_trace(go.Histogram(x=df[c], marker_color="#8b5cf6", name=c), row=1, col=i+1)
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(style(fig), use_container_width=True)

    with tabs[2]:
        num = df.select_dtypes(include=[np.number])
        fig = px.imshow(num.corr(), aspect="auto", color_continuous_scale="RdBu_r",
                        title="Correlation heatmap", text_auto=".1f")
        st.plotly_chart(style(fig), use_container_width=True)

    with tabs[3]:
        fig = px.histogram(df, x="language", color=df["defective"].map({0: "Clean", 1: "Defective"}),
                           barmode="group", title="Defects by language",
                           color_discrete_map={"Clean": "#34d399", "Defective": "#f87171"})
        st.plotly_chart(style(fig), use_container_width=True)
        fig2 = px.box(df, x="module_type", y="cyclomatic_complexity", color="module_type",
                      title="Complexity by module type")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(style(fig2), use_container_width=True)

# ---- Pipeline ----
elif page == "Pipeline":
    st.title("ML Pipeline")
    st.markdown("Preprocessing → feature selection → training → evaluation")

    st.header("Preprocessing")
    st.code("Label encoding for language & module_type\nStandardScaler for numeric metrics\nStratified 80/20 train-test split")

    if arts:
        m = arts["metrics"]
        st.header("Feature selection")
        st.write("Method: SelectKBest + Mutual Information")
        st.write("Selected features:", m["selected_features"])

        if m.get("feature_importances"):
            imp = pd.DataFrame({
                "feature": list(m["feature_importances"].keys()),
                "importance": list(m["feature_importances"].values()),
            }).sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         title="Feature importance", color="importance",
                         color_continuous_scale="Purples")
            st.plotly_chart(style(fig), use_container_width=True)

        st.header("Evaluation")
        best = m["best_model"]
        st.success(f"Best model: **{best}**")
        res = m["results"]
        cols = st.columns(len(res))
        for i, (name, r) in enumerate(res.items()):
            with cols[i]:
                st.metric(name, f"F1 {r['f1']:.3f}", f"AUC {r['roc_auc']:.3f}")

        cm = np.array(res[best]["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Clean", "Defective"], y=["Clean", "Defective"],
                        color_continuous_scale="Purples", title="Confusion matrix",
                        labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(style(fig), use_container_width=True)

        st.dataframe(pd.DataFrame(res[best]["report"]).T.style.format("{:.3f}"), use_container_width=True)
    else:
        st.warning("Train models first: python train_models.py")

# ---- Predict ----
elif page == "Predict":
    st.title("Predict defect risk")
    st.markdown("Score a module using static and process metrics.")

    if arts is None:
        st.error("Models not loaded.")
        st.stop()

    with st.form("score"):
        st.subheader("Size & structure")
        a, b, c = st.columns(3)
        with a:
            loc = st.number_input("LOC", 10, 50000, 800)
            loc_executable = st.number_input("Executable LOC", 5, 45000, 600)
            loc_comment = st.number_input("Comment LOC", 0, 10000, 80)
        with b:
            loc_blank = st.number_input("Blank LOC", 0, 5000, 50)
            num_lines = st.number_input("Total lines", 10, 55000, 930)
            cyclomatic_complexity = st.number_input("Cyclomatic complexity", 1.0, 100.0, 12.0, 0.5)
        with c:
            essential_complexity = st.number_input("Essential complexity", 1.0, 80.0, 6.0, 0.5)
            design_complexity = st.number_input("Design complexity", 1.0, 55.0, 5.0, 0.5)
            branch_count = st.number_input("Branch count", 0, 250, 15)

        st.subheader("Halstead & graph")
        d, e, f = st.columns(3)
        with d:
            decision_count = st.number_input("Decision count", 0, 300, 12)
            edge_count = st.number_input("Edge count", 0, 500, 40)
            node_count = st.number_input("Node count", 2, 400, 25)
        with e:
            num_operators = st.number_input("Operators", 1, 100000, 1200)
            num_operands = st.number_input("Operands", 1, 80000, 900)
            num_unique_operators = st.number_input("Unique operators", 5, 100, 25)
        with f:
            num_unique_operands = st.number_input("Unique operands", 5, 250, 60)
            halstead_volume = st.number_input("Halstead volume", 1.0, 1e7, 5000.0)
            ind = st.number_input("Halstead effort", 1.0, 1e8, 8000.0)

        st.subheader("Process & context")
        g, h, i = st.columns(3)
        with g:
            halstead_difficulty = st.number_input("Halstead difficulty", 1.0, 50.0, 12.0, 0.1)
            call_pairs = st.number_input("Call pairs", 0, 100, 8)
            condition_count = st.number_input("Condition count", 0, 120, 10)
        with h:
            parameter_count = st.number_input("Parameters", 0, 20, 3)
            path_count = st.number_input("Path count", 1, 8000, 40)
            percent_comments = st.number_input("% comments", 0.0, 100.0, 10.0, 0.5)
        with i:
            avg_block_length = st.number_input("Avg block length", 1.0, 50.0, 12.0, 0.5)
            code_churn = st.number_input("Code churn", 0.0, 250.0, 15.0, 0.5)
            review_coverage = st.slider("Review coverage", 0.0, 1.0, 0.7, 0.01)

        j, k, l = st.columns(3)
        with j:
            test_coverage = st.slider("Test coverage", 0.0, 1.0, 0.65, 0.01)
            team_experience = st.slider("Team experience", 0.0, 1.0, 0.6, 0.01)
        with k:
            language = st.selectbox("Language", ["Java", "C++", "Python", "C#", "JavaScript"])
        with l:
            module_type = st.selectbox("Module type", ["Core", "UI", "API", "Data", "Utility", "Test"])

        go_pred = st.form_submit_button("Score module", use_container_width=True)

    if go_pred:
        row = {
            "loc": loc, "loc_executable": loc_executable, "loc_comment": loc_comment,
            "loc_blank": loc_blank, "num_lines": num_lines,
            "cyclomatic_complexity": cyclomatic_complexity,
            "essential_complexity": essential_complexity, "design_complexity": design_complexity,
            "branch_count": branch_count, "decision_count": decision_count,
            "edge_count": edge_count, "node_count": node_count,
            "num_operators": num_operators, "num_operands": num_operands,
            "num_unique_operators": num_unique_operators, "num_unique_operands": num_unique_operands,
            "halstead_volume": halstead_volume, "halstead_effort": ind,
            "halstead_difficulty": halstead_difficulty, "call_pairs": call_pairs,
            "condition_count": condition_count, "parameter_count": parameter_count,
            "path_count": path_count, "percent_comments": percent_comments,
            "avg_block_length": avg_block_length, "code_churn": code_churn,
            "review_coverage": review_coverage, "test_coverage": test_coverage,
            "team_experience": team_experience, "language": language, "module_type": module_type,
        }
        X = preprocess_row(row, arts)
        pred = int(arts["model"].predict(X)[0])
        proba = float(arts["model"].predict_proba(X)[0][1])

        st.markdown("---")
        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Prediction", "Defective" if pred else "Clean")
        c2.metric("Defect probability", f"{proba*100:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            title={"text": "Defect risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8b5cf6"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(52,211,153,0.25)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.25)"},
                    {"range": [60, 100], "color": "rgba(248,113,113,0.25)"},
                ],
            },
        ))
        fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="#d4d4d8")
        st.plotly_chart(fig, use_container_width=True)

        if proba >= 0.6 or pred:
            st.error("Elevated defect risk — prioritize review, tests, and refactoring.")
        elif proba >= 0.35:
            st.warning("Moderate risk — increase review coverage and targeted tests.")
        else:
            st.success("Low predicted risk — maintain current quality practices.")

# ---- Data ----
elif page == "Data":
    st.title("Dataset")
    st.markdown("Module-level software metrics with defect labels.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", len(df.columns) - 3)
    c3.metric("Defect rate", f"{df['defective'].mean()*100:.1f}%")

    docs = pd.DataFrame([
        {"Feature": "loc / loc_executable / …", "Group": "Size", "Description": "Lines of code and related counts"},
        {"Feature": "cyclomatic_complexity", "Group": "Complexity", "Description": "McCabe cyclomatic complexity"},
        {"Feature": "halstead_*", "Group": "Halstead", "Description": "Volume, effort, difficulty"},
        {"Feature": "branch_count / path_count", "Group": "Control flow", "Description": "Structural graph metrics"},
        {"Feature": "code_churn", "Group": "Process", "Description": "Recent change volume"},
        {"Feature": "review_coverage / test_coverage", "Group": "Process", "Description": "Quality process signals"},
        {"Feature": "team_experience", "Group": "Process", "Description": "Team familiarity score"},
        {"Feature": "language / module_type", "Group": "Context", "Description": "Technology and role"},
        {"Feature": "defective", "Group": "Target", "Description": "Binary defect label"},
        {"Feature": "severity / defect_count", "Group": "Target", "Description": "Severity and count when defective"},
    ])
    st.dataframe(docs, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "software_metrics.csv",
        "text/csv",
    )

# ---- API ----
elif page == "API":
    st.title("API")
    st.markdown("Score modules over HTTP.")
    st.code("POST /predict")
    st.json({
        "loc": 800, "cyclomatic_complexity": 18, "code_churn": 40,
        "review_coverage": 0.4, "test_coverage": 0.35, "team_experience": 0.4,
        "language": "Java", "module_type": "Core",
    })
    st.markdown("Start the server:")
    st.code("uvicorn api:app --host 0.0.0.0 --port 8000 --reload")
    st.info("Interactive docs: http://localhost:8000/docs")

st.markdown("---")
st.caption("BugScope · Software defect prediction · Demo dataset")
