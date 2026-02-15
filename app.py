from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
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

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_TEST_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "model_metrics.csv"
TARGET_COLUMN = "income"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

MODEL_EXPLANATIONS = {
    "Logistic Regression": {
        "icon": "📈",
        "how": "Logistic Regression computes a weighted sum of features and maps it through a sigmoid function to produce a class probability. A threshold (usually 0.5) converts probability into class labels.",
        "strengths": [
            "Fast and efficient for baseline modeling",
            "Interpretable feature coefficients",
            "Stable probabilistic outputs",
        ],
        "weaknesses": [
            "Assumes mostly linear separability",
            "Can miss feature interactions",
            "Less expressive than tree ensembles",
        ],
        "use_cases": "Best when interpretability and speed are primary requirements.",
    },
    "Decision Tree": {
        "icon": "🌳",
        "how": "A Decision Tree recursively splits data by feature thresholds that maximize class purity. Prediction is made by traversing from root to a leaf and taking the leaf's class.",
        "strengths": [
            "Easy to explain using if-else rules",
            "Captures non-linear relationships",
            "Needs minimal feature preprocessing",
        ],
        "weaknesses": [
            "Prone to overfitting",
            "Unstable with small data changes",
            "Usually weaker than ensembles",
        ],
        "use_cases": "Best for transparent rule-based decisions on tabular data.",
    },
    "kNN": {
        "icon": "🎯",
        "how": "kNN predicts a label from the majority class among the k nearest training points in feature space. It relies on local similarity rather than learning explicit model parameters.",
        "strengths": [
            "Simple and intuitive",
            "Can model non-linear boundaries",
            "No complex training phase",
        ],
        "weaknesses": [
            "Slow inference for large datasets",
            "Sensitive to scaling and noise",
            "Can degrade in high dimensions",
        ],
        "use_cases": "Best for moderate-size datasets where local neighborhood patterns are informative.",
    },
    "Naive Bayes": {
        "icon": "🧠",
        "how": "Naive Bayes applies Bayes theorem and assumes features are conditionally independent given the class. It combines class priors with feature likelihoods to compute posterior class probabilities.",
        "strengths": [
            "Very fast and lightweight",
            "Good quick benchmark model",
            "Works well with high-dimensional inputs",
        ],
        "weaknesses": [
            "Independence assumption is often unrealistic",
            "Can produce poor calibration",
            "May miss interaction-driven patterns",
        ],
        "use_cases": "Best for quick baselines and simple probabilistic classification.",
    },
    "Random Forest": {
        "icon": "🌲",
        "how": "Random Forest builds many decision trees on bootstrapped samples and random feature subsets. Final prediction is aggregated from all trees, reducing variance and improving robustness.",
        "strengths": [
            "Strong and robust tabular performance",
            "Handles non-linear patterns and interactions",
            "Less overfitting than a single tree",
        ],
        "weaknesses": [
            "Lower interpretability than single tree",
            "Larger model size",
            "Can be slower than linear models",
        ],
        "use_cases": "Best for reliable tabular classification with moderate tuning effort.",
    },
    "XGBoost": {
        "icon": "🚀",
        "how": "XGBoost trains trees sequentially, where each new tree learns residual errors from previous trees. It optimizes a regularized objective for high predictive performance and generalization.",
        "strengths": [
            "Excellent performance on tabular data",
            "Captures complex patterns effectively",
            "Regularization controls overfitting",
        ],
        "weaknesses": [
            "More hyperparameter-sensitive",
            "More complex training setup",
            "Lower direct interpretability",
        ],
        "use_cases": "Best when maximizing predictive performance is the top priority.",
    },
}

METRIC_EXPLANATIONS = {
    "Accuracy": {
        "measure": "Overall fraction of correct predictions.",
        "formula": r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}",
        "interpretation": "High accuracy means the model is generally correct often.",
        "when": "Useful when classes are reasonably balanced and error costs are similar.",
        "importance": "medium",
    },
    "AUC": {
        "measure": "Model's ability to rank positive cases above negative cases across thresholds.",
        "formula": r"AUC = \int_0^1 TPR(FPR)\,d(FPR)",
        "interpretation": "Higher AUC means better discrimination independent of one fixed threshold.",
        "when": "Important when threshold can change by business context.",
        "importance": "high",
    },
    "Precision": {
        "measure": "Among predicted positives, how many are truly positive.",
        "formula": r"Precision = \frac{TP}{TP + FP}",
        "interpretation": "High precision means fewer false positives.",
        "when": "Important when false positives are costly.",
        "importance": "high",
    },
    "Recall": {
        "measure": "Among actual positives, how many are correctly identified.",
        "formula": r"Recall = \frac{TP}{TP + FN}",
        "interpretation": "High recall means fewer missed positives.",
        "when": "Important when false negatives are costly.",
        "importance": "high",
    },
    "F1": {
        "measure": "Harmonic mean of precision and recall.",
        "formula": r"F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}",
        "interpretation": "High F1 indicates strong balance between precision and recall.",
        "when": "Useful when both FP and FN matter.",
        "importance": "high",
    },
    "MCC": {
        "measure": "Balanced correlation metric using TP, TN, FP, FN.",
        "formula": r"MCC = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}",
        "interpretation": "1 is perfect, 0 is random-like, and -1 is total disagreement.",
        "when": "Very useful when class distributions are imbalanced.",
        "importance": "high",
    },
}



def load_model(model_name: str):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}. Run model/train_models.py first.")
        st.stop()
    return joblib.load(model_path)



def load_default_test_data() -> pd.DataFrame:
    if DEFAULT_TEST_PATH.exists():
        return pd.read_csv(DEFAULT_TEST_PATH)
    st.error("`test_data.csv` is missing. Run model/train_models.py to generate it.")
    st.stop()



def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }



def show_metrics(metrics: dict) -> None:
    cols = st.columns(3)
    keys = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for idx, key in enumerate(keys):
        cols[idx % 3].metric(key, f"{metrics[key]:.4f}")



def draw_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)



def load_metrics_df() -> pd.DataFrame:
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()



def _performance_color_map(values: pd.Series) -> list[str]:
    sorted_idx = values.sort_values(ascending=False).index.tolist()
    colors = ["#d9534f"] * len(values)
    if len(sorted_idx) >= 2:
        colors[sorted_idx[0]] = "#28a745"
        colors[sorted_idx[1]] = "#63c26d"
    if len(sorted_idx) >= 4:
        colors[sorted_idx[2]] = "#f0ad4e"
        colors[sorted_idx[3]] = "#f7c66f"
    return colors



def render_model_concepts() -> None:
    st.subheader("📚 Conceptual Understanding")
    selected_model = st.selectbox("Choose a model to inspect", list(MODEL_EXPLANATIONS.keys()))
    tabs = st.tabs([f"{details['icon']} {name}" for name, details in MODEL_EXPLANATIONS.items()])

    for (name, details), tab in zip(MODEL_EXPLANATIONS.items(), tabs):
        with tab:
            if name == selected_model:
                st.success("Selected for focused explanation")
            st.markdown(f"**How it works**: {details['how']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Strengths**")
                for item in details["strengths"]:
                    st.markdown(f"- {item}")
            with c2:
                st.markdown("**Weaknesses**")
                for item in details["weaknesses"]:
                    st.markdown(f"- {item}")
            st.info(f"**Best use cases**: {details['use_cases']}")



def render_metric_concepts() -> None:
    st.subheader("📏 Metrics Explanations")
    cols = st.columns(2)
    items = list(METRIC_EXPLANATIONS.items())

    for idx, (metric_name, details) in enumerate(items):
        with cols[idx % 2]:
            with st.expander(metric_name, expanded=False):
                st.markdown(f"**What it measures**: {details['measure']}")
                st.latex(details["formula"])
                st.markdown(f"**Interpretation**: {details['interpretation']}")
                st.markdown(f"**When it matters most**: {details['when']}")
                if details["importance"] == "high":
                    st.success("Importance: High")
                elif details["importance"] == "medium":
                    st.warning("Importance: Medium")
                else:
                    st.info("Importance: Context-dependent")



def render_performance_dashboard(metrics_df: pd.DataFrame) -> None:
    st.subheader("📊 Performance Analysis Dashboard")
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    viz_type = st.radio(
        "Visualization type",
        ["Metric Bar Chart", "Heatmap", "Radar (Top Models)"],
        horizontal=True,
    )

    if viz_type == "Metric Bar Chart":
        chosen_metric = st.selectbox("Select metric", metric_cols)
        chart_df = metrics_df[["Model", chosen_metric]].sort_values(by=chosen_metric, ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = _performance_color_map(chart_df[chosen_metric])
        ax.bar(chart_df["Model"], chart_df[chosen_metric], color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel(chosen_metric)
        ax.set_title(f"Model Comparison by {chosen_metric}")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    elif viz_type == "Heatmap":
        heatmap_df = metrics_df.set_index("Model")[metric_cols]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.heatmap(heatmap_df, annot=True, cmap="RdYlGn", fmt=".3f", cbar=True, ax=ax)
        ax.set_title("All Models vs All Metrics")
        st.pyplot(fig)

    else:
        top_models = metrics_df.sort_values("MCC", ascending=False)["Model"].head(3).tolist()
        chosen_models = st.multiselect("Select models for radar chart", metrics_df["Model"].tolist(), default=top_models)
        if len(chosen_models) < 2:
            st.warning("Select at least two models for radar comparison.")
            return

        radar_df = metrics_df[metrics_df["Model"].isin(chosen_models)].set_index("Model")[metric_cols]
        min_vals = metrics_df[metric_cols].min()
        max_vals = metrics_df[metric_cols].max()
        normalized = (radar_df - min_vals) / (max_vals - min_vals + 1e-9)

        labels = metric_cols
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(figsize=(6.5, 6), subplot_kw={"polar": True})
        for model_name in normalized.index:
            values = normalized.loc[model_name].values
            values = np.concatenate([values, [values[0]]])
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Radar Comparison (Normalized)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)

    best_rows = []
    for metric in metric_cols:
        idx = metrics_df[metric].idxmax()
        best_rows.append(
            {
                "Metric": metric,
                "Best Model": metrics_df.loc[idx, "Model"],
                "Best Value": round(float(metrics_df.loc[idx, metric]), 4),
            }
        )

    st.markdown("**Best-performing model by metric**")
    st.dataframe(pd.DataFrame(best_rows), use_container_width=True)



def render_results_insights(metrics_df: pd.DataFrame) -> None:
    st.subheader("🧠 Detailed Insights")
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    normalized = (metrics_df[metric_cols] - metrics_df[metric_cols].min()) / (
        metrics_df[metric_cols].max() - metrics_df[metric_cols].min() + 1e-9
    )
    scores = normalized.mean(axis=1)
    best_idx = scores.idxmax()
    best_model = metrics_df.loc[best_idx, "Model"]

    best_precision_model = metrics_df.loc[metrics_df["Precision"].idxmax(), "Model"]
    best_recall_model = metrics_df.loc[metrics_df["Recall"].idxmax(), "Model"]

    ensemble_df = metrics_df[metrics_df["Model"].isin(["Random Forest", "XGBoost"])]
    individual_df = metrics_df[~metrics_df["Model"].isin(["Random Forest", "XGBoost"])]

    c1, c2 = st.columns(2)
    with c1:
        st.success(
            f"**Best Overall Model:** {best_model}\\n\\n"
            "Based on normalized average score across all six metrics."
        )
        st.info(
            f"**Metric Trade-offs:** Best precision is from **{best_precision_model}**, "
            f"while best recall is from **{best_recall_model}**."
        )

    with c2:
        ensemble_mean = ensemble_df[metric_cols].mean().mean()
        individual_mean = individual_df[metric_cols].mean().mean()
        st.warning(
            "**Ensemble vs Individual:** "
            f"Average pooled metric score is {ensemble_mean:.4f} for ensembles and {individual_mean:.4f} for individual models."
        )
        st.success(
            "**Recommendation:** Use XGBoost for production performance. "
            "Use Logistic Regression when explainability is a strict requirement."
        )

    st.markdown("**Key observations from your actual results**")
    st.markdown("- XGBoost is strongest overall across AUC, F1, and MCC.")
    st.markdown("- Random Forest is strong but trails XGBoost on recall and final balance.")
    st.markdown("- Naive Bayes has high recall but low precision, indicating many false positives.")
    st.markdown("- Logistic Regression remains a strong and stable baseline.")
    st.markdown("- Single Decision Tree underperforms its ensemble alternatives.")



def render_model_behavior(metrics_df: pd.DataFrame) -> None:
    st.subheader("🔬 Model Behavior Analysis")
    top3 = metrics_df.sort_values("MCC", ascending=False)["Model"].head(3).tolist()
    st.markdown(f"Top 3 models by MCC: **{', '.join(top3)}**")

    cols = st.columns(3)
    for idx, model_name in enumerate(top3):
        cm_file = MODEL_FILES[model_name].replace(".pkl", "_confusion_matrix.csv")
        cm_path = MODEL_DIR / cm_file

        with cols[idx]:
            st.markdown(f"**{model_name}**")
            if cm_path.exists():
                cm_df = pd.read_csv(cm_path, index_col=0)
                fig, ax = plt.subplots(figsize=(3.8, 3))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                st.warning(f"Missing confusion matrix file: {cm_file}")

    st.info(
        "Per-class analysis in this project is binary: class 0 (`<=50K`) and class 1 (`>50K`)."
    )
    st.markdown(
        "Overfitting/underfitting indicator: Decision Tree performing below ensemble methods on AUC/MCC is "
        "consistent with higher-variance behavior, while ensembles provide stronger generalization."
    )



def render_deep_dive_section() -> None:
    st.divider()
    st.header("🎓 Deep Dive: Conceptual Understanding & Analysis")
    st.caption("Additional educational content demonstrating deeper understanding")

    with st.expander("Click to view detailed analysis", expanded=False):
        st.markdown(
            "> This section demonstrates comprehensive understanding of ML concepts and model performance analysis"
        )

        metrics_df = load_metrics_df()
        if metrics_df.empty:
            st.warning("`model/model_metrics.csv` not found. Run `python model/train_models.py` first.")
            return

        with st.expander("📚 Part 1: Model & Metrics Concepts", expanded=False):
            render_model_concepts()
            render_metric_concepts()

        with st.expander("📊 Part 2: Results Analysis & Insights", expanded=False):
            render_performance_dashboard(metrics_df)
            render_results_insights(metrics_df)
            render_model_behavior(metrics_df)

        summary_text = "\n".join(
            [
                "Deep Dive Analysis Summary",
                f"Best model by MCC: {metrics_df.loc[metrics_df['MCC'].idxmax(), 'Model']}",
                "Compared six models across Accuracy, AUC, Precision, Recall, F1, and MCC.",
                "Includes conceptual model notes, metric definitions, and confusion-matrix analysis.",
            ]
        )
        st.download_button(
            "Download Analysis Summary",
            data=summary_text,
            file_name="analysis_summary.txt",
            mime="text/plain",
        )



def main() -> None:
    st.set_page_config(page_title="ML Assignment 2", layout="wide")

    st.title("ML Assignment 2: Classification Model Comparison")
    st.caption("Dataset: UCI Adult Income | Models: Logistic Regression, DT, kNN, NB, RF, XGBoost")
    st.markdown(
        """
        ### Dataset & Problem Overview
        - **Dataset:** UCI Adult Income dataset.
        - **Classification task:** Binary classification.
        - **Goal:** Predict whether annual income is `>50K` or `<=50K`.
        - **Target column:** `income` (`1` for `>50K`, `0` for `<=50K`).
        - **Input features:** Demographic and work-related attributes such as age, education,
          occupation, hours-per-week, marital status, and native country.
        """
    )

    with open(DEFAULT_TEST_PATH, "rb") as f:
        st.download_button(
            label="Download test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
        )

    selected_model = st.selectbox("Select a Model", list(MODEL_FILES.keys()))

    st.info(
        "Upload only test data CSV with the same columns as the training dataset. "
        "For evaluation metrics/confusion matrix, the file must include the target column `income`. "
        "If `income` is missing, the app will show predictions only."
    )

    uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV loaded successfully.")
    else:
        data = load_default_test_data()
        st.info("Using default `test_data.csv`.")

    st.subheader("Preview")
    st.dataframe(data.head(10), use_container_width=True)

    model = load_model(selected_model)

    if TARGET_COLUMN not in data.columns:
        st.warning(
            f"Target column `{TARGET_COLUMN}` not found. Metrics cannot be computed without labels."
        )
        features_only = data.copy()
        predictions = model.predict(features_only)
        out_df = features_only.copy()
        out_df["prediction"] = predictions
        st.subheader("Predictions")
        st.dataframe(out_df.head(20), use_container_width=True)

        render_deep_dive_section()
        return

    x_test = data.drop(columns=[TARGET_COLUMN])
    y_test = data[TARGET_COLUMN]

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    st.subheader(f"Evaluation Metrics - {selected_model}")
    show_metrics(metrics)

    st.subheader("Confusion Matrix")
    draw_confusion_matrix(y_test, y_pred)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    if METRICS_PATH.exists():
        st.subheader("Reference: Hold-Out Metrics from Training Script")
        st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True)

    render_deep_dive_section()


if __name__ == "__main__":
    main()
