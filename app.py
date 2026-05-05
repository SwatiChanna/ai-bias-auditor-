"""Streamlit dashboard for AI Bias & Fairness Auditor (India-Aware).

This module provides an interactive web interface for auditing AI models for bias
across gender, caste, language, and region dimensions. It integrates fairness
metrics, counterfactual testing, explanations, and mitigation recommendations.
"""

from __future__ import annotations

from fpdf import FPDF
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state

from src import bias_engine
from src.counterfactual_templates import CounterfactualGenerator


def _get_traffic_light_color(score: float, metric: str) -> str:
    """Return traffic light color based on fairness score thresholds.

    Args:
        score: The fairness metric value.
        metric: Either 'disparate_impact' or 'statistical_parity'.

    Returns:
        Color string: 'red', 'yellow', or 'green'.
    """
    if metric == "disparate_impact":
        if score < 0.8 or score > 1.25:
            return "red"
        elif 0.8 <= score <= 1.25:
            return "green"
        else:
            return "yellow"
    elif metric == "statistical_parity":
        if abs(score) > 0.1:
            return "red"
        else:
            return "green"
    return "gray"


def _create_fairness_chart(scores: Dict[str, float]) -> Any:
    """Create a Plotly bar chart for fairness scores.

    Args:
        scores: Dictionary with 'disparate_impact' and 'statistical_parity' keys.

    Returns:
        Plotly figure object.
    """
    df = pd.DataFrame({
        "Metric": ["Disparate Impact", "Statistical Parity Difference"],
        "Score": [scores["disparate_impact"], scores["statistical_parity"]],
        "Color": [
            _get_traffic_light_color(scores["disparate_impact"], "disparate_impact"),
            _get_traffic_light_color(scores["statistical_parity"], "statistical_parity"),
        ]
    })
    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Color",
        color_discrete_map={"red": "red", "yellow": "yellow", "green": "green"},
        title="Fairness Scores",
    )
    return fig


def _run_counterfactual(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Apply basic counterfactual transformation for the given dimension.

    Args:
        df: Original dataset.
        dimension: One of 'gender', 'caste', 'language', 'region'.

    Returns:
        Modified dataset with swapped protected attributes.
    """
    df_copy = df.copy()
    if dimension == "gender":
        # Swap gender: assume 0->1, 1->0
        if "gender" in df_copy.columns:
            df_copy["gender"] = 1 - df_copy["gender"]
    elif dimension == "caste":
        # Swap caste_binary: assume 0->1, 1->0
        if "caste_binary" in df_copy.columns:
            df_copy["caste_binary"] = 1 - df_copy["caste_binary"]
    elif dimension == "language":
        # Swap language_binary
        if "language_binary" in df_copy.columns:
            df_copy["language_binary"] = 1 - df_copy["language_binary"]
    elif dimension == "region":
        # Swap region_binary
        if "region_binary" in df_copy.columns:
            df_copy["region_binary"] = 1 - df_copy["region_binary"]
    return df_copy


def _generate_explanation(scores: Dict[str, float], dataset_type: str) -> str:
    """Generate plain English explanation of bias findings.

    Args:
        scores: Fairness scores.
        dataset_type: The demo dataset type.

    Returns:
        Short explanation string.
    """
    di = scores["disparate_impact"]
    spd = scores["statistical_parity"]
    if di < 0.8 or abs(spd) > 0.1:
        return f"⚠️ Potential bias detected in {dataset_type} data. Disparate impact is {di:.2f}, indicating unequal outcomes. Statistical parity difference is {spd:.2f}, showing group disparities."
    else:
        return f"✅ No significant bias found in {dataset_type} data. Scores are within acceptable ranges."


def _generate_recommendations() -> List[str]:
    """Return a list of mitigation recommendations.

    Returns:
        List of three recommendation strings.
    """
    return [
        "Reweigh training data using AIF360 to balance protected groups.",
        "Apply fairness constraints during model training with Fairlearn.",
        "Post-process predictions to enforce equalized odds across groups.",
    ]


def _create_report(scores: Dict[str, float], explanation: str, recommendations: List[str]) -> bytes:
    """Create a PDF report for download.

    Args:
        scores: Fairness scores.
        explanation: Bias explanation.
        recommendations: List of recommendations.

    Returns:
        PDF content as bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Bias & Fairness Auditor Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Disparate Impact: {scores['disparate_impact']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Statistical Parity Difference: {scores['statistical_parity']:.2f}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Explanation: {explanation}")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Recommendations:", ln=True)
    for rec in recommendations:
        pdf.cell(200, 10, txt=f"- {rec}", ln=True)
    return pdf.output(dest="S").encode("latin1")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="AI Bias & Fairness Auditor",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark mode CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title
    st.title("🔍 AI Bias & Fairness Auditor (India-Aware)")

    # Initialize session state
    if "dataset" not in state:
        state.dataset = None
    if "scores" not in state:
        state.scores = {}
    if "dataset_type" not in state:
        state.dataset_type = ""

    # One-Click Demo Section
    st.header("🎯 One-Click Demo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Load Gender Bias Demo (Adult Income)"):
            state.dataset = bias_engine.load_demo_dataset("gender")
            state.dataset_type = "gender"
            state.protected_attr = "gender"
            state.label = "income"
    with col2:
        if st.button("Load Caste Bias Demo (IndiCASA)"):
            state.dataset = bias_engine.load_demo_dataset("caste")
            state.dataset_type = "caste"
            state.protected_attr = "caste_binary"
            state.label = "income"
    with col3:
        if st.button("Load Language Bias Demo"):
            state.dataset = bias_engine.load_demo_dataset("language")
            state.dataset_type = "language"
            state.protected_attr = "language_binary"
            state.label = "admit"
    with col4:
        if st.button("Load Regional Bias Demo"):
            state.dataset = bias_engine.load_demo_dataset("region")
            state.dataset_type = "region"
            state.protected_attr = "region_binary"
            state.label = "loan_approved"

    if state.dataset is not None:
        # Compute fairness scores
        di = bias_engine.calculate_disparate_impact(state.dataset, state.protected_attr, state.label)
        spd = bias_engine.calculate_statistical_parity_difference(state.dataset, state.protected_attr, state.label)
        state.scores = {"disparate_impact": di, "statistical_parity": spd}

        # Fairness Scores Section
        st.header("📊 Fairness Scores")
        col1, col2 = st.columns(2)
        with col1:
            color = _get_traffic_light_color(di, "disparate_impact")
            st.metric("Disparate Impact", f"{di:.2f}", delta_color=color)
        with col2:
            color = _get_traffic_light_color(spd, "statistical_parity")
            st.metric("Statistical Parity Difference", f"{spd:.2f}", delta_color=color)
        st.plotly_chart(_create_fairness_chart(state.scores), use_container_width=True)

        # Counterfactual Testing Section
        st.header("🔄 Counterfactual Testing")
        dimension = st.selectbox("Vary Dimension:", ["Gender", "Caste", "Language", "Region"])
        if st.button("Run Counterfactual Test"):
            generator = CounterfactualGenerator()
            if dimension == "Gender":
                cf_df = generator.generate_gender_counterfactual(state.dataset)
            elif dimension == "Caste":
                cf_df = generator.generate_caste_counterfactual(state.dataset)
            elif dimension == "Language":
                cf_df = generator.generate_language_counterfactual(state.dataset)
            else:
                cf_df = generator.generate_region_counterfactual(state.dataset)

            cf_preds = bias_engine.run_model(cf_df, state.label)
            cf_di = bias_engine.calculate_disparate_impact(cf_df, state.protected_attr, state.label)
            st.subheader("Before vs After")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Scores")
                st.metric("DI", f"{di:.2f}")
                st.metric("SPD", f"{spd:.2f}")
            with col2:
                st.write("Counterfactual Scores")
                st.metric("DI", f"{cf_di:.2f}", delta=f"{(cf_di - di):.2f}")
                if abs(cf_di - di) > 0.1:
                    st.warning("⚠️ High Sensitivity to Demographic Features")
            st.write("Counterfactual predictions generated", len(cf_preds), "rows")

        # Bias Explanation Section
        st.header("💡 Bias Explanation")
        explanation = _generate_explanation(state.scores, state.dataset_type)
        st.write(explanation)

        # Mitigation Recommendations Section
        st.header("🛠️ Mitigation Recommendations")
        recommendations = _generate_recommendations()
        for rec in recommendations:
            st.write(f"- {rec}")

        # Export Section
        st.header("📄 Export Report")
        report = _create_report(state.scores, explanation, recommendations)
        st.download_button(
            label="📄 Download PDF Report",
            data=_create_report(state.scores, explanation, recommendations),
            file_name="bias_audit_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
