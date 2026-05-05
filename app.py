"""Streamlit dashboard for AI Bias & Fairness Auditor (India-Aware).

This module provides an interactive web interface for auditing AI models for bias
across gender, caste, language, and region dimensions. It integrates fairness
metrics, counterfactual testing, explanations, and mitigation recommendations.
"""

from __future__ import annotations

from fpdf import FPDF
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit import session_state as state
from typing import Any, Dict, List
import io

from src import bias_engine
from src import mitigation
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


def _create_fairness_dimensions_chart(dimensions_data: Dict[str, float]) -> Any:
    """Create a multi-dimensional fairness chart across bias types.

    Args:
        dimensions_data: Dict mapping dimension names to disparate impact scores.

    Returns:
        Plotly figure with threshold line at 0.8.
    """
    df = pd.DataFrame({
        "Dimension": list(dimensions_data.keys()),
        "Disparate Impact": list(dimensions_data.values()),
    })
    df["Status"] = df["Disparate Impact"].apply(
        lambda x: "Fair (≥0.8)" if x >= 0.8 else "Biased (<0.8)"
    )
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Dimension"],
        y=df["Disparate Impact"],
        marker=dict(
            color=df["Status"].map({"Fair (≥0.8)": "green", "Biased (<0.8)": "red"}),
        ),
        text=df["Disparate Impact"].round(2),
        textposition="auto",
        name="Disparate Impact"
    ))
    
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="orange",
        annotation_text="Bias Threshold (0.8)",
        annotation_position="right",
    )
    
    fig.update_layout(
        title="Fairness Scores Across Dimensions",
        xaxis_title="Bias Dimension",
        yaxis_title="Disparate Impact Score",
        hovermode="x unified",
        template="plotly_dark",
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


def _sanitize_pdf_text(text: str) -> str:
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def _create_report(scores: Dict[str, float], explanation: str, recommendations: List[str]) -> bytes:
    """Create a comprehensive PDF report for download.

    Args:
        scores: Fairness scores.
        explanation: Bias explanation.
        recommendations: List of recommendations.

    Returns:
        PDF content as bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", size=16)
    pdf.cell(200, 15, txt=_sanitize_pdf_text("AI Bias Audit Report"), ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt=_sanitize_pdf_text("Generated with AI Bias Auditor - 24hr Hackathon Entry"), ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(200, 10, txt=_sanitize_pdf_text("Executive Summary"), ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt=_sanitize_pdf_text(explanation))
    
    pdf.ln(8)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(200, 10, txt=_sanitize_pdf_text("Fairness Metrics"), ln=True)
    pdf.set_font("Arial", size=10)
    di_text = f"Disparate Impact: {scores['disparate_impact']:.3f}"
    spd_text = f"Statistical Parity Difference: {scores['statistical_parity']:.3f}"
    pdf.multi_cell(0, 6, txt=_sanitize_pdf_text(f"{di_text}\n{spd_text}"))
    
    pdf.ln(8)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(200, 10, txt=_sanitize_pdf_text("Mitigation Recommendations"), ln=True)
    pdf.set_font("Arial", size=10)
    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 6, txt=_sanitize_pdf_text(f"{i}. {rec}"))
    
    pdf.ln(10)
    pdf.set_font("Arial", "I", size=8)
    pdf.cell(200, 5, txt=_sanitize_pdf_text("This report audits AI fairness across protected dimensions."), ln=True)
    
    return pdf.output(dest="S").encode("latin-1", errors="replace")


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

    # Initialize session state for caching
    if "dataset" not in state:
        state.dataset = None
    if "scores" not in state:
        state.scores = {}
    if "dataset_type" not in state:
        state.dataset_type = ""
    if "cached_results" not in state:
        state.cached_results = {}

    # Sidebar Navigation
    st.sidebar.title("🗂️ Navigation")
    nav_option = st.sidebar.radio(
        "Select Section:",
        ["🏠 Home", "📊 Fairness Dashboard", "🔄 Counterfactual Testing", "🛠️ Mitigation Tools"],
        index=0,
        key="nav_radio",
    )

    # Home Section
    if nav_option == "🏠 Home":
        st.title("🔍 AI Bias & Fairness Auditor (India-Aware)")
        st.markdown("""
        Welcome to the **AI Bias & Fairness Auditor**, an interactive tool for detecting and mitigating 
        bias in AI systems across gender, caste, language, and region dimensions.
        
        ### Key Features:
        - 📊 **Real-time Fairness Metrics**: Disparate impact and statistical parity calculations
        - 🔄 **Counterfactual Testing**: Swap protected attributes to detect sensitivity
        - 🛠️ **Bias Mitigation**: AIF360 reweighing and Fairlearn threshold optimization
        - 💡 **Explainability**: SHAP-based feature importance explanations
        - 📄 **PDF Export**: Comprehensive bias audit reports
        
        ### Supported Bias Dimensions:
        - ⚖️ **Gender**: Income prediction disparities
        - 👥 **Caste**: IndiCASA loan approval bias (India-specific)
        - 🗣️ **Language**: Selection bias in Hinglish/multilingual data
        - 📍 **Region**: Tier-2/3 city hiring disparities
        """)

        st.sidebar.header("📁 Select Demo Dataset")
        demo_options = {
            "None": None,
            "🏦 Indian Loan Audit (Caste)": {
                "path": "data/demo_indicasa.csv",
                "dataset_type": "caste",
                "protected_attr": "caste",
                "label": "loan_approved",
            },
            "📍 Regional Hiring (Mumbai/Bangalore/Rural Bihar)": {
                "path": "data/demo_regional.csv",
                "dataset_type": "region",
                "protected_attr": "location",
                "label": "hired",
            },
            "🗣️ Language Bias (English/Hindi/Tamil/Bengali)": {
                "path": "data/demo_language.csv",
                "dataset_type": "language",
                "protected_attr": "language",
                "label": "selected",
            },
            "⚖️ Gender Income Gap (Adult Dataset)": {
                "path": "data/demo_adult.csv",
                "dataset_type": "gender",
                "protected_attr": "gender",
                "label": "income_50k",
            },
        }

        selection = st.sidebar.selectbox("Choose Demo:", list(demo_options.keys()))
        if selection != "None":
            try:
                with st.spinner(f"Loading {selection}..."):
                    metadata = demo_options[selection]
                    df = pd.read_csv(metadata["path"])
                    state.dataset = df
                    state.dataset_type = metadata["dataset_type"]
                    state.protected_attr = metadata["protected_attr"]
                    state.label = metadata["label"]
                    st.sidebar.success(f"✅ Loaded: {selection}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading dataset: {e}")
                state.dataset = None

    # Fairness Dashboard Section
    elif nav_option == "📊 Fairness Dashboard":
        st.title("📊 Fairness Dashboard")
        
        if state.dataset is None:
            st.warning("⚠️ Please load a dataset from the Home section first.")
            return

        try:
            with st.spinner("Computing fairness metrics..."):
                di = bias_engine.calculate_disparate_impact(state.dataset, state.protected_attr, state.label)
                spd = bias_engine.calculate_statistical_parity_difference(state.dataset, state.protected_attr, state.label)
                state.scores = {"disparate_impact": di, "statistical_parity": spd}

            st.header("Fairness Scores")
            col1, col2, col3 = st.columns(3)
            with col1:
                color = _get_traffic_light_color(di, "disparate_impact")
                st.metric("Disparate Impact", f"{di:.3f}", 
                         help="Values 0.8-1.25 indicate fairness. <0.8 or >1.25 indicates bias.")
            with col2:
                color = _get_traffic_light_color(spd, "statistical_parity")
                st.metric("Statistical Parity Diff", f"{spd:.3f}",
                         help="Values close to 0 indicate fairness. >0.1 indicates bias.")
            with col3:
                bias_level = "High Bias" if (di < 0.8 or abs(spd) > 0.1) else "Fair"
                st.metric("Bias Level", bias_level)

            st.plotly_chart(_create_fairness_chart(state.scores), use_container_width=True)

            # Mitigation Option
            st.header("🧩 Fairness Mitigation")
            if st.checkbox("Apply AIF360 Reweighing Mitigation"):
                try:
                    with st.spinner("Applying reweighing..."):
                        mitigated_df = mitigation.apply_reweighing(state.dataset, state.protected_attr, state.label)
                        m_di = bias_engine.calculate_disparate_impact(mitigated_df, state.protected_attr, state.label)
                        m_spd = bias_engine.calculate_statistical_parity_difference(mitigated_df, state.protected_attr, state.label)
                        
                        orig_col, mitigated_col = st.columns(2)
                        with orig_col:
                            st.subheader("Before Mitigation")
                            st.metric("Disparate Impact", f"{di:.3f}")
                            st.metric("Stat. Parity Diff", f"{spd:.3f}")
                        with mitigated_col:
                            st.subheader("After Reweighing")
                            st.metric("Disparate Impact", f"{m_di:.3f}", 
                                     delta=f"{(m_di - di):+.3f}", delta_color="inverse")
                            st.metric("Stat. Parity Diff", f"{m_spd:.3f}",
                                     delta=f"{(m_spd - spd):+.3f}", delta_color="inverse")
                except Exception as e:
                    st.error(f"❌ Mitigation failed: {e}")

            # Explanation
            st.header("💡 Bias Analysis")
            explanation = _generate_explanation(state.scores, state.dataset_type)
            st.info(explanation)

            # Recommendations
            st.header("🎯 Mitigation Recommendations")
            recommendations = _generate_recommendations()
            for i, rec in enumerate(recommendations, 1):
                st.write(f"**{i}.** {rec}")

            india_rec = mitigation.get_india_specific_recommendations(state.dataset_type)
            st.success(f"🇮🇳 India-Specific: {india_rec}")

            # Export
            st.header("📄 Export Report")
            col1, col2 = st.columns(2)
            with col1:
                report = _create_report(state.scores, explanation, recommendations)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=report,
                    file_name="bias_audit_report.pdf",
                    mime="application/pdf",
                )
            with col2:
                csv = state.dataset.to_csv(index=False)
                st.download_button(
                    label="📊 Download Dataset (CSV)",
                    data=csv,
                    file_name="dataset.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"❌ Error computing fairness metrics: {e}")

    # Counterfactual Testing Section
    elif nav_option == "🔄 Counterfactual Testing":
        st.title("🔄 Counterfactual Testing")
        
        if state.dataset is None:
            st.warning("⚠️ Please load a dataset from the Home section first.")
            return

        st.markdown("""
        Counterfactual testing swaps protected attributes to detect if the model's 
        decisions are overly sensitive to demographic features.
        """)

        dimension = st.selectbox(
            "Select Dimension to Swap:",
            ["Gender", "Caste", "Language", "Region"]
        )

        if st.button("🔄 Run Counterfactual Test"):
            try:
                with st.spinner("Generating counterfactual dataset..."):
                    generator = CounterfactualGenerator()
                    if dimension == "Gender":
                        cf_df = generator.generate_gender_counterfactual(state.dataset)
                    elif dimension == "Caste":
                        cf_df = generator.generate_caste_counterfactual(state.dataset)
                    elif dimension == "Language":
                        cf_df = generator.generate_language_counterfactual(state.dataset)
                    else:
                        cf_df = generator.generate_region_counterfactual(state.dataset)

                    cf_di = bias_engine.calculate_disparate_impact(cf_df, state.protected_attr, state.label)
                    cf_spd = bias_engine.calculate_statistical_parity_difference(cf_df, state.protected_attr, state.label)

                    orig_col, cf_col = st.columns(2)
                    with orig_col:
                        st.subheader("Original Data")
                        st.metric("Disparate Impact", f"{state.scores['disparate_impact']:.3f}")
                        st.metric("Stat. Parity Diff", f"{state.scores['statistical_parity']:.3f}")
                    with cf_col:
                        st.subheader(f"After {dimension} Swap")
                        st.metric("Disparate Impact", f"{cf_di:.3f}",
                                 delta=f"{(cf_di - state.scores['disparate_impact']):+.3f}")
                        st.metric("Stat. Parity Diff", f"{cf_spd:.3f}",
                                 delta=f"{(cf_spd - state.scores['statistical_parity']):+.3f}")

                    sensitivity = abs(cf_di - state.scores['disparate_impact'])
                    if sensitivity > 0.1:
                        st.warning(f"⚠️ High Sensitivity: Score changed by {sensitivity:.3f}")
                    else:
                        st.success(f"✅ Low Sensitivity: Score change is {sensitivity:.3f}")

                    st.subheader("Counterfactual Sample (First 5 rows)")
                    st.dataframe(cf_df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Counterfactual test failed: {e}")

    # Mitigation Tools Section
    elif nav_option == "🛠️ Mitigation Tools":
        st.title("🛠️ Bias Mitigation Strategies")
        
        if state.dataset is None:
            st.warning("⚠️ Please load a dataset from the Home section first.")
            return

        st.markdown("""
        This section provides mitigation strategies tailored to detected biases.
        """)

        st.header("AIF360 Reweighing")
        st.markdown("""
        **How it works**: Reweighing adjusts sample weights to equalize outcome 
        rates across protected groups before model training.
        """)
        
        if st.button("Apply Reweighing"):
            try:
                with st.spinner("Computing reweighing..."):
                    reweighed_df = mitigation.apply_reweighing(state.dataset, state.protected_attr, state.label)
                    st.success("✅ Reweighing applied successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Weight Statistics (Original)**")
                        st.metric("Mean", "1.0")
                        st.metric("Std Dev", "0.0")
                    with col2:
                        st.write("**Weight Statistics (Reweighted)**")
                        weights = reweighed_df['instance_weights']
                        st.metric("Mean", f"{weights.mean():.3f}")
                        st.metric("Std Dev", f"{weights.std():.3f}")
                    
                    st.download_button(
                        label="📥 Download Reweighted Dataset",
                        data=reweighed_df.to_csv(index=False),
                        file_name="reweighted_dataset.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"❌ Reweighing failed: {e}")

        st.header("India-Specific Recommendations")
        for dimension in ["gender", "caste", "language", "region"]:
            rec = mitigation.get_india_specific_recommendations(dimension)
            st.info(f"**{dimension.title()}**: {rec}")


if __name__ == "__main__":
    main()
