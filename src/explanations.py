"""SHAP explanations for AI Bias & Fairness Auditor.

This module provides functions to generate feature importance explanations
and plain-English bias summaries, with special handling for sensitive
attributes in the Indian fairness context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


def _build_explainer(model: BaseEstimator, data: pd.DataFrame) -> Any:
    """Return a SHAP explainer appropriate for the model type."""
    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model, data)
    if hasattr(model, "predict_proba"):
        return shap.KernelExplainer(model.predict_proba, data.iloc[: min(len(data), 100)])
    return shap.KernelExplainer(model.predict, data.iloc[: min(len(data), 100)])


def _mean_abs_shap_values(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output into mean absolute values for each feature."""
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    array = np.asarray(shap_values, dtype=float)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2:
        return np.abs(array).mean(axis=0)
    if array.ndim == 1:
        return np.abs(array)
    raise ValueError("Unexpected SHAP values shape")


def generate_shap_explanation(
    model: BaseEstimator, input_data: pd.DataFrame, sensitive_attribute: str
) -> Tuple[Dict[str, float], bool]:
    """Generate SHAP importance values and detect sensitive-feature bias.

    Args:
        model: A trained scikit-learn estimator.
        input_data: Feature data used for SHAP explanation.
        sensitive_attribute: Name of the sensitive attribute to inspect.

    Returns:
        A tuple with a dict of feature importance values and a boolean flag.
    """
    x = pd.get_dummies(input_data.copy(), drop_first=True)
    explainer = _build_explainer(model, x)
    shap_values = explainer.shap_values(x)
    importance_array = _mean_abs_shap_values(shap_values)

    feature_importances = dict(zip(x.columns.tolist(), importance_array.tolist()))

    sensitive_importance = 0.0
    for feature_name, importance in feature_importances.items():
        if sensitive_attribute.lower() in feature_name.lower():
            sensitive_importance = max(sensitive_importance, importance)

    is_bias_source = sensitive_importance > 0.1
    return feature_importances, is_bias_source


def generate_plain_english_explanation(
    metric_name: str, score: float, threshold: float, bias_source: str
) -> str:
    """Generate a Markdown explanation for a fairness metric.

    Args:
        metric_name: 'disparate_impact' or 'statistical_parity'.
        score: The metric value.
        threshold: Threshold used for bias detection.
        bias_source: The sensitive feature causing bias.

    Returns:
        Formatted explanation string.
    """
    if metric_name == "disparate_impact":
        return (
            f"**Disparate Impact**: Model gives {score * 100:.0f}% approval to privileged group "
            f"vs 100% to unprivileged. This is discriminatory (threshold: {threshold * 100:.0f}%)."
        )
    if metric_name == "statistical_parity":
        return (
            f"**Statistical Parity**: Difference in approval rates is {score * 100:.1f}% "
            f"(threshold: {threshold * 100:.1f}%). Sensitive attribute '{bias_source}' is driving bias."
        )
    return f"**{metric_name}**: Score {score:.2f} (threshold: {threshold:.2f}). Sensitive attribute '{bias_source}'."


def identify_bias_source(shap_values: np.ndarray, feature_names: List[str]) -> List[str]:
    """Identify the top 3 features contributing to bias from SHAP values.

    Sensitive attributes are prioritized in the ranking.
    """
    importances = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(feature_names, importances), key=lambda item: item[1], reverse=True
    )

    sensitive_keywords = ["gender", "caste", "language", "region", "religion"]
    priority: List[Tuple[str, float]] = []
    non_priority: List[Tuple[str, float]] = []

    for feature_name, importance in ranked:
        if any(keyword in feature_name.lower() for keyword in sensitive_keywords):
            priority.append((feature_name, importance))
        else:
            non_priority.append((feature_name, importance))

    top_features = [feature for feature, _ in priority[:3]]
    if len(top_features) < 3:
        top_features.extend([feature for feature, _ in non_priority[: 3 - len(top_features)]])

    return top_features[:3]
