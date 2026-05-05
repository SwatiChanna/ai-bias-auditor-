import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src import explanations


def test_generate_shap_explanation_detects_sensitive_bias():
    df = pd.DataFrame(
        {
            "gender": [0, 1, 0, 1],
            "age": [22, 45, 23, 39],
            "income": [0, 1, 0, 1],
        }
    )
    X = df[["gender", "age"]]
    y = df["income"]
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X, y)

    feature_importances, is_bias_source = explanations.generate_shap_explanation(
        model, X, "gender"
    )

    assert isinstance(feature_importances, dict)
    assert any("gender" in key.lower() for key in feature_importances)
    assert is_bias_source is True


def test_generate_plain_english_explanation_disparate_impact():
    explanation = explanations.generate_plain_english_explanation(
        "disparate_impact", 0.75, 0.8, "gender"
    )
    assert "75% approval" in explanation
    assert "discriminatory" in explanation
    assert "80%" in explanation


def test_generate_plain_english_explanation_statistical_parity():
    explanation = explanations.generate_plain_english_explanation(
        "statistical_parity", 0.15, 0.1, "caste"
    )
    assert "15.0%" in explanation
    assert "caste" in explanation
    assert "threshold: 10.0%" in explanation


def test_identify_bias_source_prioritizes_sensitive_attributes():
    shap_values = np.array(
        [[0.02, 0.30, 0.10, 0.05], [0.01, 0.25, 0.12, 0.08]]
    )
    feature_names = ["age", "gender", "income", "education"]
    top_features = explanations.identify_bias_source(shap_values, feature_names)

    assert top_features[0] == "gender"
    assert len(top_features) == 3
    assert "gender" in top_features
