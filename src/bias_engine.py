from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _binary_labels(series: pd.Series) -> pd.Series:
    """Convert binary labels to integers 0 or 1.

    This helper supports boolean, numeric, and string label encodings that are
    common in fairness datasets. The function is intentionally strict so that
    only meaningful binary outcomes are passed to the fairness metrics.
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series.dtype):
        unique_values = set(series.dropna().astype(int).unique())
        if not unique_values <= {0, 1}:
            raise ValueError("Label values must be binary 0/1.")
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "y": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "n": 0,
    }
    mapped = normalized.map(mapping)
    if mapped.isnull().any():
        raise ValueError(
            "Label values must be binary strings 'yes/no', 'true/false', or 0/1."
        )
    return mapped.astype(int)


def _validate_binary_protected_attr(df: pd.DataFrame, protected_attr: str) -> tuple[Any, Any]:
    """Validate the protected attribute is binary and return ordered values."""
    if protected_attr not in df.columns:
        raise ValueError(f"Protected attribute '{protected_attr}' is missing from data.")

    distinct_values = df[protected_attr].dropna().unique()
    if len(distinct_values) != 2:
        raise ValueError(
            "Protected attribute must have exactly two distinct groups for binary fairness metrics."
        )

    ordered_values = sorted(distinct_values, key=lambda value: str(value))
    return ordered_values[0], ordered_values[1]


def _positive_rate(df: pd.DataFrame, protected_attr: str, label: str, group_value: Any) -> float:
    """Calculate the positive outcome rate for a binary protected group."""
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' is missing from data.")

    group_df = df[df[protected_attr] == group_value]
    if group_df.empty:
        raise ValueError(
            f"No records found for protected group value '{group_value}'."
        )

    y = _binary_labels(group_df[label])
    return float(y.mean())


def calculate_disparate_impact(df: pd.DataFrame, protected_attr: str, label: str) -> float:
    """Return the disparate impact ratio for a binary protected attribute.

    The disparate impact ratio measures how often the protected group with value
    A=0 receives the favorable outcome compared with A=1.

    Formula: P(Y=1 | A=0) / P(Y=1 | A=1)

    In the Indian context, this ratio can highlight whether historically
    disadvantaged groups receive systematically worse outcomes. For caste
    discrimination analysis, a ratio below 0.8 or above 1.25 is commonly
    interpreted as a sign of bias.
    """
    a0_value, a1_value = _validate_binary_protected_attr(df, protected_attr)
    p_a0 = _positive_rate(df, protected_attr, label, a0_value)
    p_a1 = _positive_rate(df, protected_attr, label, a1_value)

    if p_a1 == 0.0:
        return float("inf")
    return float(p_a0 / p_a1)


def calculate_statistical_parity_difference(
    df: pd.DataFrame, protected_attr: str, label: str
) -> float:
    """Return the statistical parity difference for a binary protected attribute.

    Statistical parity difference compares the positive outcome rates directly.

    Formula: P(Y=1 | A=0) - P(Y=1 | A=1)

    In Indian bias auditing, large differences can indicate that one group is
    systematically advantaged relative to the other. Differences above 0.1 are
    often used as a practical warning threshold.
    """
    a0_value, a1_value = _validate_binary_protected_attr(df, protected_attr)
    p_a0 = _positive_rate(df, protected_attr, label, a0_value)
    p_a1 = _positive_rate(df, protected_attr, label, a1_value)
    return float(p_a0 - p_a1)


def _gender_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 34, 28, 41, 22, 36],
            "education": [
                "Bachelors",
                "HS-grad",
                "Masters",
                "Some-college",
                "Bachelors",
                "HS-grad",
            ],
            "gender": [0, 1, 0, 1, 0, 1],
            "income": [1, 0, 1, 0, 1, 0],
        }
    )


def _caste_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "education": [
                "Graduate",
                "High School",
                "Graduate",
                "High School",
                "Graduate",
                "High School",
            ],
            "caste_group": ["General", "SC", "OBC", "ST", "OBC", "General"],
            "caste_binary": [0, 1, 1, 1, 1, 0],
            "income": [1, 0, 1, 0, 1, 0],
        }
    )


def _language_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utterance": [
                "The applicant has strong qualifications.",
                "Agle hafte interview ke liye ready ho jao.",
                "Candidate has good English communication skills.",
                "Interview mein achha perform kiya.",
            ],
            "language": ["English", "Hindi", "English", "Hindi"],
            "language_binary": [0, 1, 0, 1],
            "admit": [1, 0, 1, 0],
        }
    )


def _region_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": [
                "Punjab",
                "Tamil Nadu",
                "Uttar Pradesh",
                "Karnataka",
                "Haryana",
                "Kerala",
            ],
            "region": ["North India", "South India", "North India", "South India", "North India", "South India"],
            "region_binary": [0, 1, 0, 1, 0, 1],
            "loan_approved": [1, 0, 1, 0, 1, 0],
        }
    )


def load_demo_dataset(dataset_type: str) -> pd.DataFrame:
    """Load a demo dataset for the requested fairness dimension.

    The function is intentionally robust: if the dataset file is empty or missing,
    it returns a synthetic dataset that preserves the intended protected groups.
    """
    dataset_type_key = dataset_type.strip().lower()
    dataset_map = {
        "gender": "demo_adult.csv",
        "caste": "demo_indicasa.csv",
        "language": "demo_regional.csv",
        "region": "demo_regional.csv",
    }

    if dataset_type_key not in dataset_map:
        raise ValueError(
            f"Unknown demo dataset type '{dataset_type}'. "
            "Valid options are gender, caste, language, region."
        )

    file_path = DATA_DIR / dataset_map[dataset_type_key]
    try:
        dataset = pd.read_csv(file_path)
        if not dataset.empty:
            return dataset
    except (pd.errors.EmptyDataError, FileNotFoundError):
        pass

    if dataset_type_key == "gender":
        return _gender_demo_dataset()
    if dataset_type_key == "caste":
        return _caste_demo_dataset()
    if dataset_type_key == "language":
        return _language_demo_dataset()
    return _region_demo_dataset()


def run_model(df: pd.DataFrame, target_column: str) -> np.ndarray:
    """Train a simple logistic regression and return binary model predictions.

    The function is intended for demo fairness analysis and supports simple
    categorical data through one-hot encoding of non-target features.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    if df.shape[0] == 0:
        raise ValueError("Input dataset must contain at least one row.")

    y = _binary_labels(df[target_column])
    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes for classification.")

    X = df.drop(columns=[target_column])
    X_encoded = pd.get_dummies(X, drop_first=True)
    if X_encoded.shape[1] == 0:
        raise ValueError("No usable features found for model training.")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y)
    return model.predict(X_encoded)
