import pandas as pd
import pytest

from src import bias_engine


def test_calculate_disparate_impact_binary_values():
    df = pd.DataFrame(
        {"protected": [0, 0, 1, 1, 1, 0], "label": [1, 0, 1, 0, 0, 1]}
    )
    assert bias_engine.calculate_disparate_impact(df, "protected", "label") == 2.0


def test_calculate_statistical_parity_difference_binary_values():
    df = pd.DataFrame(
        {"protected": [0, 0, 1, 1, 1, 0], "label": [1, 0, 1, 0, 0, 1]}
    )
    assert pytest.approx(
        bias_engine.calculate_statistical_parity_difference(df, "protected", "label"), rel=1e-6
    ) == 1.0 / 3.0


def test_calculate_disparate_impact_with_string_protected_groups():
    df = pd.DataFrame(
        {"protected": ["A", "A", "B", "B"], "label": ["yes", "no", "yes", "no"]}
    )
    assert bias_engine.calculate_disparate_impact(df, "protected", "label") == 1.0


def test_calculate_statistical_parity_difference_with_string_labels():
    df = pd.DataFrame(
        {"protected": [0, 1, 0, 1], "label": ["yes", "no", "no", "yes"]}
    )
    assert bias_engine.calculate_statistical_parity_difference(df, "protected", "label") == 0.0


def test_calculate_disparate_impact_multiclass_protected_group():
    df = pd.DataFrame(
        {
            "protected": ["A", "A", "A", "B", "B", "C"],
            "label": [1, 0, 1, 0, 1, 1],
        }
    )
    # Most frequent group 'A' becomes privileged; others are grouped as unprivileged.
    assert bias_engine.calculate_disparate_impact(df, "protected", "label") == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(
    "dataset_type, expected_columns",
    [
        ("gender", ["gender", "income"]),
        ("caste", ["caste_group", "income"]),
        ("language", ["language", "admit"]),
        ("region", ["region", "loan_approved"]),
    ],
)
def test_load_demo_dataset_returns_expected_columns(dataset_type, expected_columns):
    df = bias_engine.load_demo_dataset(dataset_type)
    assert not df.empty
    for column in expected_columns:
        assert column in df.columns


def test_load_demo_dataset_invalid_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown demo dataset type"):
        bias_engine.load_demo_dataset("unknown")


def test_run_model_predicts_binary_labels():
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "a", "b"],
            "target": ["yes", "no", "yes", "no"],
        }
    )
    predictions = bias_engine.run_model(df, "target")
    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})


def test_run_model_missing_target_raises_value_error():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column 'target' is not present"):
        bias_engine.run_model(df, "target")


def test_load_demo_dataset_reads_existing_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "demo_adult.csv"
    pd.DataFrame({"gender": [0, 1], "income": [1, 0]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(bias_engine, "DATA_DIR", tmp_path)
    dataset = bias_engine.load_demo_dataset("gender")
    assert len(dataset) == 2
    assert list(dataset.columns) == ["gender", "income"]
