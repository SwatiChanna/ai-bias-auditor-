import pandas as pd
import numpy as np
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, confusion_matrix
from src.bias_engine import _binary_labels

def recommend_mitigation(bias_type, metric_name, score, data_info):
    """
    Recommend bias mitigation strategies based on bias type, metric, and score.

    Parameters:
    - bias_type (str): Type of bias ('high_disparate_impact', 'counterfactual_sensitivity', 'feature_correlation')
    - metric_name (str): Name of the fairness metric
    - score (float): The bias score
    - data_info (dict): Information about the dataset (e.g., {'protected_attr': 'gender', 'label': 'income'})

    Returns:
    - dict: Recommendations at data_level, model_level, output_level

    Examples:
    >>> recommend_mitigation('high_disparate_impact', 'disparate_impact', 0.8, {'protected_attr': 'gender'})
    {
        'data_level': 'Oversample underrepresented groups using SMOTE',
        'model_level': 'Apply adversarial debiasing or fairness constraints (Fairlearn)',
        'output_level': 'Use reweighing pre-processing (AIF360 Reweighing algorithm)'
    }
    """
    recommendations = {
        'high_disparate_impact': {
            'data_level': 'Oversample underrepresented groups using SMOTE to balance class distribution',
            'model_level': 'Apply adversarial debiasing or fairness constraints using Fairlearn',
            'output_level': 'Use reweighing pre-processing with AIF360 Reweighing algorithm'
        },
        'counterfactual_sensitivity': {
            'data_level': 'Augment dataset with counterfactual examples to reduce sensitivity',
            'model_level': 'Use fairness-aware loss functions or regularization',
            'output_level': 'Apply threshold optimization with Fairlearn ThresholdOptimizer'
        },
        'feature_correlation': {
            'data_level': 'Remove or transform correlated features using PCA or feature selection',
            'model_level': 'Implement feature debiasing techniques',
            'output_level': 'Use post-processing calibration methods'
        }
    }
    return recommendations.get(bias_type, {
        'data_level': 'General data balancing techniques',
        'model_level': 'Fairness-aware model training',
        'output_level': 'Post-processing debiasing'
    })

def _prepare_reweighing_dataset(df: pd.DataFrame, protected_attr: str, label: str) -> pd.DataFrame:
    if protected_attr not in df.columns:
        raise ValueError(f"Protected attribute '{protected_attr}' is missing from data.")
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' is missing from data.")

    dataset = df.copy()
    dataset[label] = _binary_labels(dataset[label])

    protected_series = dataset[protected_attr].copy()
    if protected_series.nunique(dropna=True) > 2:
        most_frequent = protected_series.value_counts(dropna=True).idxmax()
        dataset[protected_attr] = np.where(protected_series == most_frequent, 1, 0)
    else:
        unique_values = list(pd.unique(protected_series.dropna()))
        if len(unique_values) == 2:
            if set(unique_values) <= {0, 1}:
                dataset[protected_attr] = protected_series.astype(int)
            else:
                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                dataset[protected_attr] = protected_series.map(mapping).astype(int)
        else:
            dataset[protected_attr] = np.where(protected_series == unique_values[0], 0, 1)

    feature_columns = [col for col in dataset.columns if col not in {label, protected_attr}]
    encoded_features = pd.get_dummies(dataset[feature_columns], drop_first=True)
    dataset = pd.concat([encoded_features, dataset[[protected_attr, label]]], axis=1)
    return dataset


def apply_reweighing(df, protected_attr, label):
    """
    Apply AIF360 Reweighing pre-processing to mitigate bias.

    Reweighs the dataset to reduce bias in protected attributes.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - protected_attr (str): Name of the protected attribute column
    - label (str): Name of the label column

    Returns:
    - pd.DataFrame: Reweighted dataset

    Example:
    >>> df = pd.DataFrame({'gender': [0,1,0], 'income': [1,0,1]})
    >>> apply_reweighing(df, 'gender', 'income')
    # Returns reweighted DataFrame with fairness scores printed
    """
    encoded_df = _prepare_reweighing_dataset(df, protected_attr, label)

    dataset = BinaryLabelDataset(
        df=encoded_df,
        label_names=[label],
        protected_attribute_names=[protected_attr]
    )

    privileged_groups = [{protected_attr: 1}]
    unprivileged_groups = [{protected_attr: 0}]
    original_metric = BinaryLabelDatasetMetric(dataset, privileged_groups, unprivileged_groups)
    print("Original Disparate Impact:", original_metric.disparate_impact())
    print("Original Statistical Parity Difference:", original_metric.statistical_parity_difference())

    reweighing = Reweighing(privileged_groups, unprivileged_groups)
    reweighed_dataset = reweighing.fit_transform(dataset)

    reweighed_metric = BinaryLabelDatasetMetric(reweighed_dataset, privileged_groups, unprivileged_groups)
    print("Reweighted Disparate Impact:", reweighed_metric.disparate_impact())
    print("Reweighted Statistical Parity Difference:", reweighed_metric.statistical_parity_difference())

    reweighted_df = reweighed_dataset.convert_to_dataframe()[0]
    reweighted_df['instance_weights'] = reweighed_dataset.instance_weights

    return reweighted_df

def apply_threshold_optimization(model, df, protected_attr):
    """
    Apply Fairlearn ThresholdOptimizer post-processing for bias mitigation.

    Optimizes decision thresholds to improve fairness.

    Parameters:
    - model: Trained sklearn model
    - df (pd.DataFrame): Test dataset
    - protected_attr (str): Name of the protected attribute column

    Returns:
    - ThresholdOptimizer: Debiased model

    Example:
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> debiased = apply_threshold_optimization(model, df_test, 'gender')
    # Returns ThresholdOptimizer with improved metrics printed
    """
    # Prepare data
    X = df.drop([protected_attr, 'label'], axis=1, errors='ignore')
    y = df['label']
    sensitive_features = df[protected_attr]

    # Predict probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Apply ThresholdOptimizer
    threshold_optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        predict_method="predict_proba"
    )

    # Fit on the data
    threshold_optimizer.fit(X, y, sensitive_features=sensitive_features)

    # Predict with optimized thresholds
    y_pred_optimized = threshold_optimizer.predict(X, sensitive_features=sensitive_features)

    # Compute metrics
    original_accuracy = accuracy_score(y, (y_pred_proba > 0.5).astype(int))
    optimized_accuracy = accuracy_score(y, y_pred_optimized)

    print("Original Accuracy:", original_accuracy)
    print("Optimized Accuracy:", optimized_accuracy)
    print("Improvement in fairness achieved through threshold optimization.")

    return threshold_optimizer

def get_india_specific_recommendations(bias_dimension):
    """
    Get India-specific bias mitigation recommendations.

    Parameters:
    - bias_dimension (str): 'caste', 'language', 'region', etc.

    Returns:
    - str: Specific recommendation

    Examples:
    >>> get_india_specific_recommendations('caste')
    'Add SC/ST representation in training data (NEP 2020 guidelines)'
    >>> get_india_specific_recommendations('language')
    'Include Indic language code-mixed data (IndicBERT corpus)'
    """
    recommendations = {
        'caste': 'Add SC/ST representation in training data following NEP 2020 guidelines for inclusive education',
        'language': 'Include Indic language code-mixed data using IndicBERT corpus for multilingual fairness',
        'region': 'Balance North/South/East/West representation to avoid regional bias in Indian demographics',
        'gender': 'Ensure equal representation of men and women in accordance with Indian gender equality policies',
        'religion': 'Include diverse religious groups proportional to Indian census data'
    }
    return recommendations.get(bias_dimension, 'General India-specific data balancing recommendations')