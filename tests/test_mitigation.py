import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from src.mitigation import (
    recommend_mitigation,
    apply_reweighing,
    apply_threshold_optimization,
    get_india_specific_recommendations,
)

class TestRecommendMitigation:
    def test_high_disparate_impact(self):
        result = recommend_mitigation('high_disparate_impact', 'disparate_impact', 0.8, {'protected_attr': 'gender'})
        assert 'data_level' in result
        assert 'Oversample' in result['data_level']
        assert 'model_level' in result
        assert 'adversarial' in result['model_level']
        assert 'output_level' in result
        assert 'reweighing' in result['output_level']

    def test_counterfactual_sensitivity(self):
        result = recommend_mitigation('counterfactual_sensitivity', 'sensitivity', 0.9, {})
        assert 'counterfactual' in result['data_level']
        assert 'threshold' in result['output_level']

    def test_unknown_bias_type(self):
        result = recommend_mitigation('unknown', 'metric', 0.5, {})
        assert 'General' in result['data_level']

class TestApplyReweighing:
    def test_reweighing_application(self):
        df = pd.DataFrame({
            'gender': [0, 1, 0, 1, 0],
            'feature': [1, 2, 3, 4, 5],
            'income': [1, 0, 1, 0, 1]
        })
        result = apply_reweighing(df, 'gender', 'income')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Check if weights are added
        assert 'instance_weights' in result.columns

    def test_reweighing_with_categorical_and_multiclass_protected_attribute(self):
        df = pd.DataFrame({
            'location': ['Mumbai', 'Bangalore', 'Rural Bihar', 'Mumbai', 'Bangalore'],
            'education': ['Bachelors', 'Masters', 'High-school', 'Bachelors', 'Masters'],
            'hired': [1, 1, 0, 1, 0]
        })
        result = apply_reweighing(df, 'location', 'hired')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert 'instance_weights' in result.columns

class TestApplyThresholdOptimization:
    def test_threshold_optimization(self):
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
        df['gender'] = np.random.choice([0, 1], 100)
        df['label'] = y

        model = LogisticRegression(random_state=42)
        model.fit(df.drop(['gender', 'label'], axis=1), df['label'])

        result = apply_threshold_optimization(model, df, 'gender')
        assert hasattr(result, 'predict')

class TestIndiaSpecificRecommendations:
    def test_caste_recommendation(self):
        result = get_india_specific_recommendations('caste')
        assert 'SC/ST' in result
        assert 'NEP 2020' in result

    def test_language_recommendation(self):
        result = get_india_specific_recommendations('language')
        assert 'Indic' in result
        assert 'IndicBERT' in result

    def test_region_recommendation(self):
        result = get_india_specific_recommendations('region')
        assert 'North/South/East/West' in result

    def test_unknown_dimension(self):
        result = get_india_specific_recommendations('unknown')
        assert 'General' in result