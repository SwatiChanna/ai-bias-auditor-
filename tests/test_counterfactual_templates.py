import pytest
import pandas as pd
from src.counterfactual_templates import (
    generate_gender_counterfactual,
    generate_caste_counterfactual,
    generate_language_counterfactual,
    generate_region_counterfactual,
    validate_counterfactual,
)

class TestGenderCounterfactual:
    def test_text_swap_pronouns(self):
        original = "He is a great actor and she is his queen."
        expected = "she is a great actress and she is her queen."
        result = generate_gender_counterfactual(original)
        assert result == expected

    def test_text_case_insensitive(self):
        original = "The MAN is KING."
        expected = "The woman is queen."
        result = generate_gender_counterfactual(original)
        assert result == expected

    def test_dataframe_flip_gender(self):
        df = pd.DataFrame({'gender': [0, 1, 0], 'name': ['Alice', 'Bob', 'Charlie']})
        result = generate_gender_counterfactual(df)
        expected = pd.DataFrame({'gender': [1, 0, 1], 'name': ['Alice', 'Bob', 'Charlie']})
        pd.testing.assert_frame_equal(result, expected)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            generate_gender_counterfactual(123)

class TestCasteCounterfactual:
    def test_text_swap_caste(self):
        original = "Ramesh is a Brahmin and Dharmesh is Dalit."
        expected = "Ramesh is a Brahmin and Ramesh is Brahmin."
        result = generate_caste_counterfactual(original)
        assert result == expected

    def test_text_name_swap(self):
        original = "Rahul got the job."
        expected = "Dharmesh got the job."
        result = generate_caste_counterfactual(original)
        assert result == expected

    def test_dataframe_swap_caste(self):
        df = pd.DataFrame({'caste': ['Brahmin', 'Dalit'], 'name': ['Ramesh', 'Dharmesh']})
        result = generate_caste_counterfactual(df)
        expected = pd.DataFrame({'caste': ['Dalit', 'Brahmin'], 'name': ['Dharmesh', 'Ramesh']})
        pd.testing.assert_frame_equal(result, expected)

class TestLanguageCounterfactual:
    def test_hindi_translation(self):
        original = "He got the job"
        expected = "woh job mila"
        result = generate_language_counterfactual(original, target_lang='hi')
        assert result == expected

    def test_tamil_translation(self):
        original = "She is a doctor"
        expected = "aval oru maruthuvar"
        result = generate_language_counterfactual(original, target_lang='ta')
        assert result == expected

    def test_unsupported_lang(self):
        original = "He is here"
        result = generate_language_counterfactual(original, target_lang='fr')
        assert result == original

class TestRegionCounterfactual:
    def test_text_swap_region(self):
        original = "IIT Delhi graduate from Mumbai"
        expected = "IIT Jharkhand graduate from Rural Bihar"
        result = generate_region_counterfactual(original)
        assert result == expected

    def test_dataframe_swap_region(self):
        df = pd.DataFrame({'region': ['Mumbai', 'Delhi'], 'education': ['IIT', 'DU']})
        result = generate_region_counterfactual(df)
        expected = pd.DataFrame({'region': ['Rural Bihar', 'Jharkhand'], 'education': ['IIT', 'DU']})
        pd.testing.assert_frame_equal(result, expected)

class TestValidateCounterfactual:
    def test_valid_dataframe(self):
        orig = pd.DataFrame({'gender': [0, 1], 'name': ['Alice', 'Bob'], 'age': [25, 30]})
        cf = pd.DataFrame({'gender': [1, 0], 'name': ['Alice', 'Bob'], 'age': [25, 30]})
        assert validate_counterfactual(orig, cf, ['name', 'age']) == True

    def test_invalid_dataframe(self):
        orig = pd.DataFrame({'gender': [0, 1], 'name': ['Alice', 'Bob']})
        cf = pd.DataFrame({'gender': [1, 0], 'name': ['Alice', 'Charlie']})
        assert validate_counterfactual(orig, cf, ['name']) == False

    def test_non_dataframe(self):
        assert validate_counterfactual("text", "cf", []) == True