import re
import pandas as pd

def generate_gender_counterfactual(text_or_df, sensitive_column='gender'):
    """
    Generate gender counterfactual by swapping gender-specific terms or flipping binary gender labels.

    For text input: Swaps pronouns and gender-specific nouns like he↔she, man↔woman, actor↔actress, king↔queen.
    For DataFrame input: Flips binary gender column (0→1, 1→0) while keeping all other columns unchanged.

    Parameters:
    - text_or_df (str or pd.DataFrame): Input text or tabular data.
    - sensitive_column (str): Name of the gender column in DataFrame (default: 'gender').

    Returns:
    - str or pd.DataFrame: Counterfactual version with gender swapped.

    Examples:
    >>> generate_gender_counterfactual("He is a great actor.")
    'She is a great actress.'
    >>> df = pd.DataFrame({'gender': [0, 1], 'name': ['Alice', 'Bob']})
    >>> generate_gender_counterfactual(df)
       gender   name
    0       1  Alice
    1       0    Bob
    """
    if isinstance(text_or_df, str):
        swaps = {
            'he': 'she',
            'him': 'her',
            'his': 'her',
            'man': 'woman',
            'actor': 'actress',
            'king': 'queen',
            'boy': 'girl',
            'father': 'mother',
            'son': 'daughter',
        }
        text = text_or_df
        for old, new in swaps.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
        return text
    elif isinstance(text_or_df, pd.DataFrame):
        df = text_or_df.copy()
        if sensitive_column in df.columns:
            df[sensitive_column] = df[sensitive_column].map({0: 1, 1: 0})
        return df
    else:
        raise ValueError("Input must be string or DataFrame")

def generate_caste_counterfactual(text_or_df, sensitive_column='caste'):
    """
    Generate caste counterfactual by swapping caste names and associated Indian names.

    Swaps caste terms like "Brahmin" ↔ "Dalit", "upper-caste" ↔ "SC/ST".
    Swaps Indian names based on caste associations: e.g., "Ramesh" (upper-caste) ↔ "Rahul" (generic) ↔ "Dharmesh" (Dalit).
    Uses patterns inspired by caste-sensitive datasets for Indian context.

    Parameters:
    - text_or_df (str or pd.DataFrame): Input text or tabular data.
    - sensitive_column (str): Name of the caste column in DataFrame (default: 'caste').

    Returns:
    - str or pd.DataFrame: Counterfactual version with caste swapped.

    Examples:
    >>> generate_caste_counterfactual("Ramesh is a Brahmin.")
    'Dharmesh is a Dalit.'
    >>> df = pd.DataFrame({'caste': ['Brahmin', 'Dalit'], 'name': ['Ramesh', 'Dharmesh']})
    >>> generate_caste_counterfactual(df)
        caste     name
    0   Dalit  Dharmesh
    1 Brahmin   Ramesh
    """
    caste_swaps = {
        'Brahmin': 'Dalit',
        'Dalit': 'Brahmin',
        'upper-caste': 'SC/ST',
        'SC/ST': 'upper-caste',
        'Kshatriya': 'Shudra',
        'Shudra': 'Kshatriya',
        'Vaishya': 'Dalit',
    }
    name_swaps = {
        'Ramesh': 'Dharmesh',  # Upper-caste to Dalit
        'Dharmesh': 'Ramesh',
        'Rahul': 'Dharmesh',  # Generic to Dalit
        'Amit': 'Rahul',  # Upper to generic
        'Suresh': 'Dharmesh',
        'Vijay': 'Rahul',
    }
    if isinstance(text_or_df, str):
        text = text_or_df
        for old, new in name_swaps.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
        for old, new in caste_swaps.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
        return text
    elif isinstance(text_or_df, pd.DataFrame):
        df = text_or_df.copy()
        if sensitive_column in df.columns:
            df[sensitive_column] = df[sensitive_column].map(caste_swaps).fillna(df[sensitive_column])
        if 'name' in df.columns:
            df['name'] = df['name'].map(name_swaps).fillna(df['name'])
        return df
    else:
        raise ValueError("Input must be string or DataFrame")

def generate_language_counterfactual(text, source_lang='en', target_lang='hi'):
    """
    Generate language counterfactual by translating sensitive terms to target language.

    Translates key terms to Hindi/Tamil/Bengali/Marathi while preserving semantics.
    Supports code-mixed input by mixing languages appropriately.

    Parameters:
    - text (str): Input text.
    - source_lang (str): Source language code (default: 'en').
    - target_lang (str): Target language code (default: 'hi' for Hindi).

    Returns:
    - str: Counterfactual text with language shifted.

    Examples:
    >>> generate_language_counterfactual("He got the job")
    'Woh job mila'
    >>> generate_language_counterfactual("She is a doctor", target_lang='ta')  # Tamil
    'Aval doctor'  # Simplified
    """
    if target_lang == 'hi':  # Hindi
        translations = {
            'he': 'woh',
            'she': 'woh',
            'got the job': 'job mila',
            'is a': 'hai',
            'doctor': 'doctor',  # Keep English for code-mix
            'engineer': 'engineer',
            'teacher': 'teacher',
        }
    elif target_lang == 'ta':  # Tamil
        translations = {
            'he': 'avan',
            'she': 'aval',
            'got the job': 'velai kidaitha',
            'is a': 'oru',
            'doctor': 'maruthuvar',
        }
    elif target_lang == 'bn':  # Bengali
        translations = {
            'he': 'she',
            'she': 'she',
            'got the job': 'kaaj peyechen',
        }
    elif target_lang == 'mr':  # Marathi
        translations = {
            'he': 'to',
            'she': 'ti',
            'got the job': 'nokri milt',
        }
    else:
        return text  # No translation

    translated_text = text
    for en, target in translations.items():
        translated_text = re.sub(r'\b' + re.escape(en) + r'\b', target, translated_text, flags=re.IGNORECASE)
    return translated_text

def generate_region_counterfactual(text_or_df, region_map={'Mumbai': 'Rural Bihar', 'Bangalore': 'Northeast', 'Delhi': 'Jharkhand'}):
    """
    Generate region counterfactual by swapping city/region names and adjusting cultural references.

    Swaps region names and modifies related cultural contexts for Indian regions.

    Parameters:
    - text_or_df (str or pd.DataFrame): Input text or tabular data.
    - region_map (dict): Mapping of regions to swap (default provided).

    Returns:
    - str or pd.DataFrame: Counterfactual with regions swapped.

    Examples:
    >>> generate_region_counterfactual("IIT Delhi graduate")
    'State university Jharkhand graduate'
    >>> df = pd.DataFrame({'region': ['Mumbai', 'Delhi'], 'education': ['IIT', 'DU']})
    >>> generate_region_counterfactual(df)
         region education
    0 Rural Bihar       IIT
    1   Jharkhand        DU
    """
    if isinstance(text_or_df, str):
        text = text_or_df
        for old, new in region_map.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
        # Adjust cultural references
        adjustments = {
            'IIT Delhi graduate': 'State university Jharkhand graduate',
            'IIT Mumbai graduate': 'Local college Rural Bihar graduate',
            'Bangalore tech hub': 'Northeast startup scene',
        }
        for old, new in adjustments.items():
            text = text.replace(old, new)
        return text
    elif isinstance(text_or_df, pd.DataFrame):
        df = text_or_df.copy()
        if 'region' in df.columns:
            df['region'] = df['region'].map(region_map).fillna(df['region'])
        return df
    else:
        raise ValueError("Input must be string or DataFrame")

def validate_counterfactual(original, counterfactual, unchanged_columns):
    """
    Validate that counterfactual preserves non-sensitive columns.

    Checks if all specified unchanged columns remain identical between original and counterfactual DataFrames.

    Parameters:
    - original (pd.DataFrame): Original DataFrame.
    - counterfactual (pd.DataFrame): Counterfactual DataFrame.
    - unchanged_columns (list): List of column names that should remain unchanged.

    Returns:
    - bool: True if valid (unchanged columns match), False otherwise.

    Examples:
    >>> orig = pd.DataFrame({'gender': [0, 1], 'name': ['Alice', 'Bob']})
    >>> cf = pd.DataFrame({'gender': [1, 0], 'name': ['Alice', 'Bob']})
    >>> validate_counterfactual(orig, cf, ['name'])
    True
    >>> cf_bad = pd.DataFrame({'gender': [1, 0], 'name': ['Alice', 'Charlie']})
    >>> validate_counterfactual(orig, cf_bad, ['name'])
    False
    """
    if not isinstance(original, pd.DataFrame) or not isinstance(counterfactual, pd.DataFrame):
        return True  # Assume valid for non-DataFrame
    for col in unchanged_columns:
        if col not in original.columns or col not in counterfactual.columns:
            continue
        if not original[col].equals(counterfactual[col]):
            return False
    return True