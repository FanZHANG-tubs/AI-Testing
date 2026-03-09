
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
from preprocess import preprocess

def make_sample_df():
    """build DataFrame containing all edge cases"""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived':    [0, 1, 1, 0, 1],
        'Pclass':      [3, 1, 3, 1, 2],
        'Name':        ['A', 'B', 'C', 'D', 'E'],
        'Sex':         ['male', 'female', 'female', 'male', 'female'],
        'Age':         [22.0, None, 26.0, None, 35.0],
        'SibSp':       [1, 1, 0, 0, 0],
        'Parch':       [0, 0, 0, 0, 0],
        'Ticket':      ['T1','T2','T3','T4','T5'],
        'Fare':        [7.25, 999.0, 7.92, 53.1, 8.05],
        'Cabin':       [None, 'C85', None, 'C123', None],
        'Embarked':    ['S', 'C', 'S', 'Q', None],  # Ensure S/C/Q values are present
    })

# Test 1: Output has no missing values
def test_no_missing_values():
    df = make_sample_df()
    result = preprocess(df)
    assert result.isnull().sum().sum() == 0, \
        f"[FAIL] Still has missing values:\n{result.isnull().sum()}"

# Test 2: Output has correct column count (9 columns)
def test_column_count():
    df = make_sample_df()
    result = preprocess(df)
    assert result.shape[1] == 9, \
        f"[FAIL] Column count mismatch, expected 9, got {result.shape[1]}, columns: {result.columns.tolist()}"

# Test 3: All required columns exist
def test_required_columns_exist():
    df = make_sample_df()
    result = preprocess(df)
    required = ['Survived', 'Pclass', 'Age', 'Fare', 'Sex_male']
    for col in required:
        assert col in result.columns, f"[FAIL] Missing column: {col}"

# Test 4: bool columns have been converted to int
def test_no_bool_columns():
    df = make_sample_df()
    result = preprocess(df)
    bool_cols = result.select_dtypes(include='bool').columns.tolist()
    assert len(bool_cols) == 0, f"[FAIL] Still has bool columns: {bool_cols}"