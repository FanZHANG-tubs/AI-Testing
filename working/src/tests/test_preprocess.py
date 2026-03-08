# src/tests/test_preprocess.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
from preprocess import preprocess

def make_sample_df():
    """构造含缺失值、异常值的最小 DataFrame"""
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
        'Embarked':    ['S', 'C', 'S', 'Q', None],  # 确保有 S/C/Q 三个值
    })

# 测试1：输出无缺失值
def test_no_missing_values():
    df = make_sample_df()
    result = preprocess(df)
    assert result.isnull().sum().sum() == 0, \
        f"[FAIL] 仍有缺失值:\n{result.isnull().sum()}"

# 测试2：输出列数正确（9列）
def test_column_count():
    df = make_sample_df()
    result = preprocess(df)
    assert result.shape[1] == 9, \
        f"[FAIL] 列数不符，期望9，实际{result.shape[1]}，列名: {result.columns.tolist()}"

# 测试3：必要列都存在
def test_required_columns_exist():
    df = make_sample_df()
    result = preprocess(df)
    required = ['Survived', 'Pclass', 'Age', 'Fare', 'Sex_male']
    for col in required:
        assert col in result.columns, f"[FAIL] 缺少列: {col}"

# 测试4：bool 列已转为 int
def test_no_bool_columns():
    df = make_sample_df()
    result = preprocess(df)
    bool_cols = result.select_dtypes(include='bool').columns.tolist()
    assert len(bool_cols) == 0, f"[FAIL] 仍有 bool 列: {bool_cols}"