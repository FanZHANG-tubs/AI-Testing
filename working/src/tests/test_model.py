# src/tests/test_model.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_model, evaluate_model

def make_mock_data(n=200):
    """生成小型模拟数据"""
    np.random.seed(42)
    X = pd.DataFrame({
        'Pclass':      np.random.choice([1,2,3], n),
        'Age':         np.random.randn(n),
        'SibSp':       np.random.randn(n),
        'Parch':       np.random.randn(n),
        'Fare':        np.random.randn(n),
        'Sex_male':    np.random.choice([0,1], n),
        'Embarked_Q':  np.random.choice([0,1], n),
        'Embarked_S':  np.random.choice([0,1], n),
    })
    y = pd.Series(np.random.choice([0,1], n))
    return X, y

# 测试1：预测标签范围合法（只含 0/1）
def test_predict_labels_valid():
    X, y = make_mock_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model, _ = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    assert set(y_pred).issubset({0, 1}), \
        f"[FAIL] 预测值超出范围: {set(y_pred)}"

# 测试2：CV 分数在合理区间
def test_cv_scores_range():
    X, y = make_mock_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    _, cv_scores = train_model(X_train, y_train)
    assert cv_scores.mean() > 0.4, \
        f"[FAIL] CV 均值过低: {cv_scores.mean():.4f}"
    assert cv_scores.mean() <= 1.0, \
        f"[FAIL] CV 均值异常: {cv_scores.mean():.4f}"

# 测试3：evaluate_model 返回合法 accuracy
def test_evaluate_model_accuracy():
    X, y = make_mock_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model, _ = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)
    assert 0.0 <= acc <= 1.0, \
        f"[FAIL] accuracy 超出范围: {acc}"
    assert isinstance(report, str), \
        "[FAIL] report 不是字符串"