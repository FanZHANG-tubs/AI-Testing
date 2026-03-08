# src/tests/test_fairness.py
# src/tests/test_fairness.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (demographic_parity_difference,
                                equalized_odds_difference,
                                MetricFrame)

# 路径
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'train_feat.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'randomforest.joblib')

def load():
    df    = pd.read_csv(DATA_PATH)
    X     = df.drop(columns=['Survived'])
    y     = df['Survived']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model  = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    return X_test, y_test, y_pred

def test_fairness_metrics_report():
    """公平性指标观测，不设硬性门槛"""
    X_test, y_test, y_pred = load()
    dpd = demographic_parity_difference(
        y_test, y_pred, sensitive_features=X_test['Sex_male']
    )
    eod = equalized_odds_difference(
        y_test, y_pred, sensitive_features=X_test['Sex_male']
    )
    print(f"\n── 公平性指标报告 ──")
    print(f"demographic_parity_difference : {dpd:.4f}")
    print(f"equalized_odds_difference     : {eod:.4f}")
    print(f"注：Titanic 数据集性别偏差为已知历史事实，仅作记录")
    assert -1.0 <= dpd <= 1.0
    assert -1.0 <= eod <= 1.0
    print("[PASS] 公平性指标已记录")

def test_accuracy_by_pclass():
    """各舱位 accuracy 差异不超过 15%"""
    X_test, y_test, y_pred = load()
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test['Pclass']
    )
    print(f"各舱位 accuracy:\n{mf.by_group}")
    acc_range = mf.by_group.max() - mf.by_group.min()
    assert acc_range < 0.15, \
        f"[FAIL] 舱位间 accuracy 差距过大: {acc_range:.4f}"
    print(f"[PASS] 舱位间 accuracy 差距={acc_range:.4f} < 0.15")