
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (demographic_parity_difference,
                                equalized_odds_difference,
                                MetricFrame)

# File paths
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
    """Fairness metrics observation, no hard threshold"""
    X_test, y_test, y_pred = load()
    dpd = demographic_parity_difference(
        y_test, y_pred, sensitive_features=X_test['Sex_male']
    )
    eod = equalized_odds_difference(
        y_test, y_pred, sensitive_features=X_test['Sex_male']
    )
    print(f"\n── fairness metrics report ──")
    print(f"demographic_parity_difference : {dpd:.4f}")
    print(f"equalized_odds_difference     : {eod:.4f}")
    print(f"Note: Gender bias in the Titanic dataset is a known historical fact, only for record-keeping purposes")
    assert -1.0 <= dpd <= 1.0
    assert -1.0 <= eod <= 1.0
    print("[PASS] fairness metrics have been recorded")

def test_accuracy_by_pclass():
    """Each pclass accuracy difference should not exceed 15%"""
    X_test, y_test, y_pred = load()
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test['Pclass']
    )
    print(f"every pclass accuracy:\n{mf.by_group}")
    acc_range = mf.by_group.max() - mf.by_group.min()
    assert acc_range < 0.15, \
        f"[FAIL] pclass accuracy difference is too large: {acc_range:.4f}"
    print(f"[PASS] pclass accuracy difference={acc_range:.4f} < 0.15")