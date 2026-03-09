
import os
import sys
import pandas as pd
import numpy as np
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'train.csv')
FEAT_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'train_feat.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'randomforest.joblib')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import preprocess
from model import train_model, evaluate_model

def test_end_to_end_pipeline():
    """From raw data to prediction results"""

    # 1. Load raw data
    df = pd.read_csv(RAW_PATH)
    assert df.shape[0] > 0, "[FAIL] raw data is empty"
    print(f"[PASS] raw data loaded successfully: {df.shape}")

    # 2. Preprocess
    df_feat = preprocess(df)
    assert df_feat.isnull().sum().sum() == 0, "[FAIL] preprocessed data contains nulls"
    assert df_feat.shape[1] == 9,             "[FAIL] feature count mismatch"
    print(f"[PASS] preprocess completed: {df_feat.shape}")

    # 3. Train
    from sklearn.model_selection import train_test_split
    X = df_feat.drop(columns=['Survived'])
    y = df_feat['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model, cv_scores = train_model(X_train, y_train)
    assert cv_scores.mean() > 0.70, \
        f"[FAIL] CV mean is too low: {cv_scores.mean():.4f}"
    print(f"[PASS] Training completed: cv={cv_scores.mean():.4f}")

    # 4. Evaluate
    acc, report = evaluate_model(model, X_test, y_test)
    assert acc > 0.75, f"[FAIL] End-to-end accuracy is too low: {acc:.4f}"
    print(f"[PASS] Evaluation completed: acc={acc:.4f}")

    # 5. Predict output is valid
    y_pred = model.predict(X_test)
    assert set(y_pred).issubset({0, 1}), "[FAIL] Predicted values are out of range"
    print("[PASS] Predicted values are within the valid range")

def test_single_sample_prediction():
    """Single sample input and output format is correct"""
    df    = pd.read_csv(FEAT_PATH)
    X     = df.drop(columns=['Survived']).iloc[[0]]
    model = joblib.load(MODEL_PATH)

    pred = model.predict(X)
    prob = model.predict_proba(X)

    assert pred.shape == (1,),       "[FAIL] Predicted output shape is incorrect"
    assert prob.shape  == (1, 2),    "[FAIL] Probability output shape is incorrect"
    assert pred[0] in {0, 1},        "[FAIL] Predicted value is invalid"
    assert 0.0 <= prob[0][1] <= 1.0, "[FAIL] Probability value is out of range"
    print(f"[PASS] Single sample prediction successful  pred={pred[0]}  prob={prob[0][1]:.4f}")

def test_batch_prediction():
    """Batch prediction output count matches input"""
    df    = pd.read_csv(FEAT_PATH)
    X     = df.drop(columns=['Survived'])
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    assert len(y_pred) == len(X), \
        f"[FAIL] Predicted count does not match input: {len(y_pred)} vs {len(X)}"
    assert y_prob.shape == (len(X), 2), \
        f"[FAIL] Probability matrix shape is incorrect: {y_prob.shape}"
    assert set(y_pred).issubset({0, 1}), \
        "[FAIL] Batch predicted values are out of range"
    print(f"[PASS] Batch prediction successful: {len(y_pred)} samples")
    