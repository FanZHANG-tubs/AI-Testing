# src/tests/test_integration.py
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
    """从原始数据到预测结果的完整链路"""

    # 1 加载原始数据
    df = pd.read_csv(RAW_PATH)
    assert df.shape[0] > 0, "[FAIL] 原始数据为空"
    print(f"[PASS] 原始数据加载成功: {df.shape}")

    # 2 预处理
    df_feat = preprocess(df)
    assert df_feat.isnull().sum().sum() == 0, "[FAIL] 预处理后有缺失值"
    assert df_feat.shape[1] == 9,             "[FAIL] 特征数不符"
    print(f"[PASS] 预处理完成: {df_feat.shape}")

    # 3 训练
    from sklearn.model_selection import train_test_split
    X = df_feat.drop(columns=['Survived'])
    y = df_feat['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model, cv_scores = train_model(X_train, y_train)
    assert cv_scores.mean() > 0.70, \
        f"[FAIL] CV 均值过低: {cv_scores.mean():.4f}"
    print(f"[PASS] 训练完成: cv={cv_scores.mean():.4f}")

    # 4 评估
    acc, report = evaluate_model(model, X_test, y_test)
    assert acc > 0.75, f"[FAIL] 端到端 accuracy 过低: {acc:.4f}"
    print(f"[PASS] 评估完成: acc={acc:.4f}")

    # 5 预测输出合法
    y_pred = model.predict(X_test)
    assert set(y_pred).issubset({0, 1}), "[FAIL] 预测值超出范围"
    print("[PASS] 预测值范围合法")

def test_single_sample_prediction():
    """单条样本输入输出格式正确"""
    df    = pd.read_csv(FEAT_PATH)
    X     = df.drop(columns=['Survived']).iloc[[0]]
    model = joblib.load(MODEL_PATH)

    pred = model.predict(X)
    prob = model.predict_proba(X)

    assert pred.shape == (1,),       "[FAIL] 预测输出 shape 不对"
    assert prob.shape  == (1, 2),    "[FAIL] 概率输出 shape 不对"
    assert pred[0] in {0, 1},        "[FAIL] 预测值不合法"
    assert 0.0 <= prob[0][1] <= 1.0, "[FAIL] 概率值超出范围"
    print(f"[PASS] 单条预测正常  pred={pred[0]}  prob={prob[0][1]:.4f}")

def test_batch_prediction():
    """批量预测输出数量与输入一致"""
    df    = pd.read_csv(FEAT_PATH)
    X     = df.drop(columns=['Survived'])
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    assert len(y_pred) == len(X), \
        f"[FAIL] 预测数量不符: {len(y_pred)} vs {len(X)}"
    assert y_prob.shape == (len(X), 2), \
        f"[FAIL] 概率矩阵 shape 不对: {y_prob.shape}"
    assert set(y_pred).issubset({0, 1}), \
        "[FAIL] 批量预测值超出范围"
    print(f"[PASS] 批量预测正常: {len(y_pred)} 条")
    