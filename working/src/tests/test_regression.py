import joblib
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 路径基准
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH     = os.path.join(BASE_DIR, 'data', 'processed', 'train_feat.csv')
MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'randomforest.joblib')
BASELINE_PATH = os.path.join(BASE_DIR, 'results', 'baseline_metrics.json')

THRESHOLDS = {
    'accuracy': 0.78,
    'auc':      0.82,
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=['Survived'])
    y  = df['Survived']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

def test_accuracy_above_threshold():
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_data()
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= THRESHOLDS['accuracy'], \
        f"[FAIL] accuracy {acc:.4f} < 门槛 {THRESHOLDS['accuracy']}"
    print(f"[PASS] accuracy={acc:.4f} >= {THRESHOLDS['accuracy']}")

def test_auc_above_threshold():
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_data()
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc >= THRESHOLDS['auc'], \
        f"[FAIL] auc {auc:.4f} < 门槛 {THRESHOLDS['auc']}"
    print(f"[PASS] auc={auc:.4f} >= {THRESHOLDS['auc']}")

def test_no_regression_vs_baseline():
    if not os.path.exists(BASELINE_PATH):
        print("[SKIP] 无基准文件，跳过回归测试")
        return
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_data()
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert acc >= baseline['accuracy'] - 0.02, \
        f"[FAIL] accuracy 回退: {acc:.4f} vs baseline {baseline['accuracy']:.4f}"
    assert auc >= baseline['auc'] - 0.02, \
        f"[FAIL] auc 回退: {auc:.4f} vs baseline {baseline['auc']:.4f}"
    print(f"[PASS] 无性能回退  acc={acc:.4f}  auc={auc:.4f}")

def save_baseline():
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_data()
    metrics = {
        'accuracy': round(accuracy_score(y_test, model.predict(X_test)), 6),
        'auc':      round(roc_auc_score(y_test,
                          model.predict_proba(X_test)[:, 1]), 6),
    }
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[PASS] 基准已保存: {metrics}")

if __name__ == '__main__':
    save_baseline()