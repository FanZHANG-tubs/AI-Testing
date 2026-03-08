# src/model.py
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """训练 RandomForest，返回模型和 CV 分数"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    model.fit(X_train, y_train)
    return model, cv_scores

def evaluate_model(model, X_test, y_test):
    """输出 accuracy 和 classification_report"""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['Not Survived', 'Survived']
    )
    return acc, report

def save_model(model, path='../models/randomforest.joblib'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path='../models/randomforest.joblib'):
    return joblib.load(path)