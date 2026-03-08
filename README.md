# ML/AI Testing Framework — Titanic Survival Prediction

![CI](https://github.com/FanZHANG-tubs/AI-Testing/actions/workflows/ml_tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A end-to-end ML testing framework built around the Titanic dataset, demonstrating how to apply systematic software quality assurance principles to machine learning systems — from raw data validation to model explainability and CI/CD integration.

> Built as a hands-on portfolio project aligned with **ISTQB® Certified Tester AI Testing** principles.

---

## What This Project Covers

| Area | What Was Done |
|------|--------------|
| Data Validation | Schema contracts with `pandera`, label distribution checks |
| Preprocessing | Missing value imputation, one-hot encoding, standardization |
| Model Training | Baseline comparison (LogisticRegression, RandomForest, XGBoost), cross-validation |
| Functional Testing | `pytest` unit tests for preprocessing and model logic |
| Robustness Testing | Gaussian noise, random mislabeling, extreme value injection |
| Fairness Testing | Demographic parity & equalized odds with `fairlearn` |
| Explainability | SHAP summary, bar, and waterfall plots |
| Regression Testing | Automated baseline comparison — blocks deployment if performance drops >2% |
| Integration Testing | End-to-end pipeline test from raw CSV to prediction output |
| CI/CD | GitHub Actions pipeline: auto-runs all tests on every push |
| Coverage Report | `pytest-cov` HTML coverage report |

---

## Project Structure

```
AI-Testing/
├── .github/
│   └── workflows/
│       └── ml_tests.yml          # GitHub Actions CI pipeline
├── working/
│   ├── data/
│   │   ├── raw/                  # Original Titanic dataset
│   │   └── processed/            # Cleaned and feature-engineered data
│   ├── models/
│   │   └── randomforest.joblib   # Trained model artifact
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   ├── 04_functional_testing.ipynb
│   │   ├── 05_robustness_testing.ipynb
│   │   └── 06_nonfunctional_explainability.ipynb
│   ├── results/
│   │   ├── metrics.txt
│   │   ├── robustness.txt
│   │   ├── baseline_metrics.json
│   │   ├── nonfunctional_report.txt
│   │   ├── test_report.html
│   │   ├── coverage_report/
│   │   └── figures/              # Confusion matrix, ROC, SHAP plots
│   └── src/
│       ├── preprocess.py         # preprocess() — reusable pipeline function
│       ├── model.py              # train_model(), evaluate_model()
│       └── tests/
│           ├── test_preprocess.py
│           ├── test_model.py
│           ├── test_data_contract.py
│           ├── test_regression.py
│           ├── test_fairness.py
│           └── test_integration.py
└── requirements.txt
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/FanZHANG-tubs/AI-Testing.git
cd AI-Testing
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Download the dataset

Place the [Titanic dataset](https://www.kaggle.com/c/titanic/data) (`train.csv`) into:
```
working/data/raw/train.csv
```

### 3. Run the notebooks in order

```
01 → 02 → 03 → 04 → 05 → 06
```

### 4. Run all tests

```bash
# Unit + data contract tests
pytest working/src/tests/test_preprocess.py working/src/tests/test_model.py working/src/tests/test_data_contract.py -v

# Save regression baseline (run once)
python working/src/tests/test_regression.py

# Full test suite with coverage report
pytest working/src/tests/ -v \
  --cov=working/src \
  --cov-report=html:working/results/coverage_report \
  --html=working/results/test_report.html \
  --self-contained-html
```

---

## Key Design Decisions

**Why StratifiedKFold instead of plain KFold?**
The Titanic dataset is imbalanced (~38% survived). StratifiedKFold preserves the class ratio in each fold, giving more reliable CV scores.

**Why SHAP instead of LIME?**
LIME approximates model behavior locally via a linear surrogate, making results sensitive to random sampling. SHAP is grounded in Shapley values from cooperative game theory, guaranteeing consistency and global interpretability. In this project, SHAP revealed that `Sex_male` is by far the most influential feature, followed by `Fare` and `Age`.

**Why observational-only fairness thresholds for Sex?**
The `demographic_parity_difference` for Sex is 0.63 — high, but this reflects a genuine historical pattern (women and children first). Enforcing a strict threshold would suppress a real signal. In a production business context (e.g. credit scoring), hard thresholds would be appropriate.

**Why pandera for data contracts?**
Data quality issues caught late are expensive. pandera allows defining explicit schemas (column types, value ranges, nullable rules) that fail fast at the data ingestion stage — a practical implementation of Shift-Left Testing.

---

## Test Results Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_preprocess.py | 4 | Passed |
| test_model.py | 3 | Passed |
| test_data_contract.py | 4 | Passed |
| test_regression.py | 3 | Passed |
| test_fairness.py | 2 | Passed |
| test_integration.py | 3 | Passed |
| **Total** | **19** | **All Passed** |

**Model performance (hold-out set):**
- Accuracy: 0.8156
- AUC: 0.8339

---

## Robustness Test Highlights

The model was tested under 9 different perturbation scenarios:

| Scenario | Acc Drop | Assessment |
|----------|----------|------------|
| Gaussian noise Age+Fare (std=0.5) | +0.0838 | Acceptable |
| Gaussian noise Age+Fare (std=2.0) | +0.0950 | Borderline stable |
| Sex mislabeling 20% | +0.0838 | High sensitivity (expected) |
| Embarked mislabeling 20% | -0.0112 | Negligible impact |
| Extreme Age=0 | +0.0670 | Moderate impact |
| Extreme Fare=0 | +0.0168 | Minimal impact |

**Key finding:** Age and Sex are the most sensitive features. Embarked contributes minimally to predictions. AUC remained above 0.5 in all scenarios.

---

## CI/CD Pipeline

Every push to `master` or `main` triggers the full pipeline:

1. Set up Python 3.11
2. Install all dependencies
3. Download and preprocess the Titanic dataset
4. Run data contract tests
5. Run unit tests (preprocessing + model)
6. Upload test reports as artifacts

Pipeline config: [`.github/workflows/ml_tests.yml`](.github/workflows/ml_tests.yml)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | scikit-learn, XGBoost |
| Data Validation | pandera |
| Explainability | SHAP |
| Fairness | fairlearn |
| Testing | pytest, pytest-cov, pytest-html |
| CI/CD | GitHub Actions |
| Visualization | matplotlib, seaborn |

---

## Background

This project was developed as part of a practical study for the **ISTQB® Certified Tester AI Testing** certification, with the goal of bridging the gap between ML engineering and systematic software quality assurance. It demonstrates that AI systems require testing strategies that go beyond traditional functional tests — including data contracts, robustness validation, fairness auditing, and explainability.
