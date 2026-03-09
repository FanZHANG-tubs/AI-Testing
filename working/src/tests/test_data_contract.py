
import pandas as pd
import os
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
RAW_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'train.csv')
FEAT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_feat.csv')

# define data contracts
raw_schema = DataFrameSchema({
    'Survived':   Column(int,   Check.isin([0, 1])),
    'Pclass':     Column(int,   Check.isin([1, 2, 3])),
    'Age':        Column(float, Check.in_range(0, 120), nullable=True),
    'SibSp':      Column(int,   Check.ge(0)),
    'Parch':      Column(int,   Check.ge(0)),
    'Fare':       Column(float, Check.ge(0)),
    'Sex':        Column(str,   Check.isin(['male', 'female'])),
    'Embarked':   Column(str,   Check.isin(['S', 'C', 'Q']), nullable=True),
})

feat_schema = DataFrameSchema({
    'Survived':    Column(int,   Check.isin([0, 1])),
    'Pclass':      Column(int,   Check.isin([1, 2, 3])),
    'Age':         Column(float),
    'SibSp':       Column(float),
    'Parch':       Column(float),
    'Fare':        Column(float),
    'Sex_male':    Column(int,   Check.isin([0, 1])),
    'Embarked_Q':  Column(int,   Check.isin([0, 1])),
    'Embarked_S':  Column(int,   Check.isin([0, 1])),
})

def test_raw_schema():
    df = pd.read_csv(RAW_PATH)
    raw_schema.validate(df)
    print("[PASS] raw schema verification passed")

def test_feat_schema():
    df = pd.read_csv(FEAT_PATH)
    feat_schema.validate(df)
    print("[PASS] feat schema verification passed")

def test_no_duplicate_rows():
    df = pd.read_csv(RAW_PATH)
    assert df.duplicated().sum() == 0, \
        f"[FAIL] raw data contains duplicate rows: {df.duplicated().sum()}"
    print("[PASS] raw data has no duplicate rows")

def test_label_distribution():
    df = pd.read_csv(FEAT_PATH)
    ratio = df['Survived'].mean()
    assert 0.2 < ratio < 0.8, \
        f"[FAIL] label distribution is extremely imbalanced: {ratio:.4f}"
    print(f"[PASS] label distribution is normal: {ratio:.4f}")