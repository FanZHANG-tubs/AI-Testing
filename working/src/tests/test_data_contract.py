# src/tests/test_data_contract.py
import pandas as pd
import os
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# 在文件顶部定义基准路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
RAW_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'train.csv')
FEAT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_feat.csv')

# 定义数据契约
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
    print("[PASS] raw schema 验证通过")

def test_feat_schema():
    df = pd.read_csv(FEAT_PATH)
    feat_schema.validate(df)
    print("[PASS] feat schema 验证通过")

def test_no_duplicate_rows():
    df = pd.read_csv(RAW_PATH)
    assert df.duplicated().sum() == 0, \
        f"[FAIL] 原始数据存在重复行: {df.duplicated().sum()}"
    print("[PASS] 原始数据无重复行")

def test_label_distribution():
    df = pd.read_csv(FEAT_PATH)
    ratio = df['Survived'].mean()
    assert 0.2 < ratio < 0.8, \
        f"[FAIL] 标签分布极度不平衡: {ratio:.4f}"
    print(f"[PASS] 标签分布正常: {ratio:.4f}")