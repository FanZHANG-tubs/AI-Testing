# src/tests/test_data_contract.py
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

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
    df = pd.read_csv(r'D:\AI_testing_HandsOn\working\data\raw\train.csv')
    raw_schema.validate(df)
    print("[PASS] raw schema 验证通过")

def test_feat_schema():
    df = pd.read_csv(r'D:\AI_testing_HandsOn\working\data\processed\train_feat.csv')
    feat_schema.validate(df)
    print("[PASS] feat schema 验证通过")

def test_no_duplicate_rows():
    df = pd.read_csv(r'D:\AI_testing_HandsOn\working\data\raw\train.csv')
    assert df.duplicated().sum() == 0, \
        f"[FAIL] 存在重复行: {df.duplicated().sum()}"
    print("[PASS] 无重复行")

def test_label_distribution():
    df = pd.read_csv(r'D:\AI_testing_HandsOn\working\data\processed\train_feat.csv')
    ratio = df['Survived'].mean()
    assert 0.2 < ratio < 0.8, \
        f"[FAIL] 标签分布极度不平衡: {ratio:.4f}"
    print(f"[PASS] 标签分布正常: {ratio:.4f}")