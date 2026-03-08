# src/preprocess.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(df, scale=True):
    """
    完整预处理流程：
    - 删除无关列
    - 缺失值填补
    - One-Hot 编码
    - 标准化（可选）
    """
    df = df.copy()

    # 1 删除无关列（存在才删，避免测试数据报错）
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 2 缺失值填补
    if 'Age' in df.columns:
        age_imputer = SimpleImputer(strategy='median')
        df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()

    if 'Embarked' in df.columns:
        emb_imputer = SimpleImputer(strategy='most_frequent')
        df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()

    # 3 One-Hot 编码
    ohe_cols = [c for c in ['Sex', 'Embarked'] if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # 4 bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 5 标准化
    if scale:
        scale_cols = [c for c in ['Age', 'Fare', 'SibSp', 'Parch'] if c in df.columns]
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df