

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(df, scale=True):
    """
    Complete preprocessing pipeline:
    - Drop irrelevant columns
    - Fill missing values
    - One-Hot Encoding
    - Standardization (optional)
    """
    df = df.copy()

    # 1 Drop irrelevant columns (drop only if they exist to avoid errors in test data)
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 2 Fill missing values
    if 'Age' in df.columns:
        age_imputer = SimpleImputer(strategy='median')
        df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()

    if 'Embarked' in df.columns:
        emb_imputer = SimpleImputer(strategy='most_frequent')
        df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()

    # 3 One-Hot coding 
    ohe_cols = [c for c in ['Sex', 'Embarked'] if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # 4 bool to int conversion
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 5 Standardization
    if scale:
        scale_cols = [c for c in ['Age', 'Fare', 'SibSp', 'Parch'] if c in df.columns]
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df