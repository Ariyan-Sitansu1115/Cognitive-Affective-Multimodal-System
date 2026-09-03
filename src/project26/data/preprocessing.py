import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(dataframe, metadata):
    target_raw = metadata["target_raw"]
    target = metadata["target"]
    subject_cols = metadata["subject_cols"]
    feature_df = dataframe.drop(columns=[target_raw])
    numeric_cols = [column for column in feature_df.columns if column not in (target,) and pd.api.types.is_numeric_dtype(feature_df[column])]
    numeric_cols = [column for column in numeric_cols if feature_df[column].nunique(dropna=True) > 1]
    categorical_cols = [column for column in feature_df.columns if column not in (target,) + tuple(numeric_cols) and feature_df[column].dtype == object and feature_df[column].nunique(dropna=True) <= 50]
    print("Numeric:", len(numeric_cols), "| Categorical (<=50 unique):", len(categorical_cols))
    feature_cols = numeric_cols + categorical_cols + ([subject_cols[0]] if subject_cols else []) + [target]
    feature_cols = list(dict.fromkeys(feature_cols))
    return feature_df[feature_cols], numeric_cols, categorical_cols


def fit_transform_features(train_df, val_df, test_df, numeric_cols, categorical_cols, target):
    if numeric_cols:
        imputer = SimpleImputer(strategy="median").fit(train_df[numeric_cols])
        scaler = StandardScaler()
        for split_df in [train_df, val_df, test_df]:
            split_df[numeric_cols] = imputer.transform(split_df[numeric_cols])
        scaler.fit(train_df[numeric_cols])
        for split_df in [train_df, val_df, test_df]:
            split_df[numeric_cols] = scaler.transform(split_df[numeric_cols])
    for column in categorical_cols:
        for split_df in (train_df, val_df, test_df):
            split_df[column] = split_df[column].astype(str).fillna("missing")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_df[categorical_cols]) if categorical_cols else None
    cat_names = encoder.get_feature_names_out(categorical_cols).tolist() if encoder else []

    def build_X(split_df):
        parts = []
        if numeric_cols:
            parts.append(split_df[numeric_cols].reset_index(drop=True))
        if encoder:
            parts.append(pd.DataFrame(encoder.transform(split_df[categorical_cols]), columns=cat_names).reset_index(drop=True))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame({"_bias": np.ones(len(split_df))})

    train_X = build_X(train_df); val_X = build_X(val_df); test_X = build_X(test_df)
    print("Feature dim:", train_X.shape[1])
    return train_X, val_X, test_X
