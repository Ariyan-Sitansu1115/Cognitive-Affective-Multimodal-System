import json
import os

from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_data(feature_df, metadata, config):
    target = metadata["target"]
    subject_cols = metadata["subject_cols"]
    ratios = config["split_ratios"]
    if subject_cols:
        gss = GroupShuffleSplit(n_splits=1, train_size=ratios["train"], random_state=42)
        groups = feature_df[subject_cols[0]]
        train_idx, rest_idx = next(gss.split(feature_df, groups=groups))
        train_df, rest_df = feature_df.iloc[train_idx], feature_df.iloc[rest_idx]
        rel_val = ratios["val"] / (ratios["val"] + ratios["test"])
        gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=42)
        val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df[subject_cols[0]]))
        val_df, test_df = rest_df.iloc[val_idx], rest_df.iloc[test_idx]
        split_strategy = "group-aware"
    else:
        train_df, rest_df = train_test_split(feature_df, train_size=ratios["train"], stratify=feature_df[target], random_state=42)
        rel_val = ratios["val"] / (ratios["val"] + ratios["test"])
        val_df, test_df = train_test_split(rest_df, train_size=rel_val, stratify=rest_df[target], random_state=42)
        split_strategy = "stratified classification split"
    print("Split strategy:", split_strategy)
    print("Train / Val / Test sizes:", len(train_df), len(val_df), len(test_df))
    manifest = {"train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df), "split_strategy": split_strategy}
    with open(os.path.join(config["data_processed_dir"], "split_manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2, default=str)
    return train_df, val_df, test_df, manifest
