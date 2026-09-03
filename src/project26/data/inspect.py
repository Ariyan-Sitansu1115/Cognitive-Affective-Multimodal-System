def inspect_data(dataframe):
    print("Shape:", dataframe.shape)
    print("Duplicate rows:", dataframe.duplicated().sum())
    target_candidates = [column for column in dataframe.columns if any(key in column.lower() for key in ["risk", "label", "class", "target", "diagnosis", "condition", "stress", "depress", "anxiety"])]
    assert len(target_candidates) >= 1, f"Could not find a mental-health outcome column among: {list(dataframe.columns)}"
    target_raw = "Mental_Health_Status"
    assert target_raw in dataframe.columns, f"Required target {target_raw} is missing"
    print("Target column:", target_raw, dataframe[target_raw].unique()[:10])
    audio_id_cols = [column for column in dataframe.columns if "audio" in column.lower() or "wav" in column.lower()]
    image_id_cols = [column for column in dataframe.columns if "image" in column.lower() or "face" in column.lower() or "photo" in column.lower()]
    subject_cols = [column for column in dataframe.columns if column.lower() in {"subject_id", "patient_id", "participant_id", "user_id"}]
    print("Audio-reference columns:", audio_id_cols)
    print("Image-reference columns:", image_id_cols)
    print("Subject/session columns:", subject_cols)
    class_names = sorted(dataframe[target_raw].astype(str).unique())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    dataframe["_target"] = dataframe[target_raw].astype(str).map(class_to_idx)
    target = "_target"
    n_classes = len(class_names)
    print(f"Classification target: {target_raw}; classes={class_names}")
    target_series = dataframe[target]
    print(target_series.value_counts().sort_index())
    assert target_series.nunique() > 1, f"DEGENERATE TARGET: only {target_series.nunique()} unique value(s) found"
    if len(target_series) < 100:
        print(f"WARNING: only {len(target_series)} rows total - too few to trust reported metrics.")
    return {"target_raw": target_raw, "target": target, "class_names": class_names, "n_classes": n_classes, "audio_id_cols": audio_id_cols, "image_id_cols": image_id_cols, "subject_cols": subject_cols}
