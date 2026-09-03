import os
import json


def write_quality_report(dataframe, raw_file, metadata, has_audio, has_images, config):
    missing = dataframe.isnull().mean().sort_values(ascending=False)
    report = f'''# Data Quality Memo - Project 26: Cognitive-Affective Multimodal Support

## Dataset
- Source: {raw_file}
- Rows: {len(dataframe)}
- Duplicate rows: {dataframe.duplicated().sum()}
- Resolved target: {metadata["target_raw"]} ({metadata["n_classes"]} classes: {metadata["class_names"]})

## Modality inspection result
- Raw audio files present in download: {has_audio}
- Raw image files present in download: {has_images}
- Audio-reference columns in CSV: {metadata["audio_id_cols"]}
- Image-reference columns in CSV: {metadata["image_id_cols"]}

## Missingness
{missing[missing > 0].head(10).to_string() if (missing > 0).any() else "No missing values in top columns."}

## Leakage risks identified
- {"Subject/session column(s) found (" + str(metadata["subject_cols"]) + ") -> group-aware split used." if metadata["subject_cols"] else "No repeated-subject column found -> stratified random split used."}
'''
    path = os.path.join(config["reports_dir"], "data_quality_memo.md")
    with open(path, "w") as file:
        file.write(report)
    print(report)
    return path
