"""Persist predictions from the frozen tabular checkpoint.

The generated sample_id is deterministic but local to the tabular CSV. It is
never a claim that tabular rows correspond to audio or image samples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.config import CONFIG
from project26.data.inspect import inspect_data
from project26.data.load_raw import load_raw_data
from project26.data.loaders import create_loaders
from project26.data.preprocessing import fit_transform_features, prepare_features
from project26.data.splits import split_data
from project26.models.stacking import get_predictions_classification
from project26.models.temporal_fusion import TemporalFusionTabular
from project26.runtime import configure_runtime
from project26.training.checkpoint import load_checkpoint
from project26.uncertainty.entropy import calculate_uncertainty


def main() -> None:
    device = configure_runtime()
    checkpoint_path = ROOT / CONFIG["results_dir"] / "best_hybrid.pt"
    output_path = ROOT / CONFIG["results_dir"] / "tabular_predictions.csv"
    csv_files = [str(path) for path in (ROOT / CONFIG["data_raw_dir"]).rglob("*.csv")]

    raw_file, dataframe = load_raw_data(csv_files)
    metadata = inspect_data(dataframe)
    feature_df, numeric_cols, categorical_cols = prepare_features(dataframe, metadata)
    train_df, val_df, test_df, split_manifest = split_data(feature_df, metadata, CONFIG)
    train_X, val_X, test_X = fit_transform_features(
        train_df.copy(), val_df.copy(), test_df.copy(), numeric_cols, categorical_cols, metadata["target"]
    )
    _, val_loader, test_loader, _ = create_loaders(
        train_X, val_X, test_X, train_df, val_df, test_df, metadata["target"], CONFIG["batch_size"]
    )
    model = TemporalFusionTabular(
        input_dim=train_X.shape[1], output_dim=metadata["n_classes"], hidden_dim=CONFIG["transformer_hidden"]
    )
    model = load_checkpoint(model, checkpoint_path, device)
    test_logits, test_targets = get_predictions_classification(model, test_loader, checkpoint_path, device)
    probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    uncertainty, confidence, predictions = calculate_uncertainty(probabilities)
    normalized = uncertainty / np.log(len(metadata["class_names"]))
    class_names = metadata["class_names"]
    rows = {
        "sample_id": [f"tabular_row_{index:06d}" for index in test_df.index],
        "true_label": [class_names[int(index)] for index in test_targets],
        "predicted_label": [class_names[int(index)] for index in predictions],
        "confidence": confidence,
        "entropy_uncertainty": uncertainty,
        "normalized_uncertainty": normalized,
    }
    rows.update({f"probability_{name}": probabilities[:, index] for index, name in enumerate(class_names)})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    provenance = {
        "dataset_source": CONFIG["kaggle_dataset_slug"],
        "source_file": str(Path(raw_file).relative_to(ROOT).as_posix()),
        "modality": "tabular",
        "samples": len(test_targets),
        "class_names": class_names,
        "split_method": split_manifest["split_strategy"],
        "random_seed": CONFIG["random_seed"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "uncertainty_method": "Shannon entropy; normalized by log(number of classes)",
        "paired_samples": False,
        "sample_id_scope": "deterministic row identifier local to the tabular dataset; never a cross-modal join key",
        "prediction_model": "frozen TemporalFusionTabular checkpoint probabilities",
        "output": str(output_path.relative_to(ROOT).as_posix()),
    }
    provenance_path = output_path.with_name("tabular_predictions_metadata.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()