"""Add frozen-checkpoint image probabilities to the existing prediction CSV."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.data.image.dataset import ImageDataset, discover_image_records, split_records
from project26.models.image_classifier import ImageCNN


def main() -> None:
    source = ROOT / "data/26/results/image_predictions.csv"
    backup = ROOT / "data/26/results/image_predictions_original.csv"
    image_root = ROOT / "data/26/raw/Extracted_images"
    checkpoint = ROOT / "data/26/results/image_best_model.pt"
    records, metadata = discover_image_records(image_root)
    splits = split_records(records, metadata, seed=42)
    frame = pd.read_csv(source)
    required = {"image_path", "true_label", "predicted_label", "confidence", "entropy_uncertainty", "normalized_uncertainty"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Image prediction CSV is missing columns: {sorted(required - set(frame.columns))}")
    expected_paths = [record.path.relative_to(ROOT).as_posix() for record in splits["test"]]
    if frame["image_path"].tolist() != expected_paths:
        raise ValueError("Existing image prediction order does not match the deterministic test split")
    if not backup.exists():
        shutil.copy2(source, backup)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImageCNN(output_dim=len(metadata["classes"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    loader = DataLoader(ImageDataset(splits["test"], metadata["classes"]), batch_size=128, shuffle=False, num_workers=0)
    probability_batches = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            probability_batches.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    probabilities = np.concatenate(probability_batches)
    for index, name in enumerate(metadata["classes"]):
        column = f"probability_{name.lower()}"
        frame[column] = probabilities[:, index]
    frame.to_csv(source, index=False)
    metadata_path = ROOT / "data/26/results/image_predictions_metadata.json"
    metadata_path.write_text(json.dumps({
        "dataset_source": "zara2099/multimodal-mental-health-assessment-dataset",
        "modality": "image",
        "samples": len(frame),
        "class_names": metadata["classes"],
        "split_method": "exact-content duplicate-group stratified split, seed 42",
        "random_seed": 42,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "uncertainty_method": "existing persisted Shannon entropy; normalized by log(number of classes)",
        "paired_samples": False,
        "output": source.relative_to(ROOT).as_posix(),
        "original_prediction_artifact": backup.relative_to(ROOT).as_posix(),
    }, indent=2), encoding="utf-8")
    print(f"Enriched {source.relative_to(ROOT)}; original preserved at {backup.relative_to(ROOT)}")


if __name__ == "__main__":
    main()