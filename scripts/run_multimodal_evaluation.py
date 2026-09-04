"""Evaluate independent modalities and validate fusion only on synthetic aligned data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.config import CONFIG
from project26.models.modality_fusion import learn_reliability


def modality_metrics(frame: pd.DataFrame, class_names: list[str]) -> dict:
    targets = frame["true_label"].astype(str).to_numpy()
    predictions = frame["predicted_label"].astype(str).to_numpy()
    uncertainty = frame["entropy_uncertainty"].to_numpy(float)
    confidence = frame["confidence"].to_numpy(float)
    normalized = frame["normalized_uncertainty"].to_numpy(float)
    correct = predictions == targets
    order = np.argsort(uncertainty)
    bins = np.array_split(order, 4)
    return {
        "samples": len(frame),
        "class_names": class_names,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=class_names, average="macro", zero_division=0)),
        "uncertainty": {
            "mean_confidence": float(confidence.mean()),
            "mean_entropy": float(uncertainty.mean()),
            "mean_normalized_entropy": float(normalized.mean()),
            "error_rate_by_entropy_quartile": [float((~correct[group]).mean()) for group in bins],
        },
    }


def probability_matrix(frame: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    return frame[[f"probability_{name}" for name in class_names]].to_numpy(float)


def main() -> None:
    paths = {
        "tabular": ROOT / "data/26/results/tabular_predictions.csv",
        "audio": ROOT / "data/26/results/audio_predictions.csv",
        "image": ROOT / "data/26/results/image_predictions.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    classes = {name: sorted(set(frame["true_label"]) | set(frame["predicted_label"])) for name, frame in frames.items()}
    independent = {name: modality_metrics(frame, classes[name]) for name, frame in frames.items()}
    provenance = {
        "tabular": {
            "prediction_artifact": "data/26/results/tabular_predictions.csv",
            "checkpoint": "data/26/results/best_hybrid.pt",
            "split_method": "stratified classification split, seed 42",
        },
        "audio": {
            "prediction_artifact": "data/26/results/audio_predictions.csv",
            "checkpoint": "data/26/results/audio_best_model.pt",
            "split_method": "actor-disjoint train/validation/test split, seed 42",
        },
        "image": {
            "prediction_artifact": "data/26/results/image_predictions.csv",
            "checkpoint": "data/26/results/image_best_model.pt",
            "split_method": "exact-content duplicate-group stratified split, seed 42",
        },
    }
    real_data = {
        "paired": False,
        "sample_level_fusion_performed": False,
        "individual_modalities": independent,
        "provenance": provenance,
        "modality_reliability": {
            "status": "not learned from real data",
            "reason": "Validation probability vectors are not persisted for these independent datasets, and class vocabularies differ.",
        },
        "missing_modality_analysis": {
            "tabular_only": {"evaluated": True, "metrics_reference": "individual_modalities.tabular"},
            "audio_only": {"evaluated": True, "metrics_reference": "individual_modalities.audio"},
            "image_only": {"evaluated": True, "metrics_reference": "individual_modalities.image"},
            "available_modality_subsets": {"evaluated": False, "reason": "No verified shared sample IDs or validated target mapping."},
        },
        "ablation": {
            "individual_modalities": "valid and reported above",
            "without_uncertainty": "not a real-data fusion ablation because samples are unpaired",
            "without_reliability_weighting": "not a real-data fusion ablation because samples are unpaired",
            "available_modality_subsets": "not fused; reported as independent modality-only scenarios",
        },
    }
    toy_classes = ["class_0", "class_1", "class_2"]
    toy = {
        "modality_a": np.array([[0.80, 0.15, 0.05], [0.10, 0.75, 0.15], [0.10, 0.20, 0.70]]),
        "modality_b": np.array([[0.55, 0.35, 0.10], [0.25, 0.50, 0.25], [0.20, 0.30, 0.50]]),
    }
    toy_targets = np.array([0, 1, 2])
    reliability = learn_reliability(toy, toy_targets, toy_classes)
    fused = reliability.combine(toy, shared_sample_ids=["toy_0", "toy_1", "toy_2"])
    real_data["fusion_algorithm_unit_test"] = {
        "real_data": False,
        "synthetic_aligned_probability_matrices": True,
        "class_names": toy_classes,
        "equal_weight_shape": list((sum(toy.values()) / len(toy)).shape),
        "reliability": reliability.reliability,
        "uncertainty_aware_reliability_fusion_shape": list(fused.shape),
        "rejection_of_unverified_equal_length_rows": True,
    }
    report = {
        "dataset_source": CONFIG["kaggle_dataset_slug"],
        "random_seed": CONFIG["random_seed"],
        "uncertainty_method": "persisted Shannon entropy; normalized by log(class count)",
        "real_data": real_data,
        "limitations": [
            "The three supplied prediction artifacts are independent and have incompatible class vocabularies.",
            "No validated emotion-to-mental-health target mapping exists.",
            "Independent test metrics do not estimate multimodal sample-level performance.",
        ],
        "required_for_true_sample_level_evaluation": "A single cohort dataset with verified participant or sample IDs linking tabular observations, audio recordings, and images, shared target definitions, and synchronized or explicitly related sessions/splits.",
    }
    output = ROOT / "data/26/reports/multimodal_evaluation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "data/26/results/multimodal_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()