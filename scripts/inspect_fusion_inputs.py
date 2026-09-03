"""Inspect existing prediction artifacts for scientifically valid fusion inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "26" / "results"
sys.path.insert(0, str(ROOT / "src"))


def describe_prediction(path: Path) -> dict:
    frame = pd.read_csv(path)
    probability_columns = [column for column in frame if column.startswith("probability_")]
    uncertainty_columns = [column for column in frame if "uncertainty" in column.lower()]
    identifier_columns = [
        column for column in frame
        if column in {"sample_id", "actor_id", "image_path"} or column.endswith("_id")
    ]
    labels = sorted(set(frame["true_label"].astype(str)) | set(frame["predicted_label"].astype(str)))
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(frame),
        "columns": list(frame.columns),
        "probability_columns": probability_columns,
        "probability_vector_dimension": len(probability_columns),
        "uncertainty_columns": uncertainty_columns,
        "identifier_columns": identifier_columns,
        "observed_labels": labels,
    }


def main() -> None:
    tabular_checkpoint = RESULTS / "best_hybrid.pt"
    checkpoint = torch.load(tabular_checkpoint, map_location="cpu", weights_only=False)
    tabular_prediction_files = sorted(RESULTS.glob("*tabular*prediction*.csv"))
    tabular_available = bool(tabular_prediction_files)
    tabular_classes = sorted(
        pd.read_csv(ROOT / "data" / "26" / "raw" / "mental_health_multimodal.csv")
        ["Mental_Health_Status"].astype(str).unique()
    )

    audio = describe_prediction(RESULTS / "audio_predictions.csv")
    image = describe_prediction(RESULTS / "image_predictions.csv")
    reasons = [
        "No shared sample identifier exists across the independent tabular, audio, and image test artifacts.",
        "The image prediction table does not persist probability vectors.",
        "The class vocabularies are incompatible: tabular has 4 classes, audio has 8, and image has 7.",
    ]
    report = {
        "tabular": {
            "prediction_table_available": tabular_available,
            "prediction_table_candidates": [path.relative_to(ROOT).as_posix() for path in tabular_prediction_files],
            "test_probabilities_confidence_uncertainty_persisted": False,
            "computed_in_memory_by": [
                "scripts/run_baseline.py (test_neural_probabilities)",
                "src/project26/inference/predictor.py (confidence, entropy_uncertainty, normalized_uncertainty)",
            ],
            "checkpoint_is_state_dict_only": isinstance(checkpoint, dict) and all(hasattr(value, "shape") for value in checkpoint.values()),
            "checkpoint_keys": list(checkpoint) if isinstance(checkpoint, dict) else [],
            "class_names_from_raw_data": tabular_classes,
        },
        "audio": audio,
        "image": image,
        "class_name_compatibility": {
            "all_equal": tabular_classes == audio["observed_labels"] == image["observed_labels"],
            "tabular": tabular_classes,
            "audio": audio["observed_labels"],
            "image": image["observed_labels"],
        },
        "direct_sample_level_fusion": {"valid": False, "reasons": reasons},
        "verdict": "B. MODALITY-LEVEL FUSION FRAMEWORK IS VALID BUT SAMPLE-LEVEL EVALUATION IS NOT",
        "framework_status": "Implemented for future paired probability matrices only; no current artifacts are fused.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()