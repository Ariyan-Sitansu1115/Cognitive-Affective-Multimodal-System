"""Normalized prediction artifact schema for cross-modality evaluation and fusion.

This module defines a reusable schema for persisting modality predictions in a standard
format suitable for:
- Independent modality performance evaluation
- Uncertainty analysis and calibration studies
- Framework-level multimodal fusion validation
- Research reproducibility and documentation

The schema explicitly records:
- Modality identifier and label space (prevents implicit class mapping)
- Per-sample predictions with full probability vectors
- Uncertainty metrics (confidence, entropy, normalized entropy)
- Original and predicted labels for error analysis

Usage:
    from project26.data.normalized_schema import PredictionRecord, normalize_predictions
    
    records = normalize_predictions(
        modality="audio",
        true_labels=["neutral", "calm"],
        predicted_labels=["calm", "calm"],
        probabilities=[[0.1, 0.9], [0.2, 0.8]],
        class_names=["angry", "calm"],
        confidence=[0.9, 0.8],
        entropy_uncertainty=[0.32, 0.50],
        normalized_uncertainty=[0.15, 0.24],
        sample_ids=["actor_007_sample_1", "actor_007_sample_2"],
    )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionRecord:
    """Single prediction with uncertainty metrics and probability vector.
    
    Attributes:
        modality: Source modality identifier (e.g., "audio", "image", "tabular")
        sample_id: Unique sample identifier within the modality
        true_label: Ground truth class label
        predicted_label: Model's predicted class label
        confidence: Maximum probability (argmax confidence)
        entropy_uncertainty: Shannon entropy of probability distribution
        normalized_uncertainty: Entropy normalized to [0, 1] by max possible entropy
        probability_vector: Full probability distribution as list of floats
        class_names: Ordered list of class names (defines the label space)
    """

    modality: str
    sample_id: str
    true_label: str
    predicted_label: str
    confidence: float
    entropy_uncertainty: float
    normalized_uncertainty: float
    probability_vector: tuple[float, ...]
    class_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary suitable for DataFrame serialization."""
        d = asdict(self)
        # Expand probability vector into separate columns
        for idx, class_name in enumerate(self.class_names):
            d[f"probability_{class_name.lower()}"] = self.probability_vector[idx]
        del d["probability_vector"]
        # Convert tuples to lists for JSON serialization
        d["class_names"] = list(d["class_names"])
        return d

    @classmethod
    def from_row(cls, row: Mapping[str, Any], class_names: Sequence[str]) -> PredictionRecord:
        """Reconstruct from DataFrame row with expanded probability columns."""
        class_names_tuple = tuple(class_names)
        prob_vector = tuple(row[f"probability_{name.lower()}"] for name in class_names)
        return cls(
            modality=row["modality"],
            sample_id=row["sample_id"],
            true_label=row["true_label"],
            predicted_label=row["predicted_label"],
            confidence=row["confidence"],
            entropy_uncertainty=row["entropy_uncertainty"],
            normalized_uncertainty=row["normalized_uncertainty"],
            probability_vector=prob_vector,
            class_names=class_names_tuple,
        )


def normalize_predictions(
    modality: str,
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    confidence: np.ndarray,
    entropy_uncertainty: np.ndarray,
    normalized_uncertainty: np.ndarray,
    sample_ids: Sequence[str],
) -> list[PredictionRecord]:
    """Create normalized prediction records from modality outputs.

    Args:
        modality: Source modality name (e.g., "audio", "image", "tabular")
        true_labels: Ground truth labels for each sample
        predicted_labels: Model predictions for each sample
        probabilities: (n_samples, n_classes) probability matrix
        class_names: Ordered class label names
        confidence: (n_samples,) confidence scores (max probability)
        entropy_uncertainty: (n_samples,) Shannon entropy scores
        normalized_uncertainty: (n_samples,) entropy normalized to [0, 1]
        sample_ids: (n_samples,) unique sample identifiers

    Returns:
        List of PredictionRecord instances.

    Raises:
        ValueError: If array shapes or lengths are inconsistent.
    """
    n_samples = len(true_labels)
    if (
        len(predicted_labels) != n_samples
        or len(confidence) != n_samples
        or len(entropy_uncertainty) != n_samples
        or len(normalized_uncertainty) != n_samples
        or len(sample_ids) != n_samples
    ):
        raise ValueError("All input sequences must have the same length")

    if probabilities.shape != (n_samples, len(class_names)):
        raise ValueError(
            f"Probability matrix shape {probabilities.shape} does not match "
            f"({n_samples}, {len(class_names)})"
        )

    class_names_tuple = tuple(class_names)
    records = []
    for i in range(n_samples):
        record = PredictionRecord(
            modality=modality,
            sample_id=str(sample_ids[i]),
            true_label=str(true_labels[i]),
            predicted_label=str(predicted_labels[i]),
            confidence=float(confidence[i]),
            entropy_uncertainty=float(entropy_uncertainty[i]),
            normalized_uncertainty=float(normalized_uncertainty[i]),
            probability_vector=tuple(float(p) for p in probabilities[i]),
            class_names=class_names_tuple,
        )
        records.append(record)
    return records


def records_to_dataframe(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """Convert normalized records to DataFrame with expanded probability columns.
    
    Args:
        records: List of PredictionRecord instances
        
    Returns:
        DataFrame with modality, sample_id, labels, uncertainty metrics, and
        per-class probability columns (probability_<classname_lowercase>).
    """
    if not records:
        raise ValueError("Records list cannot be empty")
    rows = [record.to_dict() for record in records]
    return pd.DataFrame(rows)


def dataframe_to_records(frame: pd.DataFrame) -> list[PredictionRecord]:
    """Reconstruct normalized records from DataFrame.
    
    Args:
        frame: DataFrame produced by records_to_dataframe()
        
    Returns:
        List of PredictionRecord instances.
    """
    # Infer class names from probability columns
    prob_cols = [col for col in frame.columns if col.startswith("probability_")]
    class_names = [col.replace("probability_", "").title() for col in prob_cols]
    
    records = []
    for _, row in frame.iterrows():
        record = PredictionRecord.from_row(row, class_names)
        records.append(record)
    return records
