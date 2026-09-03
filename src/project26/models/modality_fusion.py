"""Uncertainty-aware fusion for genuinely paired modality predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _probability_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} probabilities must be a 2D array")
    if matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} probabilities must be finite with at least two classes")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{name} probabilities must be non-negative and sum to one")
    return matrix


@dataclass(frozen=True)
class FusionParameters:
    """Reliability learned from aligned held-out predictions."""

    class_names: tuple[str, ...]
    validation_nll: Mapping[str, float]
    reliability: Mapping[str, float]

    def combine(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        """Combine aligned probability rows using reliability and entropy weights."""
        names = tuple(probabilities)
        expected = tuple(self.reliability)
        if set(names) != set(expected):
            raise ValueError(f"Expected modalities {expected}, received {names}")

        matrices = {name: _probability_matrix(probabilities[name], name) for name in names}
        shape = next(iter(matrices.values())).shape
        if any(matrix.shape != shape for matrix in matrices.values()):
            raise ValueError("All modality predictions must have the same paired row and class dimensions")

        entropy = {
            name: -np.sum(matrix * np.log(np.clip(matrix, 1e-12, 1.0)), axis=1)
            for name, matrix in matrices.items()
        }
        raw_weights = {
            name: self.reliability[name] / (entropy[name] + 1e-8)
            for name in names
        }
        denominator = np.sum(np.stack(tuple(raw_weights.values()), axis=1), axis=1)
        fused = np.zeros(shape, dtype=float)
        for name in names:
            fused += (raw_weights[name] / denominator)[:, None] * matrices[name]
        return fused / fused.sum(axis=1, keepdims=True)


def learn_reliability(
    validation_probabilities: Mapping[str, np.ndarray],
    validation_targets: Sequence[int],
    class_names: Sequence[str],
) -> FusionParameters:
    """Learn non-manual modality reliabilities from aligned validation predictions.

    Reliability is ``exp(-NLL)``. This is only valid when every modality's validation
    row refers to the same example and target.
    """
    classes = tuple(class_names)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_names must contain at least two unique classes")
    targets = np.asarray(validation_targets, dtype=int)
    if targets.ndim != 1 or np.any((targets < 0) | (targets >= len(classes))):
        raise ValueError("validation_targets must be valid class indices")

    nll: dict[str, float] = {}
    for name, values in validation_probabilities.items():
        matrix = _probability_matrix(values, name)
        if matrix.shape != (len(targets), len(classes)):
            raise ValueError(f"{name} validation predictions are not aligned to targets and classes")
        nll[name] = float(-np.log(np.clip(matrix[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())
    if not nll:
        raise ValueError("At least one modality is required")
    return FusionParameters(
        class_names=classes,
        validation_nll=nll,
        reliability={name: float(np.exp(-loss)) for name, loss in nll.items()},
    )