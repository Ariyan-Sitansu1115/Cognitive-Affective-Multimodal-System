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

    def combine(
        self,
        probabilities: Mapping[str, np.ndarray],
        shared_sample_ids: Sequence[object] | None = None,
    ) -> np.ndarray:
        """Combine explicitly paired rows using uncertainty-aware reliability weighting.
        
        ALGORITHM: Entropy-Aware Modality Reliability Fusion
        
        For each sample with predictions from M aligned modalities:
        
        1. Compute Shannon entropy for each modality m:
           H_m(i) = -sum_k p_m,k(i) * log(p_m,k(i))
           
           Higher entropy indicates lower confidence/higher uncertainty.
           Range: [0, log(n_classes)]
           
        2. Compute dynamic weight combining learned reliability and entropy:
           w_m(i) = reliability_m / (H_m(i) + eps)
           
           where:
           - reliability_m ∈ (0, 1] learned from validation set: reliability = exp(-NLL)
           - H_m(i) is per-sample entropy (instantaneous uncertainty)
           - eps = 1e-8 prevents division by zero
           
           Interpretation: 
           - Higher reliability_m → higher weight for that modality
           - Higher H_m(i) → lower weight for that sample from that modality
           - Together: "trust reliable modalities, but less when uncertain"
           
        3. Normalize weights per sample:
           w̃_m(i) = w_m(i) / sum_m' w_m'(i)
           
        4. Fuse as weighted average:
           p_fused(i) = sum_m w̃_m(i) * p_m(i)
           
        5. Renormalize final probabilities to sum to 1.
        
        VALIDITY REQUIREMENTS:
        - All probability matrices must be (n_samples, n_classes) and normalized
        - All modalities must have identical sample and class dimensions
        - shared_sample_ids must uniquely identify aligned rows (prevents misalignment)
        - Class spaces must be identical (prevent semantic mismatch)
        
        FAILS SAFELY: Raises ValueError if any requirement violated.
        """
        if shared_sample_ids is None:
            raise ValueError("Sample-level fusion requires verified shared sample IDs")
        identifiers = np.asarray(shared_sample_ids, dtype=object)
        if identifiers.ndim != 1 or len(set(identifiers.tolist())) != len(identifiers):
            raise ValueError("shared_sample_ids must be a unique one-dimensional sequence")
        names = tuple(probabilities)
        expected = tuple(self.reliability)
        if set(names) != set(expected):
            raise ValueError(f"Expected modalities {expected}, received {names}")

        matrices = {name: _probability_matrix(probabilities[name], name) for name in names}
        shape = next(iter(matrices.values())).shape
        if shape[0] != len(identifiers):
            raise ValueError("shared_sample_ids must have one value per paired prediction row")
        if shape[1] != len(self.class_names):
            raise ValueError("Probability dimensions must match FusionParameters.class_names")
        if any(matrix.shape != shape for matrix in matrices.values()):
            raise ValueError("All modality predictions must have the same paired row and class dimensions")

        # Step 1: Compute per-sample Shannon entropy for each modality
        entropy = {
            name: -np.sum(matrix * np.log(np.clip(matrix, 1e-12, 1.0)), axis=1)
            for name, matrix in matrices.items()
        }
        
        # Step 2: Compute raw dynamic weights (reliability / entropy)
        raw_weights = {
            name: self.reliability[name] / (entropy[name] + 1e-8)
            for name in names
        }
        
        # Step 3: Normalize weights per sample
        denominator = np.sum(np.stack(tuple(raw_weights.values()), axis=1), axis=1)
        
        # Step 4: Weighted average fusion
        fused = np.zeros(shape, dtype=float)
        for name in names:
            fused += (raw_weights[name] / denominator)[:, None] * matrices[name]
        
        # Step 5: Renormalize probabilities
        return fused / fused.sum(axis=1, keepdims=True)


def learn_reliability(
    validation_probabilities: Mapping[str, np.ndarray],
    validation_targets: Sequence[int],
    class_names: Sequence[str],
) -> FusionParameters:
    """Learn modality reliability scores from aligned validation predictions.
    
    RELIABILITY METRIC: exp(-Negative Log-Likelihood)
    
    For each modality m over aligned validation set:
    
    1. Compute Negative Log-Likelihood (NLL):
       NLL_m = (1/n) * sum_i -log(p_m,true_i(i))
       
       where p_m,true_i(i) is the predicted probability of the true class.
       
       Lower NLL = modality assigns higher probability to true class
                  = more reliable modality
    
    2. Compute reliability as:
       reliability_m = exp(-NLL_m)
       
       This maps NLL to a factor in (0, 1]:
       - Perfect predictions (p_true ≈ 1): NLL ≈ 0 → reliability ≈ 1.0
       - Random guessing (p_true ≈ 1/C): NLL ≈ log(C) → reliability ≈ 1/C
       - Terrible predictions (p_true ≈ 0): NLL → ∞ → reliability ≈ 0
    
    VALIDITY CONSTRAINT:
    - Every row i in validation_probabilities across all modalities must correspond
      to the SAME example with SAME ground truth label.
    - Violating this assumption produces meaningless reliability scores.
    
    Args:
        validation_probabilities: Dict mapping modality names to (n_val, n_classes)
                                 probability matrices. ALL must have identical dimensions
                                 and aligned row order.
        validation_targets: (n_val,) array of true class indices (0 to n_classes-1)
        class_names: Ordered list of class label names (defines class space)
    
    Returns:
        FusionParameters with learned reliability scores and validation NLL values.
    
    Raises:
        ValueError: If class_names has duplicates, probability shapes mismatch,
                   or array lengths disagree.
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