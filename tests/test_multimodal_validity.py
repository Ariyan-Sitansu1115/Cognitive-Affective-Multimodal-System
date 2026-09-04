import numpy as np
import pytest

from project26.models.modality_fusion import learn_reliability
from project26.uncertainty.entropy import calculate_uncertainty


def test_probability_dimensions_and_uncertainty():
    probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
    entropy, confidence, predictions = calculate_uncertainty(probabilities)
    assert probabilities.shape == (2, 2)
    assert np.allclose(confidence, [0.8, 0.75])
    assert predictions.tolist() == [0, 1]
    assert np.all(entropy >= 0)


def test_class_consistency_and_reliability_weighting():
    probabilities = {"a": np.array([[0.9, 0.1], [0.2, 0.8]]), "b": np.array([[0.6, 0.4], [0.4, 0.6]])}
    parameters = learn_reliability(probabilities, [0, 1], ["healthy", "stress"])
    fused = parameters.combine(probabilities, ["sample_a", "sample_b"])
    assert fused.shape == (2, 2)
    assert parameters.reliability["a"] > parameters.reliability["b"]
    assert np.allclose(fused.sum(axis=1), 1.0)


def test_missing_modality_can_be_rejected_without_pairing():
    parameters = learn_reliability({"a": np.array([[0.8, 0.2]])}, [0], ["x", "y"])
    with pytest.raises(ValueError, match="shared sample IDs"):
        parameters.combine({"a": np.array([[0.8, 0.2]])})


def test_equal_lengths_without_verified_identity_reject_fusion():
    parameters = learn_reliability(
        {"tabular": np.tile([[0.8, 0.2]], (100, 1)), "image": np.tile([[0.7, 0.3]], (100, 1))},
        np.zeros(100, dtype=int),
        ["healthy", "stress"],
    )
    with pytest.raises(ValueError, match="shared sample IDs"):
        parameters.combine({"tabular": np.tile([[0.8, 0.2]], (100, 1)), "image": np.tile([[0.7, 0.3]], (100, 1))})