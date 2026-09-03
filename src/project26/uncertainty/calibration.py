import numpy as np


def calculate_ece(probabilities, targets, n_bins=10):
    confidences = probabilities.max(axis=1); predictions = probabilities.argmax(axis=1); ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lower) & ((confidences <= upper) if i == n_bins - 1 else (confidences < upper))
        if not np.any(mask): continue
        ece += np.mean(mask) * abs(np.mean(predictions[mask] == targets[mask]) - np.mean(confidences[mask]))
    return ece
