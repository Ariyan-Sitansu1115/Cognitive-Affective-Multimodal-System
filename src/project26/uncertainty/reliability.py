import numpy as np


def get_reliability_data(probabilities, targets, n_bins=10):
    confidences = probabilities.max(axis=1); predictions = probabilities.argmax(axis=1); bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences, bin_accuracies, bin_counts = [], [], []
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lower) & ((confidences <= upper) if i == n_bins - 1 else (confidences < upper))
        if np.any(mask):
            bin_confidences.append(confidences[mask].mean()); bin_accuracies.append((predictions[mask] == targets[mask]).mean()); bin_counts.append(mask.sum())
    return np.asarray(bin_confidences), np.asarray(bin_accuracies), np.asarray(bin_counts)
