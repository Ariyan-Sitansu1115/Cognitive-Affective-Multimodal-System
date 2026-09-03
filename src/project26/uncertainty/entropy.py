from scipy.stats import entropy


def calculate_uncertainty(probabilities):
    return entropy(probabilities, axis=1), probabilities.max(axis=1), probabilities.argmax(axis=1)
