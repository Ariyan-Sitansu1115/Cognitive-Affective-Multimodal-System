import pandas as pd


def create_results(probabilities, uncertainty, confidence, predictions, targets, class_names):
    return pd.DataFrame({"true_state": [class_names[int(y)] for y in targets], "predicted_state": [class_names[int(p)] for p in predictions], "confidence": confidence, "uncertainty_entropy": uncertainty})
