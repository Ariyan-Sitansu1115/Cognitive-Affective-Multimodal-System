def summarize(predictions, targets, confidence, uncertainty):
    correct_mask = predictions == targets
    return {"mean_confidence": confidence.mean(), "mean_uncertainty": uncertainty.mean(), "min_uncertainty": uncertainty.min(), "max_uncertainty": uncertainty.max(), "correct": correct_mask.sum(), "incorrect": (~correct_mask).sum()}
