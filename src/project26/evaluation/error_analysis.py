import pandas as pd


def error_counts(hybrid_preds, hybrid_targets, class_names):
    misclassified = hybrid_preds != hybrid_targets
    return pd.Series(hybrid_targets[misclassified]).map(dict(enumerate(class_names))).value_counts()


def error_pairs(uncertainty_results):
    return uncertainty_results[uncertainty_results["true_state"] != uncertainty_results["predicted_state"]].groupby(["true_state", "predicted_state"]).size().reset_index(name="count").sort_values("count", ascending=False)
