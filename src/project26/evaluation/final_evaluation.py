import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_final(test_preds, test_targets_final, class_names):
    test_preds = np.asarray(test_preds); test_targets_final = np.asarray(test_targets_final)
    metrics = {"accuracy": accuracy_score(test_targets_final, test_preds), "precision_macro": precision_score(test_targets_final, test_preds, average="macro", zero_division=0), "recall_macro": recall_score(test_targets_final, test_preds, average="macro", zero_division=0), "f1_macro": f1_score(test_targets_final, test_preds, average="macro", zero_division=0)}
    return metrics, classification_report(test_targets_final, test_preds, target_names=class_names, zero_division=0), confusion_matrix(test_targets_final, test_preds)
