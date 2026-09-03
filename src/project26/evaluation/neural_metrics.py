from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_classification(predictions, targets):
    precision, recall, macro_f1, _ = precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(targets, predictions), "precision_macro": precision, "recall_macro": recall, "f1_macro": macro_f1}
