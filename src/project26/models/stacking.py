import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_predictions_classification(model, loader, ckpt_path, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device)); model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            all_preds.append(model(x.to(device)).cpu().numpy()); all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def fit_stacked_hybrid(val_parts, test_parts, val_targets, test_targets, seed):
    stack_val_X, stack_test_X = np.hstack(val_parts), np.hstack(test_parts)
    meta_pipeline = Pipeline([("scale", StandardScaler()), ("meta", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed))])
    meta_search = GridSearchCV(meta_pipeline, {"meta__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}, scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
    meta_search.fit(stack_val_X, val_targets)
    return meta_search, stack_val_X, stack_test_X, meta_search.predict(stack_test_X), test_targets
