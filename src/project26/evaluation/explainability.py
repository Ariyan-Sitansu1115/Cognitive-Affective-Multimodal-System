import numpy as np
import pandas as pd
import torch


def permutation_importance(hybrid, test_X, hybrid_targets, meta_search, baseline_models, device, seed, base_score, sample_limit=15):
    rng = np.random.default_rng(seed); importances = {}
    for col in list(test_X.columns[:sample_limit]):
        X_perm = test_X.copy(); X_perm[col] = rng.permutation(X_perm[col].values)
        with torch.no_grad(): logits_perm = hybrid(torch.tensor(X_perm.values.astype(np.float32)).to(device)).cpu().numpy()
        perm_parts = [torch.softmax(torch.tensor(logits_perm), dim=1).numpy()]
        perm_parts.extend(baseline.predict_proba(X_perm) for baseline in baseline_models.values())
        importances[col] = base_score - accuracy_score(hybrid_targets, meta_search.predict(np.hstack(perm_parts)))
    return pd.Series(importances).sort_values()
