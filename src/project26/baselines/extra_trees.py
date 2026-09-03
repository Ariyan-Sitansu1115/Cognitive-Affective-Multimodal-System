from sklearn.ensemble import ExtraTreesClassifier


def create_model(seed):
    return ExtraTreesClassifier(n_estimators=700, min_samples_leaf=1, max_features=0.9, class_weight="balanced", n_jobs=-1, random_state=seed)
