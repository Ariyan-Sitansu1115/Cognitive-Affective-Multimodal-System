from lightgbm import LGBMClassifier


def create_model(seed):
    return LGBMClassifier(n_estimators=600, num_leaves=15, min_child_samples=20, learning_rate=0.025, reg_lambda=1.0, class_weight="balanced", random_state=seed, verbosity=-1)
