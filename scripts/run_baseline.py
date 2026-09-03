from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.baselines.extra_trees import create_model as create_extra_trees
from project26.baselines.lightgbm import create_model as create_lightgbm
from project26.config import CONFIG
from project26.data.inspect import inspect_data
from project26.data.load_raw import load_raw_data
from project26.data.loaders import create_loaders
from project26.data.preprocessing import fit_transform_features, prepare_features
from project26.data.splits import split_data
from project26.evaluation.final_evaluation import evaluate_final
from project26.models.stacking import fit_stacked_hybrid, get_predictions_classification
from project26.models.temporal_fusion import TemporalFusionTabular
from project26.runtime import configure_runtime
from project26.training.checkpoint import load_checkpoint


def main():
    device = configure_runtime()
    checkpoint_path = ROOT / CONFIG["results_dir"] / "best_hybrid.pt"
    csv_files = [str(path) for path in (ROOT / CONFIG["data_raw_dir"]).rglob("*.csv")]

    raw_file, dataframe = load_raw_data(csv_files)
    metadata = inspect_data(dataframe)
    feature_df, numeric_cols, categorical_cols = prepare_features(dataframe, metadata)
    train_df, val_df, test_df, _ = split_data(feature_df, metadata, CONFIG)
    train_X, val_X, test_X = fit_transform_features(
        train_df.copy(),
        val_df.copy(),
        test_df.copy(),
        numeric_cols,
        categorical_cols,
        metadata["target"],
    )
    _, val_loader, test_loader, _ = create_loaders(
        train_X,
        val_X,
        test_X,
        train_df,
        val_df,
        test_df,
        metadata["target"],
        CONFIG["batch_size"],
    )

    hybrid = TemporalFusionTabular(
        input_dim=train_X.shape[1],
        output_dim=metadata["n_classes"],
        hidden_dim=CONFIG["transformer_hidden"],
    )
    hybrid = load_checkpoint(hybrid, checkpoint_path, device)
    val_logits, val_targets = get_predictions_classification(
        hybrid, val_loader, checkpoint_path, device
    )
    test_logits, test_targets = get_predictions_classification(
        hybrid, test_loader, checkpoint_path, device
    )
    val_neural_probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    test_neural_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()

    baseline_models = {
        "extra_trees": create_extra_trees(CONFIG["random_seed"]),
        "lightgbm": create_lightgbm(CONFIG["random_seed"]),
    }
    val_parts = [val_neural_probabilities]
    test_parts = [test_neural_probabilities]
    for model in baseline_models.values():
        model.fit(train_X, train_df[metadata["target"]])
        val_parts.append(model.predict_proba(val_X))
        test_parts.append(model.predict_proba(test_X))

    meta_search, _, _, test_preds, test_targets = fit_stacked_hybrid(
        val_parts,
        test_parts,
        val_targets,
        test_targets,
        CONFIG["random_seed"],
    )
    metrics, _, confusion = evaluate_final(
        test_preds,
        test_targets,
        metadata["class_names"],
    )

    print("Checkpoint path:", checkpoint_path)
    print("Device:", device)
    print("Test samples:", len(test_targets))
    print("Selected meta C:", meta_search.best_params_["meta__C"])
    print("Test accuracy:", metrics["accuracy"])
    print("Macro precision:", metrics["precision_macro"])
    print("Macro recall:", metrics["recall_macro"])
    print("Macro F1:", metrics["f1_macro"])
    print("Confusion matrix:\n", confusion)


if __name__ == "__main__":
    main()