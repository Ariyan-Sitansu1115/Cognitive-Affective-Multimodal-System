from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.models.audio_classifier import AudioMLP
from project26.runtime import set_seed
from project26.uncertainty.entropy import calculate_uncertainty


SEED = 42
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_actors(actor_ids: list[int]) -> dict[str, list[int]]:
    generator = np.random.RandomState(SEED)
    shuffled = list(actor_ids)
    generator.shuffle(shuffled)
    return {"train": shuffled[:17], "validation": shuffled[17:21], "test": shuffled[21:]}


def evaluate(model: nn.Module, features: np.ndarray, targets: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    uncertainty, confidence, predictions = calculate_uncertainty(probabilities)
    return probabilities, predictions, confidence, uncertainty


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_path = ROOT / "data/26/processed/audio/audio_features.csv"
    checkpoint_path = ROOT / "data/26/results/audio_best_model.pt"
    metrics_path = ROOT / "data/26/results/audio_metrics.json"
    predictions_path = ROOT / "data/26/results/audio_predictions.csv"
    baseline_path = ROOT / "data/26/results/best_hybrid.pt"
    baseline_hash_before = file_sha256(baseline_path)

    dataframe = pd.read_csv(input_path)
    report = json.loads((ROOT / "data/26/reports/audio_feature_extraction.json").read_text(encoding="utf-8"))
    feature_names = report["feature_names"]
    class_names = report["label_classes"]
    required_columns = {"actor_id", "label", *feature_names}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Audio feature table is missing columns: {sorted(missing)}")

    splits = split_actors(sorted(dataframe["actor_id"].unique().tolist()))
    split_frames = {name: dataframe[dataframe["actor_id"].isin(actor_ids)].copy() for name, actor_ids in splits.items()}
    if set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"]) or set(splits["validation"]) & set(splits["test"]):
        raise AssertionError("Actor split overlap detected")

    encoder = LabelEncoder().fit(class_names)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(split_frames["train"][feature_names])
    train_features = scaler.fit_transform(train_values).astype(np.float32)
    transformed = {
        "train": train_features,
        "validation": scaler.transform(imputer.transform(split_frames["validation"][feature_names])).astype(np.float32),
        "test": scaler.transform(imputer.transform(split_frames["test"][feature_names])).astype(np.float32),
    }
    targets = {name: encoder.transform(frame["label"]).astype(np.int64) for name, frame in split_frames.items()}

    train_counts = np.bincount(targets["train"], minlength=len(class_names))
    class_weights = train_counts.sum() / (len(class_names) * np.maximum(train_counts, 1))
    model = AudioMLP(len(feature_names), len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(torch.from_numpy(transformed["train"]), torch.from_numpy(targets["train"])), batch_size=BATCH_SIZE, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(torch.from_numpy(transformed["validation"]).to(device)), torch.from_numpy(targets["validation"]).to(device)).item()
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_val_loss - 1e-8:
            best_val_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    checkpoint = {"model_state_dict": best_state, "feature_names": feature_names, "class_names": class_names, "architecture": "AudioMLP(34 -> 128 -> 64 -> 8)", "seed": SEED}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    probabilities, predictions, confidence, uncertainty = evaluate(model, transformed["test"], targets["test"], device)
    normalized_uncertainty = uncertainty / np.log(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(targets["test"], predictions, labels=np.arange(len(class_names)), zero_division=0)
    matrix = confusion_matrix(targets["test"], predictions, labels=np.arange(len(class_names)))
    metrics = {
        "experiment": "independent audio-modality classification; not paired with the tabular CSV",
        "seed": SEED,
        "device": str(device),
        "feature_names": feature_names,
        "class_names": class_names,
        "actor_split": splits,
        "sample_split": {name: int(len(frame)) for name, frame in split_frames.items()},
        "actor_counts": {name: len(ids) for name, ids in splits.items()},
        "class_distribution_train": {class_names[index]: int(count) for index, count in enumerate(train_counts)},
        "class_weighted_loss": True,
        "class_weights": class_weights.tolist(),
        "model_architecture": "AudioMLP(34 -> 128 -> 64 -> 8)",
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_validation_loss": best_val_loss,
        "accuracy": float(np.mean(predictions == targets["test"])),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": {class_names[index]: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index in range(len(class_names))},
        "uncertainty_summary": {"mean_confidence": float(confidence.mean()), "mean_entropy_uncertainty": float(uncertainty.mean()), "mean_normalized_uncertainty": float(normalized_uncertainty.mean()), "min_entropy_uncertainty": float(uncertainty.min()), "max_entropy_uncertainty": float(uncertainty.max())},
        "preprocessing": "SimpleImputer(median) and StandardScaler fitted on training actors only",
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path),
    }
    prediction_data = {"actor_id": split_frames["test"]["actor_id"].to_numpy(), "true_label": split_frames["test"]["label"].to_numpy(), "predicted_label": encoder.inverse_transform(predictions), "confidence": confidence, "entropy_uncertainty": uncertainty, "normalized_uncertainty": normalized_uncertainty}
    prediction_data.update({f"probability_{name}": probabilities[:, index] for index, name in enumerate(class_names)})
    pd.DataFrame(prediction_data).to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if file_sha256(baseline_path) != baseline_hash_before:
        raise AssertionError("Frozen tabular checkpoint changed")
    if not np.isfinite(probabilities).all() or not np.isfinite(uncertainty).all() or not np.isfinite(normalized_uncertainty).all():
        raise AssertionError("Non-finite prediction or uncertainty values")
    if len(predictions) != len(split_frames["test"]):
        raise AssertionError("Missing test predictions")

    print("AUDIO MODEL TRAINING COMPLETE")
    print("Independent audio-modality experiment: not paired with the tabular CSV.")
    print("Actor split:", json.dumps(splits))
    print("Sample split:", metrics["sample_split"])
    print("Model architecture:", metrics["model_architecture"])
    print("Class-weighted cross entropy:", metrics["class_weights"])
    print("Training result: best epoch", best_epoch, "of", len(history), "epochs")
    print("Test accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Confusion matrix:\n", np.asarray(matrix))
    print("Uncertainty summary:", json.dumps(metrics["uncertainty_summary"]))
    print("Checkpoint path:", checkpoint_path)
    print("Metrics path:", metrics_path)
    print("Predictions path:", predictions_path)
    print("Tabular baseline untouched: yes")


if __name__ == "__main__":
    main()