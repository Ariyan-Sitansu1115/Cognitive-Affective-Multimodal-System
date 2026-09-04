"""Train and evaluate the independent 48x48 grayscale image classifier."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.data.image.dataset import ImageDataset, discover_image_records, split_records
from project26.models.image_classifier import ImageCNN
from project26.runtime import set_seed
from project26.uncertainty.entropy import calculate_uncertainty


IMAGE_ROOT = ROOT / "data" / "26" / "raw" / "Extracted_images"
RESULTS_ROOT = ROOT / "data" / "26" / "results"
REPORT_PATH = ROOT / "data" / "26" / "reports" / "image_model_training.json"
CHECKPOINT_PATH = RESULTS_ROOT / "image_best_model.pt"
METRICS_PATH = RESULTS_ROOT / "image_metrics.json"
PREDICTIONS_PATH = RESULTS_ROOT / "image_predictions.csv"
SEED = 42
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128


def make_loader(records, class_names, shuffle):
    return DataLoader(ImageDataset(records, class_names), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
    return total_loss / total


def predict(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch_targets.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def main() -> None:
    set_seed(SEED)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records, metadata = discover_image_records(IMAGE_ROOT)
    splits = split_records(records, metadata, SEED)
    class_names = metadata["classes"]

    train_loader = make_loader(splits["train"], class_names, True)
    val_loader = make_loader(splits["validation"], class_names, False)
    test_loader = make_loader(splits["test"], class_names, False)
    train_counts = Counter(record.label for record in splits["train"])
    weights = torch.tensor(
        [len(splits["train"]) / (len(class_names) * train_counts[name]) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    model = ImageCNN(output_dim=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    probabilities, targets = predict(model, test_loader, device)
    uncertainty, confidence, predicted = calculate_uncertainty(probabilities)
    normalized = uncertainty / np.log(len(class_names))
    report = classification_report(
        targets, predicted, labels=np.arange(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(targets, predicted, labels=np.arange(len(class_names)))
    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {key: float(report[name][key]) for key in ("precision", "recall", "f1-score", "support")}
            for name in class_names
        },
        "confusion_matrix": matrix.tolist(),
    }
    prediction_rows = []
    for record, true_index, predicted_index, row_confidence, row_entropy, row_normalized, row_probabilities in zip(
        splits["test"], targets, predicted, confidence, uncertainty, normalized, probabilities
    ):
        row = {
            "image_path": record.path.relative_to(ROOT).as_posix(),
            "true_label": class_names[int(true_index)],
            "predicted_label": class_names[int(predicted_index)],
            "confidence": float(row_confidence),
            "entropy_uncertainty": float(row_entropy),
            "normalized_uncertainty": float(row_normalized),
        }
        row.update({f"probability_{name.lower()}": float(row_probabilities[index]) for index, name in enumerate(class_names)})
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    uncertainty_summary = {
        "mean_confidence": float(np.mean(confidence)),
        "mean_entropy_uncertainty": float(np.mean(uncertainty)),
        "mean_normalized_uncertainty": float(np.mean(normalized)),
        "max_entropy_uncertainty": float(np.max(uncertainty)),
    }
    training_report = {
        "experiment": "independent image-modality experiment",
        "dataset_root": IMAGE_ROOT.relative_to(ROOT).as_posix(),
        "dataset_size": metadata["total_images"],
        "duplicate_handling": "SHA-256 exact-content groups are assigned wholly to one deterministic split; duplicate files are not deleted or modified. For cross-directory groups, the majority directory label (alphabetical tie-break) is used only for stratification.",
        "split_method": "Because reliable subject/actor identifiers are unavailable, this image experiment uses a stratified group-aware split based on exact image duplicates rather than a subject-independent split.",
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "class_distributions": metadata["split_class_distributions"],
        "duplicate_summary": {key: metadata[key] for key in ("unique_content_groups", "duplicate_groups", "duplicate_files_beyond_first_copy", "cross_directory_duplicate_groups")},
        "model_architecture": str(model),
        "training_epochs_requested": EPOCHS,
        "training_epochs_completed": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "test_metrics": metrics,
        "uncertainty_summary": uncertainty_summary,
        "checkpoint_path": CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        "output_paths": [METRICS_PATH.relative_to(ROOT).as_posix(), PREDICTIONS_PATH.relative_to(ROOT).as_posix(), REPORT_PATH.relative_to(ROOT).as_posix()],
        "mapping_status": "No image-to-tabular mapping was created or assumed; no multimodal fusion was performed.",
    }
    REPORT_PATH.write_text(json.dumps(training_report, indent=2), encoding="utf-8")

    print("IMAGE MODEL TRAINING COMPLETE")
    print(f"Dataset size: {metadata['total_images']}")
    print(f"Duplicate groups: {metadata['duplicate_groups']} ({metadata['duplicate_files_beyond_first_copy']} extra files); split crossings: 0")
    print(f"Split sizes: {training_report['split_sizes']}")
    print(f"Class distribution: {training_report['class_distributions']}")
    print(f"Model architecture: ImageCNN, best epoch: {best_epoch}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}")
    print(f"Confusion matrix: {matrix.tolist()}")
    print(f"Uncertainty summary: {uncertainty_summary}")
    print(f"Checkpoint path: {CHECKPOINT_PATH.relative_to(ROOT).as_posix()}")
    print(f"Output paths: {training_report['output_paths']}")
    print("Tabular and audio work untouched; no multimodal fusion performed")


if __name__ == "__main__":
    main()