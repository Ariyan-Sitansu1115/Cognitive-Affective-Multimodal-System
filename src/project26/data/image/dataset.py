from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    group_id: str


def _group_label(paths: list[Path], root: Path) -> str:
    labels = Counter(path.parent.name for path in paths)
    return sorted(labels, key=lambda label: (-labels[label], label))[0]


def discover_image_records(root: Path, class_names: list[str] | None = None) -> tuple[list[ImageRecord], dict]:
    classes = class_names or DEFAULT_CLASSES
    paths = sorted(path for path in root.rglob("*.png") if path.is_file())
    groups: defaultdict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)

    records = []
    for group_id, group_paths in sorted(groups.items()):
        for path in sorted(group_paths):
            label = path.parent.name
            if label not in classes:
                raise ValueError(f"Unexpected image label {label!r} in {path}")
            records.append(ImageRecord(path=path, label=label, group_id=group_id))

    group_labels = {group_id: _group_label(group_paths, root) for group_id, group_paths in groups.items()}
    metadata = {
        "total_images": len(records),
        "unique_content_groups": len(groups),
        "duplicate_groups": sum(len(paths) > 1 for paths in groups.values()),
        "duplicate_files_beyond_first_copy": sum(max(len(paths) - 1, 0) for paths in groups.values()),
        "cross_directory_duplicate_groups": sum(len({path.parent.name for path in paths}) > 1 for paths in groups.values()),
        "classes": classes,
        "class_distribution": dict(Counter(record.label for record in records)),
        "group_labels": group_labels,
    }
    return records, metadata


def split_records(records: list[ImageRecord], metadata: dict, seed: int = 42) -> dict[str, list[ImageRecord]]:
    groups = sorted({record.group_id for record in records})
    group_labels = metadata["group_labels"]
    train_groups, heldout_groups = train_test_split(
        groups, test_size=0.30, random_state=seed, stratify=[group_labels[group] for group in groups]
    )
    val_groups, test_groups = train_test_split(
        sorted(heldout_groups), test_size=0.50, random_state=seed, stratify=[group_labels[group] for group in heldout_groups]
    )
    group_to_split = {group: "train" for group in train_groups}
    group_to_split.update({group: "validation" for group in val_groups})
    group_to_split.update({group: "test" for group in test_groups})
    splits = {name: [record for record in records if group_to_split[record.group_id] == name] for name in ("train", "validation", "test")}
    if set(group_to_split) != set(groups):
        raise AssertionError("Every image group must belong to exactly one split")
    metadata["split_group_counts"] = {name: len({record.group_id for record in split}) for name, split in splits.items()}
    metadata["split_class_distributions"] = {name: dict(Counter(record.label for record in split)) for name, split in splits.items()}
    return splits


class ImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], class_names: list[str]):
        self.records = records
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("L")
            if image.size != (48, 48):
                raise ValueError(f"Expected 48x48 image, found {image.size} in {record.path}")
            tensor = self.transform(image)
        return tensor, self.class_to_index[record.label]