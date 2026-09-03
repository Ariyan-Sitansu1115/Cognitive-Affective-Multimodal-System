"""Inspect the raw image dataset without modifying or transforming any files."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "26" / "raw" / "Extracted_images"
REPORT_PATH = PROJECT_ROOT / "data" / "26" / "reports" / "image_dataset_inspection.json"
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def relative_path(path: Path) -> str:
    return path.relative_to(IMAGE_ROOT).as_posix()


def inspect_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "resolution": f"{image.width}x{image.height}",
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "channels": len(image.getbands()),
        }


def build_report() -> dict[str, Any]:
    all_files = sorted(path for path in IMAGE_ROOT.rglob("*") if path.is_file())
    image_files = [path for path in all_files if path.suffix.lower() in IMAGE_EXTENSIONS]
    non_image_files = [path for path in all_files if path.suffix.lower() not in IMAGE_EXTENSIONS]
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in all_files)
    valid_files: list[Path] = []
    invalid_files: list[dict[str, str]] = []
    resolutions: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    sizes: list[int] = []
    hashes: defaultdict[str, list[Path]] = defaultdict(list)
    examples: list[dict[str, Any]] = []

    for path in image_files:
        file_size = path.stat().st_size
        sizes.append(file_size)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(path)
        try:
            metadata = inspect_image(path)
        except Exception as error:  # Pillow raises several format-specific exceptions.
            invalid_files.append(
                {
                    "path": relative_path(path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        valid_files.append(path)
        resolutions[metadata["resolution"]] += 1
        channels[str(metadata["channels"])] += 1
        modes[metadata["mode"]] += 1
        if len(examples) < 10:
            examples.append({"path": relative_path(path), "file_size_bytes": file_size, **metadata})

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups)
    directory_counts = Counter(path.parent.relative_to(IMAGE_ROOT).as_posix() for path in image_files)
    class_counts = Counter(path.parent.name for path in image_files)
    label_directories = sorted(
        path.name
        for path in IMAGE_ROOT.rglob("*")
        if path.is_dir() and any(child.parent == path for child in image_files)
    )
    numeric_stems = [path.stem for path in image_files if re.fullmatch(r"\d+", path.stem)]
    non_numeric_stems = [path.stem for path in image_files if path.stem not in numeric_stems]

    report: dict[str, Any] = {
        "dataset_root": str(IMAGE_ROOT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_files_discovered": len(all_files),
        "total_image_files": len(image_files),
        "valid_files": len(valid_files),
        "invalid_files": invalid_files,
        "non_image_files": [relative_path(path) for path in non_image_files],
        "extensions": dict(sorted(extension_counts.items())),
        "resolutions": dict(resolutions.most_common()),
        "channels": dict(channels.most_common()),
        "modes": dict(modes.most_common()),
        "file_sizes_bytes": {
            "minimum": min(sizes) if sizes else None,
            "maximum": max(sizes) if sizes else None,
            "mean": statistics.mean(sizes) if sizes else None,
            "median": statistics.median(sizes) if sizes else None,
            "p05": percentile(sizes, 0.05),
            "p95": percentile(sizes, 0.95),
        },
        "directory_structure": {
            "directory_count": sum(1 for path in IMAGE_ROOT.rglob("*") if path.is_dir()),
            "directories_with_images": dict(sorted(directory_counts.items())),
            "label_directories": label_directories,
            "representative_files": [relative_path(path) for path in image_files[:10]],
        },
        "label_information": {
            "status": "available_from_directory_names",
            "label_source": "immediate parent directory of each image",
            "classes": label_directories,
            "class_counts": {label: class_counts.get(label, 0) for label in label_directories},
            "caveat": "Directory names are recorded as labels; their semantic provenance was not independently verified.",
        },
        "actor_subject_information": {
            "actor_id": "not_reliably_encoded",
            "subject_id": "not_reliably_encoded",
            "session_id": "not_reliably_encoded",
            "sequence_id": "not_reliably_encoded",
            "frame_id": "not_reliably_encoded",
            "filename_pattern": "numeric stems only" if not non_numeric_stems else "mixed stems",
            "numeric_filename_count": len(numeric_stems),
            "note": "Numeric filenames are preserved as file identifiers but are not assigned actor, subject, session, sequence, or frame meaning.",
        },
        "duplicate_information": {
            "method": "SHA-256 hash of raw file bytes",
            "total_files_hashed": len(image_files),
            "unique_files": len(hashes),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_files_beyond_first_copy": duplicate_files,
            "representative_groups": [
                [relative_path(path) for path in paths[:10]] for paths in duplicate_groups[:10]
            ],
        },
        "visual_quality": {
            "common_resolution": resolutions.most_common(1)[0][0] if resolutions else None,
            "common_channels": channels.most_common(1)[0][0] if channels else None,
            "alpha_channel_present": any(mode in {"LA", "RGBA", "PA"} for mode in modes),
            "obvious_preprocessing_observed": "Images are stored as rasterized 48x48 single-channel grayscale PNGs; transformation history is unknown.",
            "representative_examples": examples,
        },
        "mapping_status": {
            "tabular_mapping": "not_verified",
            "statement": "The image dataset is treated as an independent modality.",
            "reason": "mental_health_multimodal.csv has no explicit image-reference or sample-level image key.",
        },
        "warnings": [],
    }
    if duplicate_groups:
        report["warnings"].append("Exact-content duplicate files were detected; do not split duplicates across partitions.")
    if invalid_files:
        report["warnings"].append("Unreadable image files were detected.")
    if non_image_files:
        report["warnings"].append("Non-image files were found under the dataset root and excluded from image inspection.")
    report["warnings"].append("No actor/subject/session/frame key was verified; future subject-independent splitting is not currently possible from these paths alone.")
    return report


def main() -> None:
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Image dataset directory not found: {IMAGE_ROOT}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("IMAGE DATASET INSPECTION COMPLETE")
    print(f"Total images: {report['total_image_files']}")
    print(f"Valid images: {report['valid_files']}")
    print(f"Invalid images: {len(report['invalid_files'])}")
    print(f"Image formats: {report['extensions']}")
    print(f"Resolution summary: {report['resolutions']}")
    print(f"Channels: {report['channels']}")
    print(f"Label classes: {report['label_information']['classes']}")
    print("Actor/subject count: unavailable; no reliable actor or subject identifiers found")
    print(
        "Duplicate findings: "
        f"{report['duplicate_information']['duplicate_groups']} groups, "
        f"{report['duplicate_information']['duplicate_files_beyond_first_copy']} duplicate files"
    )
    print(f"Tabular mapping status: {report['mapping_status']['statement']}")
    print(f"Output report: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print("Tabular baseline untouched")
    print("Audio model untouched")
    print("No image model trained")


if __name__ == "__main__":
    main()