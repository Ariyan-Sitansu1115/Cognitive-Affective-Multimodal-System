"""Audit explicit metadata for defensible cross-modal sample alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.config import CONFIG


def columns_matching(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in tokens)]


def identifier_columns(columns: list[str]) -> list[str]:
    names = {
        "subject_id", "participant_id", "patient_id", "user_id", "actor_id",
        "session_id", "recording_id", "sample_id", "file_name", "filename",
        "file_path", "timestamp",
    }
    return [column for column in columns if column.lower() in names]


def main() -> None:
    tabular_path = ROOT / "data/26/raw/mental_health_multimodal.csv"
    audio_path = ROOT / "data/26/processed/audio/audio_features.csv"
    image_root = ROOT / "data/26/raw/Extracted_images"
    tabular = pd.read_csv(tabular_path, nrows=5)
    audio = pd.read_csv(audio_path, nrows=5)
    image_paths = sorted(path.relative_to(ROOT).as_posix() for path in image_root.rglob("*.png"))
    tabular_columns = tabular.columns.tolist()
    audio_columns = audio.columns.tolist()
    subject_tokens = ("subject", "participant", "patient", "user", "actor", "session", "recording", "filename", "file_path", "timestamp")
    modality_tokens = ("audio", "wav", "image", "face", "photo")
    evidence = {
        "tabular": {
            "path": tabular_path.relative_to(ROOT).as_posix(),
            "rows": int(sum(1 for _ in open(tabular_path, encoding="utf-8")) - 1),
            "columns": tabular_columns,
            "identifier_like_columns": identifier_columns(tabular_columns),
            "modality_reference_columns": columns_matching(tabular_columns, modality_tokens),
        },
        "audio": {
            "path": audio_path.relative_to(ROOT).as_posix(),
            "columns": audio_columns,
            "identifier_like_columns": identifier_columns(audio_columns),
            "actor_id_values_observed": sorted(pd.read_csv(audio_path, usecols=["actor_id"])["actor_id"].astype(str).unique().tolist()),
            "file_path_example": audio.iloc[0]["file_path"] if "file_path" in audio else None,
        },
        "image": {
            "root": image_root.relative_to(ROOT).as_posix(),
            "files": len(image_paths),
            "directory_names": sorted({Path(path).parent.name for path in image_paths}),
            "path_examples": image_paths[:5],
            "identifier_like_metadata": [],
        },
        "explicit_relationship_evidence": [],
        "negative_evidence": [
            "The tabular CSV contains no subject, participant, actor, session, recording, filename, timestamp, audio, or image reference column.",
            "Audio actor_id is local to the audio corpus and has no verified field or metadata relationship to tabular rows or image files.",
            "Image paths contain emotion directory names and numeric filenames, but no verified shared participant or session identifier.",
            "Different row counts, ordering, filenames, and labels were not used as correspondence evidence.",
            "Emotion labels are not mapped to mental-health classes because no validated mapping is supplied.",
        ],
        "classification": "C. Independent/unpaired modality datasets",
        "decision": "No scientifically defensible sample-level pairing is available in the supplied dataset.",
        "dataset_source": CONFIG["kaggle_dataset_slug"],
        "paired": False,
    }
    output = ROOT / "data/26/reports/multimodal_alignment_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()