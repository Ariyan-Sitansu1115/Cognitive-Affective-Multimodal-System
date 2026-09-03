from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.data.audio import extract_audio_dataset


def main() -> None:
    report = extract_audio_dataset(ROOT / "data/26/raw/Audios", ROOT / "data/26/processed/audio/audio_features.csv", ROOT / "data/26/reports/audio_feature_extraction.json")
    print("AUDIO FEATURE EXTRACTION COMPLETE")
    print("Files discovered:", report["discovered_wav_files"])
    print("Files processed:", report["processed_files"])
    print("Files failed:", report["failed_files"])
    print("Feature count:", report["feature_column_count"])
    print("Label classes:", ", ".join(report["label_classes"]))
    print("Class distribution:", report["class_distribution"])
    print("Actor count:", report["actor_count"])
    print("Feature table:", report["output_csv"])
    print("Report:", ROOT / "data/26/reports/audio_feature_extraction.json")
    print("Warnings:", report["warnings"] + report["audio_read_warnings"] + [report["speech_rate_warning"]])
    print("Tabular baseline modified: no")


if __name__ == "__main__":
    main()