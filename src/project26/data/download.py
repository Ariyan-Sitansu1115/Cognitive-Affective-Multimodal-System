import os
import subprocess
from pathlib import Path

import pandas as pd


def download_dataset(config):
    raw_csvs = list(Path(config["data_raw_dir"]).rglob("*.csv"))
    if raw_csvs:
        print(f"Dataset already present; skipping download: {raw_csvs[0]}")
    else:
        try:
            from google.colab import files
            if not os.path.exists("/root/.kaggle/kaggle.json"):
                print("Upload your kaggle.json:")
                uploaded = files.upload()
                os.makedirs("/root/.kaggle", exist_ok=True)
                for fname in uploaded:
                    if fname.endswith(".json"):
                        os.replace(fname, "/root/.kaggle/kaggle.json")
                os.chmod("/root/.kaggle/kaggle.json", 0o600)
        except ModuleNotFoundError:
            print("Running outside Colab; using configured Kaggle credentials.")
        subprocess.run(["kaggle", "datasets", "download", "-d", config["kaggle_dataset_slug"], "-p", config["data_raw_dir"], "--unzip"], check=True)

    raw_files = []
    for root, _, fnames in os.walk(config["data_raw_dir"]):
        for fname in fnames:
            raw_files.append(os.path.join(root, fname))
    print(f"{len(raw_files)} files found")
    assert len(raw_files) > 0, "No files found - check dataset download step before continuing."
    ext_counts = pd.Series([os.path.splitext(path)[1].lower() for path in raw_files]).value_counts()
    print("File extensions found:\n", ext_counts)
    has_audio = any(ext in (".wav", ".mp3", ".flac") for ext in ext_counts.index)
    has_images = any(ext in (".png", ".jpg", ".jpeg") for ext in ext_counts.index)
    csv_files = [path for path in raw_files if path.lower().endswith(".csv")]
    print(f"\nRESOLUTION: HAS_AUDIO={has_audio}, HAS_IMAGES={has_images}, CSV files={len(csv_files)}")
    if not has_audio and not has_images:
        print("This download is tabular/text-only. Audio Transformer and Visual ViT branches will NOT be fabricated - falling back to the documented tabular-only path.")
    return raw_files, csv_files, has_audio, has_images
