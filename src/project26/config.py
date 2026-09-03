import os

from .runtime import SEED

CONFIG = {
    "project_no": "26",
    "project_name": "Cognitive-Affective_Multimodal_Support",
    "team_no": "7",
    "task_type": "classification",
    "kaggle_dataset_slug": "zara2099/multimodal-mental-health-assessment-dataset",
    "dataset_source": "Multimodal mental health assessment dataset (see compatibility note)",
    "target_column": None,
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "random_seed": SEED,
    "transformer_hidden": 64,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-3,
    "early_stop_patience": 6,
    "data_raw_dir": "data/26/raw",
    "data_processed_dir": "data/26/processed",
    "figures_dir": "data/26/figures",
    "results_dir": "data/26/results",
    "reports_dir": "data/26/reports",
}


def create_output_directories():
    for directory in [CONFIG["data_raw_dir"], CONFIG["data_processed_dir"], CONFIG["figures_dir"], CONFIG["results_dir"], CONFIG["reports_dir"]]:
        os.makedirs(directory, exist_ok=True)
