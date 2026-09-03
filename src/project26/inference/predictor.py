from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from project26.config import CONFIG
from project26.data.inspect import inspect_data
from project26.data.load_raw import load_raw_data
from project26.data.preprocessing import fit_transform_features, prepare_features
from project26.models.temporal_fusion import TemporalFusionTabular
from project26.runtime import configure_runtime
from project26.training.checkpoint import load_checkpoint
from project26.uncertainty.entropy import calculate_uncertainty


class InferencePredictor:
    """Predict one new sample with the frozen neural checkpoint."""

    def __init__(self, root: Path | None = None, device: torch.device | None = None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.device = device if device is not None else configure_runtime()
        self.checkpoint_path = self.root / CONFIG["results_dir"] / "best_hybrid.pt"

        csv_files = [str(path) for path in (self.root / CONFIG["data_raw_dir"]).rglob("*.csv")]
        _, dataframe = load_raw_data(csv_files)
        metadata = inspect_data(dataframe)
        feature_df, numeric_cols, categorical_cols = prepare_features(dataframe, metadata)
        self.metadata = metadata
        self.feature_names = list(numeric_cols)
        self._validate_feature_contract(categorical_cols)

        train_df, _ = train_test_split(
            feature_df,
            train_size=CONFIG["split_ratios"]["train"],
            stratify=feature_df[metadata["target"]],
            random_state=42,
        )
        train_df = train_df.copy()
        self._train_df = train_df.copy()
        placeholder = pd.DataFrame(
            [{name: np.nan for name in self.feature_names} | {metadata["target"]: 0}]
        )
        self._train_X, _, self._sample_template = fit_transform_features(
            train_df,
            train_df.copy(),
            placeholder,
            numeric_cols,
            categorical_cols,
            metadata["target"],
        )

        model = TemporalFusionTabular(
            input_dim=self._train_X.shape[1],
            output_dim=metadata["n_classes"],
            hidden_dim=CONFIG["transformer_hidden"],
        )
        self.model = load_checkpoint(model, self.checkpoint_path, self.device)

    def _validate_feature_contract(self, categorical_cols):
        if categorical_cols:
            raise ValueError(
                "The frozen baseline includes categorical features, but this inference "
                "interface accepts the documented 21 numerical features only."
            )
        if len(self.feature_names) != 21:
            raise ValueError(
                f"Expected 21 numerical features from the baseline, found {len(self.feature_names)}."
            )

    def _build_sample(self, features: Mapping[str, float]) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing required feature names: {missing}")

        values = {name: features[name] for name in self.feature_names}
        sample = pd.DataFrame([values])
        for name in self.feature_names:
            sample[name] = pd.to_numeric(sample[name], errors="coerce")
            if sample[name].isna().any() and values[name] is not None and not pd.isna(values[name]):
                raise ValueError(f"Feature {name!r} must be numeric.")
            if np.isinf(sample[name].fillna(0)).any():
                raise ValueError(f"Feature {name!r} must be finite.")
        sample[self.metadata["target"]] = 0
        return sample

    def predict(self, features: Mapping[str, float]) -> dict:
        sample_df = self._build_sample(features)
        _, _, sample_X = fit_transform_features(
            self._train_df.copy(),
            self._train_df.copy(),
            sample_df,
            self.feature_names,
            [],
            self.metadata["target"],
        )
        tensor = torch.from_numpy(sample_X.to_numpy(dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        uncertainty, confidence, predicted_indices = calculate_uncertainty(probabilities)
        index = int(predicted_indices[0])
        class_names = self.metadata["class_names"]
        return {
            "predicted_state": class_names[index].replace("_", " "),
            "confidence": float(confidence[0]),
            "entropy_uncertainty": float(uncertainty[0]),
            "normalized_uncertainty": float(uncertainty[0] / np.log(len(class_names))),
            "class_probabilities": {
                name.replace("_", " "): float(probability)
                for name, probability in zip(class_names, probabilities[0])
            },
        }