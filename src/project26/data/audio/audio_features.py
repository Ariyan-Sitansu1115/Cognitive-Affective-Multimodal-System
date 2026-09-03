from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
N_MFCC = 13
EMOTION_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
FILENAME_PATTERN = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)


FEATURE_NAMES = [
    *(f"mfcc_{index:02d}_mean" for index in range(1, N_MFCC + 1)),
    *(f"mfcc_{index:02d}_variance" for index in range(1, N_MFCC + 1)),
    "spectral_centroid_mean_hz", "spectral_bandwidth_mean_hz",
    "spectral_rolloff_mean_hz", "zero_crossing_rate_mean", "rms_mean",
    "rms_variance", "f0_mean_hz", "duration_sec",
]


def _parse_filename(path: Path) -> dict[str, str] | None:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None or match.group("emotion") not in EMOTION_LABELS:
        return None
    values = match.groupdict()
    return {"actor_id": values["actor"], "label": EMOTION_LABELS[values["emotion"]]}


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio is empty or has an unsupported shape")
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    return audio, sample_rate


def _frames(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    frame_count = 1 + max(0, (audio.size - frame_length) // hop_length)
    end = (frame_count - 1) * hop_length + frame_length
    if end > audio.size:
        audio = np.pad(audio, (0, end - audio.size))
    starts = np.arange(frame_count)[:, None] * hop_length
    return audio[starts + np.arange(frame_length)[None, :]]


def _mel_filterbank(sample_rate: int, n_fft: int, n_filters: int = 26) -> np.ndarray:
    hz_to_mel = lambda frequency: 2595.0 * np.log10(1.0 + frequency / 700.0)
    mel_to_hz = lambda value: 700.0 * (10.0 ** (value / 2595.0) - 1.0)
    frequencies = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), n_filters + 2))
    bins = np.floor((n_fft + 1) * frequencies / sample_rate).astype(int)
    filters = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_filters):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = np.arange(left, center) / (center - left)
        if right > center:
            filters[index, center:right] = np.arange(right - center, 0, -1) / (right - center)
    return filters


def _pitch_mean(audio: np.ndarray, sample_rate: int) -> float:
    frame_length = 2048
    frames = _frames(audio, frame_length=frame_length, hop_length=512)
    window = np.hanning(frame_length)
    minimum_lag = int(sample_rate / 500.0)
    maximum_lag_limit = int(sample_rate / 50.0)
    pitches = []
    for frame in frames:
        centered = (frame - frame.mean()) * window
        autocorrelation = np.correlate(centered, centered, mode="full")[frame_length - 1:]
        zero_lag = autocorrelation[0]
        if zero_lag <= 1e-8:
            continue
        maximum_lag = min(maximum_lag_limit, len(autocorrelation) - 1)
        section = autocorrelation[minimum_lag:maximum_lag + 1]
        if section.size == 0:
            continue
        lag = minimum_lag + int(np.argmax(section))
        if autocorrelation[lag] / zero_lag >= 0.30:
            pitches.append(sample_rate / lag)
    return float(np.mean(pitches)) if len(pitches) >= 3 else float("nan")


def extract_features(path: Path) -> dict[str, float]:
    audio, sample_rate = _read_audio(path)
    frames = _frames(audio)
    windowed = frames * np.hanning(frames.shape[1])
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    power = (spectrum ** 2) / frames.shape[1]
    frequencies = np.fft.rfftfreq(frames.shape[1], 1.0 / sample_rate)
    mel_energy = np.maximum(power @ _mel_filterbank(sample_rate, frames.shape[1]).T, 1e-10)
    mfcc = dct(np.log(mel_energy), type=2, axis=1, norm="ortho")[:, :N_MFCC]
    magnitude_sum = np.maximum(spectrum.sum(axis=1), 1e-10)
    centroid = (spectrum * frequencies).sum(axis=1) / magnitude_sum
    bandwidth = np.sqrt((spectrum * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1) / magnitude_sum)
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff = frequencies[(cumulative >= cumulative[:, -1, None] * 0.85).argmax(axis=1)]
    zcr = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    features = {
        **{f"mfcc_{index:02d}_mean": float(value) for index, value in enumerate(mfcc.mean(axis=0), 1)},
        **{f"mfcc_{index:02d}_variance": float(value) for index, value in enumerate(mfcc.var(axis=0), 1)},
        "spectral_centroid_mean_hz": float(centroid.mean()),
        "spectral_bandwidth_mean_hz": float(bandwidth.mean()),
        "spectral_rolloff_mean_hz": float(rolloff.mean()),
        "zero_crossing_rate_mean": float(zcr.mean()),
        "rms_mean": float(rms.mean()), "rms_variance": float(rms.var()),
        "f0_mean_hz": _pitch_mean(audio, sample_rate),
        "duration_sec": float(audio.size / sample_rate),
    }
    return features


def _canonical_files(root: Path, discovered: list[Path]) -> tuple[list[Path], int]:
    canonical = sorted(path for path in discovered if len(path.relative_to(root).parts) == 2 and re.fullmatch(r"Actor_\d{2}", path.parent.name))
    return (canonical, len(discovered) - len(canonical)) if canonical else (discovered, 0)


def extract_audio_dataset(input_dir: Path, output_csv: Path, report_path: Path) -> dict:
    start = time.perf_counter()
    input_dir = Path(input_dir)
    discovered = sorted(path for path in input_dir.rglob("*") if path.is_file())
    audio_files = [path for path in discovered if path.suffix.lower() == ".wav"]
    selected, duplicate_files_skipped = _canonical_files(input_dir, audio_files)
    rows, failures, invalid_labels, content_hashes = [], [], [], set()
    audio_read_warnings = set()
    for path in selected:
        metadata = _parse_filename(path)
        if metadata is None:
            invalid_labels.append(path.relative_to(input_dir).as_posix())
            continue
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                features = extract_features(path)
            audio_read_warnings.update(str(item.message) for item in caught_warnings)
            if not all(np.isfinite(value) for value in features.values()):
                raise ValueError("one or more extracted features are non-finite")
            content_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append({"file_path": path.relative_to(input_dir).as_posix(), "actor_id": int(metadata["actor_id"]), "label": metadata["label"], **features})
        except Exception as error:
            failures.append({"file_path": path.relative_to(input_dir).as_posix(), "error": f"{type(error).__name__}: {error}"})
    dataframe = pd.DataFrame(rows, columns=["file_path", "actor_id", "label", *FEATURE_NAMES])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    class_counts = {label: int(count) for label, count in sorted(Counter(dataframe["label"]).items())}
    actor_counts = {str(actor): int(count) for actor, count in sorted(Counter(dataframe["actor_id"]).items())}
    report = {
        "input_directory": input_dir.as_posix(), "output_csv": output_csv.as_posix(),
        "discovered_files": len(discovered), "discovered_wav_files": len(audio_files),
        "duplicate_tree_files_skipped": duplicate_files_skipped, "selected_canonical_files": len(selected),
        "processed_files": len(dataframe), "failed_files": len(failures), "invalid_label_files": len(invalid_labels),
        "failures": failures, "invalid_label_paths": invalid_labels, "sample_rate_hz": TARGET_SAMPLE_RATE,
        "audio_read_warnings": sorted(audio_read_warnings),
        "feature_names": FEATURE_NAMES, "feature_column_count": len(FEATURE_NAMES),
        "label_classes": sorted(class_counts), "class_distribution": class_counts,
        "actor_count": int(dataframe["actor_id"].nunique()) if not dataframe.empty else 0,
        "actor_distribution": actor_counts, "unique_content_hashes_processed": len(content_hashes),
        "speech_rate_extracted": False, "speech_rate_warning": "Speech rate was not extracted; no dependable estimator is configured.",
        "pitch_missing_values": int(dataframe["f0_mean_hz"].isna().sum()) if not dataframe.empty else 0,
        "warnings": ["The nested audio_speech_actors_01-24 tree was excluded because it duplicates the top-level Actor_* tree.", "Stereo files are downmixed to mono before feature extraction."],
        "processing_time_sec": round(time.perf_counter() - start, 3),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report