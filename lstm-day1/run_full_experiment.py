"""Train and compare PatchCore and CNN+LSTM on MVTec AD leather."""

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageEnhance
from sklearn.metrics import roc_auc_score
from torch import nn

from run_experiment import (
    BRIGHTNESS_FACTORS,
    SEED,
    WEIGHTS,
    PatchFeatures,
    Sample,
    brightness_versions,
    build_memory_bank,
    load_image,
    make_split,
    score_images,
    write_csv,
)


CONDITIONS = {"original": 1.00, "dark": 0.75, "bright": 1.25}
MODELS = ("PatchCore", "CNN+LSTM")
EPOCHS = 10
PATCH_BATCH_SIZE = 1024
IMAGE_BATCH_SIZE = 8
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001


class BrightnessLSTM(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, HIDDEN_SIZE, batch_first=True)
        self.output = nn.Linear(HIDDEN_SIZE, feature_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.output(sequence[:, -1])


def prepare_image(data_dir: Path, sample: Sample, condition: float):
    image = load_image(data_dir / sample.relative_path)
    return ImageEnhance.Brightness(image).enhance(condition) if condition != 1 else image


def sequence_features(
    data_dir: Path,
    samples: list[Sample],
    condition: float,
    extractor: PatchFeatures,
    cache_dir: Path,
    key: str,
) -> torch.Tensor:
    # Changes to the encoder, preprocessing, split or images invalidate the cache.
    source = Path(__file__)
    encoder_source = source.with_name("run_experiment.py")
    digest = hashlib.sha256(encoder_source.read_bytes() + source.read_bytes())
    digest.update(repr((data_dir.resolve(), WEIGHTS.name, condition, BRIGHTNESS_FACTORS,
                        extractor.device.type, torch.__version__)).encode())
    for sample in samples:
        metadata = (data_dir / sample.relative_path).stat()
        digest.update(repr((sample.relative_path.as_posix(), metadata.st_size,
                            metadata.st_mtime_ns)).encode())
    cache_file = cache_dir / f"{key}-{digest.hexdigest()[:16]}.pt"
    if cache_file.is_file():
        print(f"Loading {key} features", flush=True)
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    print(f"Extracting {key} features for {len(samples)} images", flush=True)
    features = None
    for start in range(0, len(samples), IMAGE_BATCH_SIZE):
        batch = samples[start : start + IMAGE_BATCH_SIZE]
        images = [
            version
            for sample in batch
            for version in brightness_versions(prepare_image(data_dir, sample, condition))
        ]
        encoded = extractor(images).cpu()
        height, width, channels = encoded.shape[1:]
        encoded = encoded.reshape(len(batch), len(BRIGHTNESS_FACTORS), height, width, channels)
        encoded = encoded.permute(0, 2, 3, 1, 4)
        if features is None:
            features = torch.empty((len(samples), height, width, len(BRIGHTNESS_FACTORS), channels))
        features[start : start + len(batch)] = encoded

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as file:
        pending = Path(file.name)
    try:
        torch.save(features, pending)
        pending.replace(cache_file)
    finally:
        pending.unlink(missing_ok=True)
    return features


def patch_errors(model: BrightnessLSTM, patches: torch.Tensor) -> torch.Tensor:
    predictions = model(patches[:, :-1].contiguous())
    return (predictions - patches[:, -1]).square().mean(dim=1)


@torch.inference_mode()
def validate_lstm(
    model: BrightnessLSTM, features: torch.Tensor, device: torch.device
) -> tuple[float, list[float]]:
    model.eval()
    scores = []
    total_error = 0.0
    width = features.shape[-1]
    for image in features:
        patches = image.reshape(-1, len(BRIGHTNESS_FACTORS), width).to(device)
        errors = patch_errors(model, patches)
        total_error += errors.sum().item()
        scores.append(errors.max().item())
    return total_error / features.shape[:3].numel(), scores


def train_lstm(
    train_features: torch.Tensor, val_features: torch.Tensor, device: torch.device
) -> tuple[BrightnessLSTM, list[tuple[int, float, float]], int]:
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    feature_size = train_features.shape[-1]
    model = BrightnessLSTM(feature_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    patches = train_features.reshape(-1, len(BRIGHTNESS_FACTORS), feature_size)
    history = []
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for batch_indices in torch.randperm(len(patches), generator=generator).split(PATCH_BATCH_SIZE):
            batch = patches[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            errors = patch_errors(model, batch)
            errors.mean().backward()
            optimizer.step()
            total_loss += errors.detach().sum()
        train_loss = total_loss.item() / len(patches)
        val_loss, _ = validate_lstm(model, val_features, device)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f"Non-finite LSTM loss at epoch {epoch}")
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch:2d}/{EPOCHS}: train MSE {train_loss:.6f}, val MSE {val_loss:.6f}", flush=True)
    model.load_state_dict(best_weights)
    return model, history, best_epoch


def evaluate_condition(
    test: list[Sample], scores: list[float], condition: str, model: str, threshold: float
) -> tuple[tuple, list[tuple]]:
    labels = np.array([sample.label for sample in test])
    scores = np.array(scores, dtype=np.float64)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError(f"Invalid {condition} scores for {model}")
    predictions = (scores > threshold).astype(int)
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    good = int(np.sum(labels == 0))
    defective = int(np.sum(labels == 1))
    metrics = (condition, model, threshold, roc_auc_score(labels, scores),
               fp, good, fp / good, fn, defective, fn / defective)
    image_rows = [
        (condition, model, sample.relative_path.as_posix(), sample.defect_type, sample.label,
         float(score), int(prediction))
        for sample, score, prediction in zip(test, scores, predictions, strict=True)
    ]
    return metrics, image_rows


def plot_comparison(rows: list[tuple], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), layout="constrained")
    colors = ("#216c8d", "#a64c22")
    x = np.arange(len(CONDITIONS))
    for axis, (title, column) in zip(
        axes, (("Image AUROC", 3), ("False positive rate", 6), ("False negative rate", 9)), strict=True
    ):
        for i, (model, color) in enumerate(zip(MODELS, colors, strict=True)):
            values = [row[column] for row in rows if row[1] == model]
            axis.bar(x + (i - 0.5) * 0.34, values, width=0.34, label=model, color=color)
        axis.set(title=title, ylim=(0, 1.05), xticks=x, xticklabels=("Original", "Dark", "Bright"))
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_errors(
    rows: list[tuple], data_dir: Path, thresholds: dict[str, float], path: Path
) -> int:
    matches = {}
    for row in rows:
        condition, model, _, _, label, score, prediction = row
        if label == prediction:
            continue
        key = (model, "False alarm" if label == 0 else "Miss")
        threshold = thresholds[model]
        if key not in matches or abs(score - threshold) > abs(matches[key][5] - threshold):
            matches[key] = row

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 7.3), layout="constrained")
    for row_index, model in enumerate(MODELS):
        for col_index, error_type in enumerate(("False alarm", "Miss")):
            axis = axes[row_index, col_index]
            axis.set_axis_off()
            row = matches.get((model, error_type))
            if row is None:
                axis.text(0.5, 0.5, f"{model}\nNo {error_type.lower()} in test set",
                          ha="center", va="center")
                continue
            condition, _, relative_path, defect_type, _, score, _ = row
            sample = Sample(Path(relative_path), "test", defect_type)
            axis.imshow(prepare_image(data_dir, sample, CONDITIONS[condition]))
            axis.set_title(f"{model} | {error_type} | {condition}\n{relative_path}  (score {score:.4g})",
                           fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return len(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")
    start = time.perf_counter()
    device = torch.device(args.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    samples = make_split(args.data_dir)
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    extractor = PatchFeatures(device)
    train_features = sequence_features(args.data_dir, train, 1.0, extractor, args.cache_dir, "train")
    val_features = sequence_features(args.data_dir, validation, 1.0, extractor, args.cache_dir, "validation")

    train_original = train_features[..., -1, :].flatten(0, 2)
    memory_bank = build_memory_bank(train_original).to(device)
    del train_original
    print(f"PatchCore memory bank: {list(memory_bank.shape)}", flush=True)
    model, history, best_epoch = train_lstm(train_features, val_features, device)
    del train_features
    val_patchcore = score_images(val_features[..., -1, :].to(device), memory_bank)
    _, val_lstm = validate_lstm(model, val_features, device)
    del val_features
    thresholds = {
        name: float(np.percentile(values, 95))
        for name, values in zip(MODELS, (val_patchcore, val_lstm), strict=True)
    }
    print(f"Best LSTM epoch: {best_epoch}. Thresholds: {thresholds}", flush=True)
    metrics = []
    image_results = []
    for condition, factor in CONDITIONS.items():
        features = sequence_features(args.data_dir, test, factor, extractor, args.cache_dir, condition)
        patchcore = score_images(features[..., -1, :].to(device), memory_bank)
        _, lstm = validate_lstm(model, features, device)
        for name, scores in zip(MODELS, (patchcore, lstm), strict=True):
            metric, rows = evaluate_condition(test, scores, condition, name, thresholds[name])
            metrics.append(metric)
            image_results.extend(rows)
            print(f"{condition} {name}: AUROC {metric[3]:.4f}, "
                  f"FP {metric[4]}/{metric[5]}, FN {metric[7]}/{metric[8]}", flush=True)
        del features

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", [
        "condition", "model", "threshold", "image_auroc", "false_positives", "normal_count",
        "false_positive_rate", "false_negatives", "defect_count", "false_negative_rate",
    ], metrics)
    write_csv(out / "image_scores.csv", [
        "condition", "model", "relative_path", "defect_type", "label", "anomaly_score", "prediction",
    ], image_results)
    write_csv(out / "training_history.csv", ["epoch", "train_mse", "validation_mse"], history)
    write_csv(out / "validation_scores.csv", ["relative_path", "patchcore_score", "lstm_score"], [
        (sample.relative_path.as_posix(), patchcore, lstm)
        for sample, patchcore, lstm in zip(validation, val_patchcore, val_lstm, strict=True)
    ])
    plot_comparison(metrics, out / "comparison.png")
    error_count = plot_errors(image_results, args.data_dir, thresholds, out / "error_examples.png")
    torch.save({
        "lstm": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "memory_bank": memory_bank.cpu(),
        "thresholds": thresholds,
        "feature_size": memory_bank.shape[1],
        "hidden_size": HIDDEN_SIZE,
    }, out / "model_state.pt")
    summary = {
        "best_epoch": best_epoch,
        "thresholds": thresholds,
        "memory_bank_shape": list(memory_bank.shape),
        "data_counts": dict(Counter(sample.split for sample in samples)),
        "test_conditions": CONDITIONS,
        "device": args.device,
        "seed": SEED,
        "seconds": round(time.perf_counter() - start, 2),
        "error_examples": error_count,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
