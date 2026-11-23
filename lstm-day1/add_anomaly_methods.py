"""Fit and evaluate Isolation Forest and a local-feature CutPaste variant."""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from torch import nn

from run_experiment import SEED, PatchFeatures, load_image, make_split, write_csv
from run_full_experiment import CONDITIONS, evaluate_condition, plt, prepare_image

NEW_MODELS = ("Isolation Forest", "CutPaste")


class CutPasteHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Linear(32, 3)

    def forward(self, features):
        return self.classifier(self.embedding(features))


class ExtraMethods:
    def __init__(self, directory):
        self.forest = joblib.load(directory / "isolation_forest.joblib")
        state = torch.load(directory / "cutpaste.pt", map_location="cpu", weights_only=True)
        self.thresholds = state["thresholds"]
        self.cutpaste = CutPasteHead().eval()
        self.cutpaste.load_state_dict(state["head"])
        self.mean = state["mean"]
        self.precision = state["precision"]

    @torch.inference_mode()
    def grids(self, features):
        shape = features.shape[:-1]
        patches = features.reshape(-1, features.shape[-1])
        forest = torch.from_numpy(-self.forest.score_samples(patches.numpy())).float()
        delta = self.cutpaste.embedding(patches) - self.mean
        cutpaste = ((delta @ self.precision) * delta).sum(dim=1).clamp_min(0).sqrt()
        return dict(zip(NEW_MODELS, (forest.reshape(shape), cutpaste.reshape(shape)), strict=True))


def make_paste(image, rng, *, scar):
    width, height = image.size
    pw = int(width * rng.uniform(0.025, 0.05) if scar else width * rng.uniform(0.11, 0.22))
    ph = int(height * rng.uniform(0.18, 0.33) if scar else height * rng.uniform(0.11, 0.22))
    x0 = int(rng.integers(0, width - pw + 1))
    y0 = int(rng.integers(0, height - ph + 1))
    x1 = int(rng.integers(0, width - pw + 1))
    y1 = int(rng.integers(0, height - ph + 1))
    patch = image.crop((x0, y0, x0 + pw, y0 + ph))
    patch = ImageEnhance.Brightness(patch).enhance(float(rng.uniform(0.78, 1.22)))
    pasted = image.copy()
    pasted.paste(patch, (x1, y1))
    return pasted, ((x1 + pw / 2) / width, (y1 + ph / 2) / height)


def selected_patches(grid, center):
    height, width = grid.shape[:2]
    x, y = int(center[0] * width), int(center[1] * height)
    return grid[max(0, y-1):y+2, max(0, x-1):x+2].reshape(-1, grid.shape[-1])


def training_features(data, samples, extractor):
    rng = np.random.default_rng(SEED)
    classes = [[], [], []]
    forest_samples = []
    for sample in samples:
        image = load_image(data / sample.relative_path)
        box, box_center = make_paste(image, rng, scar=False)
        scar, scar_center = make_paste(image, rng, scar=True)
        clean, box_grid, scar_grid = extractor([image, box, scar])
        flat = clean.flatten(0, 1)
        forest_samples.append(flat[rng.choice(len(flat), 64, replace=False)])
        classes[0].append(flat[rng.choice(len(flat), 9, replace=False)])
        classes[1].append(selected_patches(box_grid, box_center))
        classes[2].append(selected_patches(scar_grid, scar_center))
    return torch.cat(forest_samples), [torch.cat(patches) for patches in classes]


def fit_cutpaste(clean, classes):
    torch.manual_seed(SEED)
    head = CutPasteHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    features = torch.cat(classes)
    labels = torch.cat([torch.full((len(group),), i) for i, group in enumerate(classes)])
    for _ in range(20):
        for indices in torch.randperm(len(features)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        distribution = LedoitWolf().fit(head.embedding(clean).numpy())
    return {
        "head": head.state_dict(),
        "mean": torch.from_numpy(distribution.location_).float(),
        "precision": torch.from_numpy(distribution.precision_).float(),
        "thresholds": {},
    }


def score_samples(data, samples, factor, extractor, methods):
    all_scores = {name: [] for name in NEW_MODELS}
    for start in range(0, len(samples), 12):
        images = [prepare_image(data, sample, factor) for sample in samples[start:start+12]]
        grids = methods.grids(extractor(images))
        for name, grid in grids.items():
            all_scores[name].extend(grid.flatten(1).amax(dim=1).tolist())
    return all_scores


def write_comparison(directory):
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = {(row["condition"], row["model"]): row for row in csv.DictReader(stream)}
    models = ("PatchCore", "CNN+LSTM", *NEW_MODELS)
    colors = ("#256b92", "#b34e24", "#6a4e99", "#228264")
    x = np.arange(len(CONDITIONS))
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for axis, (label, field) in zip(axes, (
        ("Image AUROC", "image_auroc"),
        ("False positive rate", "false_positive_rate"),
        ("False negative rate", "false_negative_rate"),
    ), strict=True):
        for index, (name, color) in enumerate(zip(models, colors, strict=True)):
            values = [float(rows[condition, name][field]) for condition in CONDITIONS]
            axis.bar(x + (index - 1.5) * 0.19, values, width=0.18, color=color, label=name)
        axis.set(title=label, ylim=(0, 1.04), xticks=x, xticklabels=("Original", "Dark", "Bright"))
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False)
    figure.savefig(directory / "comparison.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    samples = make_split(args.data_dir)
    train, validation, test = ([s for s in samples if s.split == split]
                               for split in ("train", "validation", "test"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = PatchFeatures(torch.device("cpu"))
    print(f"Preparing real and synthetic features from {len(train)} clean images", flush=True)
    clean, classes = training_features(args.data_dir, train, extractor)
    forest = IsolationForest(n_estimators=128, max_samples=256, random_state=SEED, n_jobs=4)
    joblib.dump(forest.fit(clean.numpy()), args.output_dir / "isolation_forest.joblib", compress=3)
    state = fit_cutpaste(clean, classes)
    torch.save(state, args.output_dir / "cutpaste.pt")
    methods = ExtraMethods(args.output_dir)
    val_scores = score_samples(args.data_dir, validation, 1.0, extractor, methods)
    state["thresholds"] = {name: float(np.percentile(values, 95)) for name, values in val_scores.items()}
    torch.save(state, args.output_dir / "cutpaste.pt")
    metrics, predictions = [], []
    for condition, factor in CONDITIONS.items():
        for name, values in score_samples(args.data_dir, test, factor, extractor, methods).items():
            row, image_rows = evaluate_condition(test, values, condition, name, state["thresholds"][name])
            metrics.append(row)
            predictions.extend(image_rows)
            print(f"{condition} {name}: AUROC {row[3]:.4f}, FP {row[4]}/{row[5]}, FN {row[7]}/{row[8]}", flush=True)
    for filename, new_rows in (("metrics.csv", metrics), ("image_scores.csv", predictions)):
        with (args.baseline_results / filename).open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            headers = next(reader)
            merged = [*reader, *new_rows]
        merged.sort(key=lambda row: (list(CONDITIONS).index(row[0]),
                                     ("PatchCore", "CNN+LSTM", *NEW_MODELS).index(row[1])))
        write_csv(args.output_dir / filename, headers, merged)
    write_csv(args.output_dir / "validation_scores.csv",
              ["relative_path", *NEW_MODELS],
              [(sample.relative_path.as_posix(), *(val_scores[name][i] for name in NEW_MODELS))
               for i, sample in enumerate(validation)])
    write_comparison(args.output_dir)


if __name__ == "__main__":
    main()
