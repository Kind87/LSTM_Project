"""Prepare the leather dataset and run the Day 1 PatchCore smoke test."""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


SEED = 42
IMAGE_SIZE = 224
BRIGHTNESS_FACTORS = (0.70, 0.80, 0.90, 1.00)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
SMOKE_TRAIN_IMAGES = 8
SMOKE_BANK_SIZE = 128


@dataclass(frozen=True)
class Sample:
    relative_path: Path
    split: str
    defect_type: str

    @property
    def label(self) -> int:
        return int(self.defect_type != "good")


def make_split(data_dir: Path) -> list[Sample]:
    normal_paths = sorted((data_dir / "train/good").glob("*.png"))
    test_paths = sorted((data_dir / "test").glob("*/*.png"))
    test_normal_count = sum(path.parent.name == "good" for path in test_paths)
    counts = (len(normal_paths), test_normal_count, len(test_paths) - test_normal_count)
    if counts != (245, 32, 92):
        raise ValueError(
            f"Expected the complete MVTec AD leather folder (245, 32, 92), got {counts}"
        )

    validation_paths = set(random.Random(SEED).sample(normal_paths, k=49))
    samples = [
        Sample(
            path.relative_to(data_dir),
            "validation" if path in validation_paths else "train",
            "good",
        )
        for path in normal_paths
    ]
    samples.extend(
        Sample(path.relative_to(data_dir), "test", path.parent.name) for path in test_paths
    )
    for sample in samples:
        with Image.open(data_dir / sample.relative_path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG: {sample.relative_path}")
            image.verify()
    return samples


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def brightness_versions(image: Image.Image) -> list[Image.Image]:
    enhancer = ImageEnhance.Brightness(image)
    return [enhancer.enhance(factor) for factor in BRIGHTNESS_FACTORS]


class PatchFeatures(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        backbone = resnet18(weights=WEIGHTS).requires_grad_(False).eval()
        self.encoder = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).to(device).eval()
        self.device = device
        weight_transforms = WEIGHTS.transforms()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(weight_transforms.mean, weight_transforms.std),
            ]
        )

    @torch.inference_mode()
    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        layers = self.encoder(batch)
        layer2 = layers["layer2"]
        layer3 = F.interpolate(
            layers["layer3"], size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1).permute(0, 2, 3, 1)
        return features.contiguous()


def build_memory_bank(
    patches: torch.Tensor, capacity: int = 1024, candidate_limit: int = 10_000
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    candidate_indices = torch.randperm(len(patches), generator=generator)[:candidate_limit]
    candidates = patches[candidate_indices.to(patches.device)]
    selected = []
    distances = candidates.new_full((len(candidates),), float("inf"))
    next_index = 0
    for _ in range(min(capacity, len(candidates))):
        selected.append(next_index)
        new_distances = (candidates - candidates[next_index]).square().sum(dim=1)
        distances = torch.minimum(distances, new_distances)
        distances[next_index] = -1
        next_index = int(distances.argmax().item())
    return candidates[selected]


def score_images(features: torch.Tensor, memory_bank: torch.Tensor) -> list[float]:
    scores = []
    for image_features in features:
        patches = image_features.flatten(0, 1)
        nearest = [torch.cdist(batch, memory_bank).amin(dim=1) for batch in patches.split(256)]
        scores.append(torch.cat(nearest).max().item())
    return scores


def write_csv(path: Path, headers: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def make_preview(data_dir: Path, normal: Sample, defect: Sample) -> Image.Image:
    normal_image = load_image(data_dir / normal.relative_path)
    defect_image = load_image(data_dir / defect.relative_path)
    defect_path = defect.relative_path
    mask_path = data_dir / "ground_truth" / defect.defect_type / f"{defect_path.stem}_mask.png"
    with Image.open(mask_path) as mask:
        bounds = mask.convert("L").getbbox()
    if bounds is None:
        raise ValueError(f"Empty defect mask: {mask_path}")
    x0, y0, x1, y1 = bounds
    radius = max(x1 - x0, y1 - y0) // 2 + 48
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    crop_size = min(radius * 2, *defect_image.size)
    left = min(max(0, center_x - radius), defect_image.width - crop_size)
    top = min(max(0, center_y - radius), defect_image.height - crop_size)
    crop_box = (left, top, left + crop_size, top + crop_size)

    rows = [
        (f"Normal training image  /  {normal.relative_path.as_posix()}", normal_image, None),
        (f"Cut test image  /  {defect_path.as_posix()}", defect_image, None),
        ("Same cut image, zoomed in around the defect", defect_image, crop_box),
    ]
    tile_size, gap, margin = 250, 12, 24
    width = 2 * margin + 4 * tile_size + 3 * gap
    height = 88 + len(rows) * (tile_size + 36) + 8
    canvas = Image.new("RGB", (width, height), "#f5f5f2")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=25)
    label_font = ImageFont.load_default(size=17)
    draw.text((margin, 18), "MVTec AD Leather | Synthetic brightness sequence", fill="#202124", font=title_font)
    for column, factor in enumerate(BRIGHTNESS_FACTORS):
        x = margin + column * (tile_size + gap)
        draw.text((x, 59), f"{factor:.0%}", fill="#202124", font=label_font)
    for row, (caption, image, crop_box) in enumerate(rows):
        y = 88 + row * (tile_size + 36)
        draw.text((margin, y), caption, fill="#202124", font=label_font)
        for column, version in enumerate(brightness_versions(image)):
            tile = version.crop(crop_box) if crop_box is not None else version
            x = margin + column * (tile_size + gap)
            canvas.paste(tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS), (x, y + 24))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="MVTec AD leather folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment. Use --device cpu.")
    device = torch.device(args.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    samples = make_split(args.data_dir)
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    print(f"Verified data: {len(train)} train, {len(validation)} validation, {len(test)} test.", flush=True)
    cut_sample = next(sample for sample in test if sample.defect_type == "cut")
    preview = make_preview(args.data_dir, train[0], cut_sample)

    train_subset = train[:SMOKE_TRAIN_IMAGES]
    test_subset = [sample for sample in test if sample.label == 0][:3]
    for defect_type in sorted({sample.defect_type for sample in test if sample.label == 1}):
        test_subset.append(next(sample for sample in test if sample.defect_type == defect_type))

    extractor = PatchFeatures(device)
    train_images = [load_image(args.data_dir / sample.relative_path) for sample in train_subset]
    train_features = extractor(train_images)
    sequence = extractor(brightness_versions(train_images[0]))
    test_features = extractor(
        [load_image(args.data_dir / sample.relative_path) for sample in test_subset]
    )
    memory_bank = build_memory_bank(train_features.flatten(0, 2), capacity=SMOKE_BANK_SIZE)
    scores = score_images(test_features, memory_bank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "split.csv",
        ["relative_path", "split", "defect_type", "label"],
        [(s.relative_path.as_posix(), s.split, s.defect_type, s.label) for s in samples],
    )
    write_csv(
        args.output_dir / "patchcore_smoke_scores.csv",
        ["relative_path", "defect_type", "label", "anomaly_score"],
        [(s.relative_path.as_posix(), s.defect_type, s.label, f"{score:.6f}")
         for s, score in zip(test_subset, scores, strict=True)],
    )
    preview.save(args.output_dir / "brightness_preview.png")
    summary = {
        "device": args.device,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "weights": WEIGHTS.name,
        "seed": SEED,
        "data_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "brightness_factors": BRIGHTNESS_FACTORS,
        "sequence_feature_shape": list(sequence.shape),
        "smoke_train_images": [s.relative_path.as_posix() for s in train_subset],
        "smoke_test_image_count": len(test_subset),
        "smoke_memory_bank_shape": list(memory_bank.shape),
    }
    (args.output_dir / "day1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Feature sequence: {list(sequence.shape)}. Memory bank: {list(memory_bank.shape)}.")
    print(f"Wrote {len(scores)} smoke-test scores to {args.output_dir}")


if __name__ == "__main__":
    main()
