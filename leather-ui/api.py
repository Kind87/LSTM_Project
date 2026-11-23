"""Local FastAPI server for the trained leather anomaly models."""

import base64
import csv
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent / "lstm-day1"
DEMO_SAMPLES = ROOT / "demo-samples"
RESULTS = EXPERIMENT / "full-results"
FOUR_MODEL_RESULTS = EXPERIMENT / "four-model-results"
DIST = ROOT / "frontend" / "dist"
sys.path.insert(0, str(EXPERIMENT))

from add_anomaly_methods import ExtraMethods
from run_experiment import PatchFeatures, brightness_versions, load_image
from run_full_experiment import BrightnessLSTM, CONDITIONS, patch_errors


Lighting = Literal["original", "dark", "bright"]
SAMPLES = {
    "normal": ("good/000.png", "正常皮革"),
    "cut": ("cut/000.png", "切痕"),
    "color": ("color/009.png", "顏色異常"),
    "fold": ("fold/000.png", "摺痕"),
    "glue": ("glue/000.png", "膠痕"),
    "poke": ("poke/000.png", "戳痕"),
}
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def encode_preview(image: Image.Image) -> str:
    image = image.copy()
    image.thumbnail((660, 660), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=89)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay(image: Image.Image, values: torch.Tensor, threshold: float) -> str:
    base = image.copy()
    base.thumbnail((540, 540), Image.Resampling.LANCZOS)
    grid = Image.fromarray((values / threshold).numpy()).resize(base.size, Image.Resampling.BILINEAR)
    relative = np.asarray(grid)
    level = np.clip(relative, 0, 1.5)
    stops = [0.0, 0.55, 0.85, 1.0, 1.5]
    colors = [(21, 91, 179), (11, 177, 200), (240, 202, 68), (255, 107, 59), (216, 48, 42)]
    heat = np.stack([np.interp(level, stops, channel) for channel in zip(*colors)], axis=-1)
    alpha = np.clip(level * 0.48, 0.15, 0.80)[..., None]
    blended = np.asarray(base) * (1 - alpha) + heat * alpha
    return encode_preview(Image.fromarray(blended.astype(np.uint8)))


class Inspector:
    def __init__(self):
        torch.set_num_threads(min(6, torch.get_num_threads()))
        state = torch.load(RESULTS / "model_state.pt", map_location="cpu", weights_only=True)
        self.extractor = PatchFeatures(torch.device("cpu"))
        self.lstm = BrightnessLSTM(state["feature_size"])
        self.lstm.load_state_dict(state["lstm"])
        self.lstm.eval()
        self.memory = state["memory_bank"]
        self.extras = ExtraMethods(FOUR_MODEL_RESULTS)
        self.thresholds = {**state["thresholds"], **self.extras.thresholds}

    @torch.inference_mode()
    def inspect(self, image: Image.Image, lighting: Lighting):
        inspected = ImageEnhance.Brightness(image).enhance(CONDITIONS[lighting])
        frames = brightness_versions(inspected)
        features = self.extractor(frames)
        _, height, width, channels = features.shape
        patches = features.permute(1, 2, 0, 3).reshape(-1, len(frames), channels)
        nearest = [
            torch.cdist(batch, self.memory).amin(dim=1)
            for batch in patches[:, -1].split(256)
        ]
        patchcore_grid = torch.cat(nearest).reshape(height, width)
        lstm_grid = patch_errors(self.lstm, patches).reshape(height, width)

        models = {}
        grids = {"PatchCore": patchcore_grid, "CNN+LSTM": lstm_grid}
        grids.update(self.extras.grids(features[-1]))
        for name, grid in grids.items():
            threshold = self.thresholds[name]
            score = grid.max().item()
            models[name] = {
                "score": score,
                "threshold": threshold,
                "anomaly": score > threshold,
                "heatmap": overlay(inspected, grid, threshold),
            }
        return {
            "preview": encode_preview(inspected),
            "frames": [encode_preview(frame) for frame in frames],
            "models": models,
            "lighting": lighting,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inspector = Inspector()
    yield


app = FastAPI(title="Surface Lab API", lifespan=lifespan)


def sample_path(sample_id: str) -> tuple[Path, str]:
    sample = SAMPLES.get(sample_id)
    if sample is None:
        raise HTTPException(404, "找不到這張樣本。")
    return DEMO_SAMPLES / sample[0], sample[1]


@app.get("/api/samples")
def list_samples():
    return [
        {"id": key, "name": value[1], "filename": value[0], "anomaly": key != "normal"}
        for key, value in SAMPLES.items()
    ]


@app.get("/api/samples/{sample_id}/image")
def sample_image(sample_id: str):
    path, _ = sample_path(sample_id)
    return FileResponse(path, media_type="image/png")


@app.get("/api/samples/{sample_id}/inspect")
def inspect_sample(sample_id: str, request: Request, lighting: Lighting = "original"):
    path, _ = sample_path(sample_id)
    return request.app.state.inspector.inspect(load_image(path), lighting)


@app.post("/api/inspect")
def inspect_upload(request: Request, file: UploadFile = File(), lighting: Lighting = Form("original")):
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "影像檔案須小於 10 MB。")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "請上傳 JPG、PNG 或 WebP 影像。")
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, "影像尺寸太大，請先縮小後再上傳。")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as error:
        raise HTTPException(400, "無法讀取這張影像，請確認檔案格式。") from error
    return request.app.state.inspector.inspect(image, lighting)


@app.get("/api/metrics")
def experiment_metrics():
    with (FOUR_MODEL_RESULTS / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@app.get("/api/artifacts/{name}")
def experiment_artifact(name: str):
    if name == "report.md":
        return FileResponse(EXPERIMENT / name, media_type="text/markdown; charset=utf-8", filename=name)
    if name == "metrics.csv":
        return FileResponse(FOUR_MODEL_RESULTS / name, media_type="text/csv; charset=utf-8", filename=name)
    if name == "comparison.png":
        return FileResponse(FOUR_MODEL_RESULTS / name, media_type="image/png")
    if name == "error_examples.png":
        return FileResponse(RESULTS / name, media_type="image/png")
    raise HTTPException(404, "找不到這份實驗資料。")


@app.get("/")
def home():
    return FileResponse(DIST / "index.html")


app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="assets")
