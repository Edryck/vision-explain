import json
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import MVTecAD, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

MVTEC_ROOT = Path("data/mvtec_ad/mvtec-ad")
RUNS_DIR = Path("data/anomalib_runs")
GRID_NAMES = [
    ["upper left", "upper center", "upper right"],
    ["middle left", "center", "middle right"],
    ["lower left", "lower center", "lower right"],
]
NORMALIZED_THRESHOLD = 0.5
DEFAULT_CKPT = RUNS_DIR / "Patchcore/MVTecAD/bottle/v1/weights/lightning/model.ckpt"
MIN_AREA_FRACTION = 0.001
MAX_REGIONS = 5


def build_engine():
    return Engine(
        accelerator="cpu",
        default_root_dir=str(RUNS_DIR),
        logger=False,
        enable_progress_bar=False,
    )


def fit_patchcore(category="bottle", coreset_sampling_ratio=0.01, root=MVTEC_ROOT):
    datamodule = MVTecAD(
        root=str(root),
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=4,
    )
    model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)
    engine = build_engine()
    engine.fit(model=model, datamodule=datamodule)
    return engine, model


def grid_position(cx, cy, width, height):
    col = min(int(cx / width * 3), 2)
    row = min(int(cy / height * 3), 2)
    return GRID_NAMES[row][col]


def describe_regions(anomaly_map, threshold, width, height):
    resized = cv2.resize(anomaly_map, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (resized >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = MIN_AREA_FRACTION * width * height
    regions = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        values = resized[labels == label]
        cx, cy = centroids[label]
        regions.append(
            {
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area_pixels": int(area),
                "area_fraction": round(float(area) / (width * height), 5),
                "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "position": grid_position(cx, cy, width, height),
                "max_score": round(float(values.max()), 4),
                "mean_score": round(float(values.mean()), 4),
            }
        )
    regions.sort(key=lambda r: r["max_score"], reverse=True)
    regions = regions[:MAX_REGIONS]
    for index, region in enumerate(regions, start=1):
        region["id"] = index
    return regions


def load_patchcore(coreset_sampling_ratio=0.01):
    return build_engine(), Patchcore(coreset_sampling_ratio=coreset_sampling_ratio)


def extract_evidence(engine, model, image_path, category="bottle", ckpt_path=DEFAULT_CKPT):
    image_path = Path(image_path)
    predictions = engine.predict(
        model=model, dataset=PredictDataset(str(image_path)), ckpt_path=str(ckpt_path)
    )
    batch = predictions[0]
    height, width = cv2.imread(str(image_path)).shape[:2]
    anomaly_map = batch.anomaly_map.squeeze().detach().cpu().numpy().astype(np.float32)
    score = float(batch.pred_score.reshape(-1)[0])
    is_anomalous = bool(batch.pred_label.reshape(-1)[0])
    regions = describe_regions(anomaly_map, NORMALIZED_THRESHOLD, width, height) if is_anomalous else []
    return {
        "image": str(image_path),
        "category": category,
        "model": {"name": "PatchCore", "backbone": "wide_resnet50_2", "library": "anomalib"},
        "image_size": {"width": int(width), "height": int(height)},
        "image_score": round(score, 4),
        "threshold": NORMALIZED_THRESHOLD,
        "is_anomalous": is_anomalous,
        "regions": regions,
    }


def evidence_to_text(evidence):
    if not evidence["is_anomalous"]:
        return "The anomaly detector found no anomalous region in this image."
    lines = [
        f"The anomaly detector flagged this image as anomalous "
        f"(score {evidence['image_score']}, threshold {evidence['threshold']})."
    ]
    for region in evidence["regions"]:
        percent = region["area_fraction"] * 100
        lines.append(
            f"Region {region['id']}: {region['position']}, covers {percent:.1f}% of the image, "
            f"peak score {region['max_score']}."
        )
    return "\n".join(lines)


def save_evidences(evidences, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(evidences, handle, ensure_ascii=False, indent=2)
