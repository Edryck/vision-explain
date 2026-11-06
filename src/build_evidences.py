import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.extract_evidence import extract_evidence, load_patchcore
from src.run_configuration import load_subset

OUTPUT_PATH = Path("experiments/exp002b_evidencias_subconjunto.json")


def boxes_mask(regions, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for region in regions:
        x_min, y_min, x_max, y_max = region["bbox"]
        mask[y_min:y_max, x_min:x_max] = True
    return mask


def evaluate_evidence(evidence, image):
    is_anomalous = image["label"] == "anomalous"
    detected = evidence["is_anomalous"] == is_anomalous
    if not is_anomalous:
        return {"detection_correct": detected, "box_iou": None, "evidence_correct": detected}
    truth = cv2.imread(image["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
    predicted = boxes_mask(evidence["regions"], *truth.shape)
    union = np.logical_or(truth, predicted).sum()
    overlap = np.logical_and(truth, predicted).sum()
    iou = float(overlap / union) if union else 0.0
    return {
        "detection_correct": detected,
        "box_iou": round(iou, 4),
        "evidence_correct": bool(detected and overlap > 0),
    }


def main():
    subset = load_subset()
    engine, model = load_patchcore()
    evidences = {}
    for image in subset["images"]:
        started = time.perf_counter()
        evidence = extract_evidence(engine, model, image["path"], category=image["category"])
        evidence["seconds"] = round(time.perf_counter() - started, 2)
        evidence["quality"] = evaluate_evidence(evidence, image)
        evidences[image["id"]] = evidence
        print(image["id"], evidence["is_anomalous"], len(evidence["regions"]), evidence["quality"], evidence["seconds"], flush=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(evidences, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
