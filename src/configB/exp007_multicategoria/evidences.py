import time
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from src.build_evidences import evaluate_evidence
from src.extract_evidence import RUNS_DIR, extract_evidence
from src.ground_truth_score import mask_cells
from src.multicategory import (
    EVIDENCES_PATH, MVTEC_ROOT, PATCHCORE_TEST_PATH, SUBSET_PATH, load_json, region_cells, save_json,
)

# Mesmo valor do bottle no exp002b, para a evidência ser comparável entre os experimentos.
CORESET_SAMPLING_RATIO = 0.01


def build_engine():
    return Engine(accelerator="auto", default_root_dir=str(RUNS_DIR), logger=False, enable_progress_bar=False)


def checkpoint_path(category):
    found = sorted(RUNS_DIR.glob(f"Patchcore/MVTecAD/{category}/v*/weights/lightning/model.ckpt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def fit_and_test(category):
    datamodule = MVTecAD(root=str(MVTEC_ROOT), category=category, train_batch_size=16, eval_batch_size=16, num_workers=2)
    model = Patchcore(coreset_sampling_ratio=CORESET_SAMPLING_RATIO)
    engine = build_engine()
    ckpt = checkpoint_path(category)
    # Retomada: se o treino já existe, só refaz o teste.
    if ckpt is None:
        engine.fit(model=model, datamodule=datamodule)
        ckpt = checkpoint_path(category)
    metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt))
    return engine, model, ckpt, {key: round(float(value), 4) for key, value in metrics[0].items()}


def cell_hit(evidence, image):
    if image["label"] == "normal" or not evidence["is_anomalous"]:
        return None
    size = evidence["image_size"]
    return bool(region_cells(evidence["regions"], size["height"], size["width"]) & mask_cells(image["mask_path"]))


def main():
    subset = load_json(SUBSET_PATH)
    evidences = load_json(EVIDENCES_PATH) if Path(EVIDENCES_PATH).exists() else {}
    tests = load_json(PATCHCORE_TEST_PATH) if Path(PATCHCORE_TEST_PATH).exists() else {}
    for category in subset["categories"]:
        images = [image for image in subset["images"] if image["category"] == category]
        if category in tests and all(image["id"] in evidences for image in images):
            continue
        engine, model, ckpt, metrics = fit_and_test(category)
        tests[category] = {"checkpoint": ckpt.as_posix(), "coreset_sampling_ratio": CORESET_SAMPLING_RATIO, **metrics}
        for image in images:
            started = time.perf_counter()
            evidence = extract_evidence(engine, model, image["path"], category=category, ckpt_path=ckpt)
            evidence["seconds"] = round(time.perf_counter() - started, 2)
            evidence["quality"] = {**evaluate_evidence(evidence, image), "cell_hit": cell_hit(evidence, image)}
            evidences[image["id"]] = evidence
        # Salva por categoria: uma interrupção perde no máximo uma categoria.
        save_json(evidences, EVIDENCES_PATH)
        save_json(tests, PATCHCORE_TEST_PATH)
        print(category, metrics, flush=True)


if __name__ == "__main__":
    main()
