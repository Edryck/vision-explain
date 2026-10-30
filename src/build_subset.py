import json
import random
from pathlib import Path

MVTEC_ROOT = Path("data/mvtec_ad/mvtec-ad")
CATEGORY = "bottle"
PER_TYPE = 10
SEED = 42
QUESTIONS = [
    {"id": "q1_presence", "text": "Is there any defect in this image?"},
    {"id": "q2_location", "text": "Where is the defect located in the image?"},
    {"id": "q3_type", "text": "What type of defect is it?"},
]


def build_subset():
    rng = random.Random(SEED)
    test_dir = MVTEC_ROOT / CATEGORY / "test"
    images = []
    for defect_dir in sorted(test_dir.iterdir()):
        files = sorted(defect_dir.glob("*.png"))
        chosen = sorted(rng.sample(files, PER_TYPE))
        for path in chosen:
            is_good = defect_dir.name == "good"
            mask = None if is_good else MVTEC_ROOT / CATEGORY / "ground_truth" / defect_dir.name / f"{path.stem}_mask.png"
            images.append(
                {
                    "id": f"{CATEGORY}_{defect_dir.name}_{path.stem}",
                    "path": path.as_posix(),
                    "category": CATEGORY,
                    "label": "normal" if is_good else "anomalous",
                    "defect_type": None if is_good else defect_dir.name,
                    "mask_path": None if is_good else mask.as_posix(),
                }
            )
    return {
        "dataset": "MVTec AD",
        "category": CATEGORY,
        "seed": SEED,
        "per_defect_type": PER_TYPE,
        "questions": QUESTIONS,
        "images": images,
    }


if __name__ == "__main__":
    subset = build_subset()
    with open("data/subset_avaliacao.json", "w", encoding="utf-8") as handle:
        json.dump(subset, handle, ensure_ascii=False, indent=2)
    print(len(subset["images"]))
