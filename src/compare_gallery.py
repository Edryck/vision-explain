import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.compare_configurations import EVIDENCES, FACTUALITY_A, FACTUALITY_B, load, pair_responses
from src.extract_evidence import evidence_to_text
from src.run_configuration import load_subset

OUTPUT_DIR = Path("docs/artigo1/figs/comparacao_A_B")


def classify(first, second):
    a, b = first["ground_truth_correct"], second["ground_truth_correct"]
    if a == 0 and b == 1:
        return "b_corrected_a"
    if a == 1 and b == 0:
        return "b_worsened_a"
    if a == 0 and b == 0:
        return "both_wrong"
    return None


def draw(first, second, image, evidence, case, path):
    picture = cv2.cvtColor(cv2.imread(image["path"]), cv2.COLOR_BGR2RGB)
    if image["mask_path"]:
        mask = cv2.imread(image["mask_path"], cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(picture, contours, -1, (255, 0, 0), 4)
    for region in evidence["regions"]:
        x_min, y_min, x_max, y_max = region["bbox"]
        cv2.rectangle(picture, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)
    figure, axis = plt.subplots(figsize=(5, 6))
    axis.imshow(picture)
    axis.axis("off")
    axis.set_title(
        f"{case} | {first['image_id']} | {first['question_id']}\nA: {first['answer'][:70]}\nB: {second['answer'][:70]}",
        fontsize=8,
    )
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main():
    pairs = pair_responses(load(FACTUALITY_A), load(FACTUALITY_B))
    evidences = load(EVIDENCES)
    images = {image["id"]: image for image in load_subset()["images"]}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, gallery, seen = {}, [], set()
    for first, second in pairs:
        case = classify(first, second)
        if case is None:
            continue
        counts[case] = counts.get(case, 0) + 1
        # Um exemplo por caso e grupo, para a galeria cobrir todos os tipos de defeito.
        signature = (case, first["defect_type"])
        if signature in seen:
            continue
        seen.add(signature)
        filename = f"{case}_{first['image_id']}_{first['question_id']}.png"
        evidence = evidences[first["image_id"]]
        draw(first, second, images[first["image_id"]], evidence, case, OUTPUT_DIR / filename)
        gallery.append(
            {
                "file": filename,
                "case": case,
                "image_id": first["image_id"],
                "question_id": first["question_id"],
                "answer_a": first["answer"],
                "answer_b": second["answer"],
                "evidence": evidence_to_text(evidence),
            }
        )
    with open(OUTPUT_DIR / "gallery.json", "w", encoding="utf-8") as handle:
        json.dump({"pair_counts": counts, "gallery": gallery}, handle, ensure_ascii=False, indent=2)
    print(counts, len(gallery))
    for item in gallery:
        print(item["case"], item["image_id"], item["question_id"], "|", item["answer_a"][:60], "|", item["answer_b"][:40])


if __name__ == "__main__":
    main()
