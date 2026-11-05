import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ground_truth_score import claims_no_defect
from src.run_configuration import load_subset

FACTUALITY_PATH = Path("experiments/results/exp003_factualidade_configA.json")
OUTPUT_DIR = Path("experiments/figs/alucinacoes_configA")
PER_CATEGORY = 2
WRONG_OBJECT_TERMS = ("lens", "black hole", "bubble", "drink", "liquid", "hole.")
CATEGORY_TITLES = {
    "missed_defect": "Defeito real negado",
    "invented_defect": "Defeito inventado em imagem normal",
    "wrong_object": "Objeto ou tipo errado",
    "phantom_context": "Referência a contexto inexistente",
    "image_quality": "Defeito confundido com qualidade da foto",
}


def categorize(response):
    answer = response["answer"].lower()
    anomalous = response["label"] == "anomalous"
    if "facts" in answer:
        return "phantom_context"
    if "blur" in answer:
        return "image_quality"
    if anomalous and claims_no_defect(response["answer"]):
        return "missed_defect"
    if response["ground_truth_correct"] == 0 and not anomalous:
        return "invented_defect"
    if response["ground_truth_correct"] == 0 and any(term in answer for term in WRONG_OBJECT_TERMS):
        return "wrong_object"
    return None


def inconsistent_images(responses):
    grouped = defaultdict(dict)
    for response in responses:
        grouped[response["image_id"]][response["question_id"]] = response
    names = []
    for image_id, answers in grouped.items():
        says_yes = answers["q1_presence"]["answer"].lower().startswith("yes")
        denies = any(claims_no_defect(answers[key]["answer"]) for key in ("q2_location", "q3_type"))
        if says_yes and denies:
            names.append(image_id)
    return names


def metric_blind_spot(responses):
    wrong = [item for item in responses if item["ground_truth_correct"] == 0 and item["faithscore"] is not None]
    accepted = [item for item in wrong if item["faithscore"] == 1.0]
    right = [item for item in responses if item["ground_truth_correct"] == 1 and item["faithscore"] is not None]
    rejected = [item for item in right if item["faithscore"] == 0.0]
    return {
        "wrong_answers_scored": len(wrong),
        "wrong_answers_with_faithscore_1": len(accepted),
        "correct_answers_scored": len(right),
        "correct_answers_with_faithscore_0": len(rejected),
    }


def draw(response, image, category, path):
    picture = cv2.cvtColor(cv2.imread(image["path"]), cv2.COLOR_BGR2RGB)
    if image["mask_path"]:
        mask = cv2.imread(image["mask_path"], cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(picture, contours, -1, (255, 0, 0), 4)
    figure, axis = plt.subplots(figsize=(5, 5.8))
    axis.imshow(picture)
    axis.axis("off")
    axis.set_title(
        f"{CATEGORY_TITLES[category]}\n{response['image_id']} | {response['question_id']}\n{response['answer'][:75]}",
        fontsize=8,
    )
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main():
    with open(FACTUALITY_PATH, encoding="utf-8") as handle:
        responses = json.load(handle)["responses"]
    images = {image["id"]: image for image in load_subset()["images"]}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    categorized = [(categorize(item), item) for item in responses]
    counts = Counter(category for category, _ in categorized if category)
    gallery = []
    for category in CATEGORY_TITLES:
        members = [item for name, item in categorized if name == category]
        seen = set()
        chosen = []
        for item in members:
            if item["answer"] not in seen and len(chosen) < PER_CATEGORY:
                seen.add(item["answer"])
                chosen.append(item)
        for item in chosen:
            filename = f"{category}_{item['image_id']}_{item['question_id']}.png"
            draw(item, images[item["image_id"]], category, OUTPUT_DIR / filename)
            gallery.append(
                {
                    "file": filename,
                    "category": category,
                    "image_id": item["image_id"],
                    "question_id": item["question_id"],
                    "answer": item["answer"],
                    "faithscore": item["faithscore"],
                    "ground_truth_correct": item["ground_truth_correct"],
                }
            )
    inconsistent = inconsistent_images(responses)
    summary = {
        "status": "draft_pending_human_review",
        "selection": "Rule-based categories; first distinct answers per category, in subset order.",
        "category_counts": dict(counts),
        "uncategorized_wrong": sum(1 for name, item in categorized if name is None and item["ground_truth_correct"] == 0),
        "inconsistent_images": inconsistent,
        "metric_blind_spot": metric_blind_spot(responses),
        "gallery": gallery,
    }
    with open(OUTPUT_DIR / "gallery.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: summary[key] for key in summary if key != "gallery"}, indent=1))
    print(len(gallery))


if __name__ == "__main__":
    main()
