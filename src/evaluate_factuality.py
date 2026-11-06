import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.faithscore import VERIFIER_THRESHOLD, faithscore
from src.ground_truth_score import score_response
from src.run_configuration import load_subset
from src.vlm import MODEL_ID, load_vlm


def summarize(values):
    # None significa indefinido: fica fora da média e é contado à parte.
    known = [value for value in values if value is not None]
    if not known:
        return {"n": 0, "n_undefined": len(values), "mean": None}
    array = np.array(known, dtype=float)
    return {
        "n": len(known),
        "n_undefined": len(values) - len(known),
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std(ddof=1)), 4) if len(known) > 1 else None,
        "median": round(float(np.median(array)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }


def group_summary(responses, field, key):
    groups = defaultdict(list)
    for response in responses:
        groups[key(response)].append(response[field])
    return {name: summarize(values) for name, values in sorted(groups.items())}


def save(result, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def build_result(input_path, configuration, scored):
    result = {
        "source": str(input_path),
        "configuration": configuration,
        "verifier": {"model": MODEL_ID, "image_splitting": False, "threshold": VERIFIER_THRESHOLD},
        "n_responses": len(scored),
        "mean_new_tokens": round(float(np.mean([item["new_tokens"] for item in scored])), 2),
        "faithscore": {
            "overall": summarize([item["faithscore"] for item in scored]),
            "by_question": group_summary(scored, "faithscore", lambda item: item["question_id"]),
            "by_defect_type": group_summary(scored, "faithscore", lambda item: item["defect_type"]),
        },
        "ground_truth_correct": {
            "overall": summarize([item["ground_truth_correct"] for item in scored]),
            "by_question": group_summary(scored, "ground_truth_correct", lambda item: item["question_id"]),
            "by_defect_type": group_summary(scored, "ground_truth_correct", lambda item: item["defect_type"]),
            "by_question_and_type": group_summary(
                scored, "ground_truth_correct", lambda item: f"{item['question_id']}|{item['defect_type']}"
            ),
        },
        "responses": scored,
    }
    return result


def evaluate(input_path, output_path):
    with open(input_path, encoding="utf-8") as handle:
        experiment = json.load(handle)
    masks = {image["id"]: image["mask_path"] for image in load_subset()["images"]}
    processor, model, device, _ = load_vlm()
    scored = []
    for index, response in enumerate(experiment["responses"], start=1):
        metric = faithscore(processor, model, device, response["image"], response["answer"])
        correct = score_response(response, masks[response["image_id"]])
        scored.append(
            {
                "image_id": response["image_id"],
                "label": response["label"],
                "defect_type": response["defect_type"] or "good",
                "question_id": response["question_id"],
                "answer": response["answer"],
                "new_tokens": response["new_tokens"],
                "claims": metric["claims"],
                "n_claims": metric["n_claims"],
                "faithscore": metric["faithscore"],
                "ground_truth_correct": None if correct is None else int(correct),
            }
        )
        print(index, len(experiment["responses"]), metric["n_claims"], flush=True)
    result = build_result(input_path, experiment["configuration"], scored)
    save(result, output_path)
    return result


def rescore(output_path):
    with open(output_path, encoding="utf-8") as handle:
        previous = json.load(handle)
    with open(previous["source"], encoding="utf-8") as handle:
        originals = {(item["image_id"], item["question_id"]): item for item in json.load(handle)["responses"]}
    masks = {image["id"]: image["mask_path"] for image in load_subset()["images"]}
    for item in previous["responses"]:
        original = originals[(item["image_id"], item["question_id"])]
        correct = score_response(original, masks[item["image_id"]])
        item["ground_truth_correct"] = None if correct is None else int(correct)
    result = build_result(previous["source"], previous["configuration"], previous["responses"])
    save(result, output_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/exp003_configA_completo.json")
    parser.add_argument("--output", default="experiments/results/exp003_factualidade_configA.json")
    parser.add_argument("--rescore", action="store_true")
    arguments = parser.parse_args()
    outcome = rescore(arguments.output) if arguments.rescore else evaluate(arguments.input, arguments.output)
    print(json.dumps({key: outcome[key]["overall"] for key in ("faithscore", "ground_truth_correct")}, indent=2))
