import argparse
import csv
import random
from collections import defaultdict

import numpy as np

from src.compare_configurations import mcnemar
from src.evaluate_factuality import summarize
from src.faithscore import decompose
from src.ground_truth_score import answer_cells, presence_prediction
from src.multicategory import (
    EVIDENCES_PATH, PATCHCORE_TEST_PATH, RESPONSES_PATHS, RESULTS_DIR, SUBSET_PATH,
    load_json, region_cells, save_json, says_no_defect, score_response,
)

QUESTIONS = ("q1_presence", "q2_location", "q3_type")
BOOTSTRAP_RESAMPLES = 2000


def binary(value):
    return None if value is None else int(value)


def evidence_fidelity(response, evidence):
    answer = response["answer"]
    question = response["question_id"]
    if question == "q1_presence":
        predicted = presence_prediction(answer)
        agrees = None if predicted is None else predicted == evidence["is_anomalous"]
        return {"agrees_with_evidence": binary(agrees)}
    if question == "q2_location":
        if not evidence["is_anomalous"]:
            return {"agrees_with_evidence": binary(says_no_defect(answer)), "copies_report_position": None}
        positions = {region["position"] for region in evidence["regions"]}
        cells = answer_cells(answer)
        size = evidence["image_size"]
        agrees = bool(cells & region_cells(evidence["regions"], size["height"], size["width"])) if cells else None
        return {
            "agrees_with_evidence": binary(agrees),
            "copies_report_position": int(answer.strip(" .").lower() in positions),
        }
    # O relatório não traz o tipo; aqui só medimos se a resposta virou rótulo genérico.
    return {"generic_anomaly_answer": int("anomal" in answer.lower())}


def score_configuration(configuration, with_faithscore):
    subset = load_json(SUBSET_PATH)
    images = {image["id"]: image for image in subset["images"]}
    evidences = load_json(EVIDENCES_PATH)
    responses = load_json(RESPONSES_PATHS[configuration])["responses"]
    verifier = None
    if with_faithscore:
        from src.faithscore import faithscore
        from src.vlm import load_vlm

        processor, model, device, _ = load_vlm()
        verifier = lambda response: faithscore(processor, model, device, response["image"], response["answer"])
    scored = []
    for index, response in enumerate(responses, start=1):
        image = images[response["image_id"]]
        evidence = evidences[response["image_id"]]
        item = {
            "image_id": response["image_id"],
            "category": image["category"],
            "label": image["label"],
            "defect_type": image["defect_type"] or "good",
            "question_id": response["question_id"],
            "answer": response["answer"],
            "new_tokens": response["new_tokens"],
            "n_words": len(response["answer"].split()),
            "n_claims": len(decompose(response["answer"])),
            "ground_truth_correct": binary(score_response(response, image)),
            "evidence_correct": int(evidence["quality"]["evidence_correct"]),
            **evidence_fidelity(response, evidence),
        }
        if verifier:
            item["faithscore"] = verifier(response)["faithscore"]
        scored.append(item)
        if verifier and index % 100 == 0:
            print(index, len(responses), flush=True)
    output = RESULTS_DIR / f"config{configuration}_pontuado.json"
    save_json({"configuration": configuration, "with_faithscore": with_faithscore, "responses": scored}, output)
    print(output, len(scored))


def image_consistency(scored):
    # Contradição: q1 diz que há defeito e q2/q3 negam, ou o contrário.
    by_image = defaultdict(dict)
    for item in scored:
        by_image[item["image_id"]][item["question_id"]] = item
    result = {}
    for image_id, answers in by_image.items():
        presence = presence_prediction(answers["q1_presence"]["answer"])
        if presence is None:
            result[image_id] = None
            continue
        denies = [says_no_defect(answers[q]["answer"]) for q in ("q2_location", "q3_type")]
        result[image_id] = int(any(denies) if presence else not all(denies))
    return result


def mean_of(items, field):
    return summarize([item.get(field) for item in items])


def configuration_summary(scored):
    consistency = image_consistency(scored)
    by_question = {q: [item for item in scored if item["question_id"] == q] for q in QUESTIONS}
    summary = {
        "ground_truth_correct": mean_of(scored, "ground_truth_correct"),
        "by_question": {q: mean_of(items, "ground_truth_correct") for q, items in by_question.items()},
        "inconsistent_images": summarize(list(consistency.values())),
        "richness": {
            "new_tokens": mean_of(scored, "new_tokens"),
            "n_words": mean_of(scored, "n_words"),
            "n_claims": mean_of(scored, "n_claims"),
            "zero_claim_rate": round(sum(item["n_claims"] == 0 for item in scored) / len(scored), 4),
        },
        "evidence_fidelity": {
            "q1_agrees_with_evidence": mean_of(by_question["q1_presence"], "agrees_with_evidence"),
            "q2_agrees_with_evidence": mean_of(by_question["q2_location"], "agrees_with_evidence"),
            "q2_copies_report_position": mean_of(by_question["q2_location"], "copies_report_position"),
            "q3_generic_anomaly_answer": mean_of(by_question["q3_type"], "generic_anomaly_answer"),
        },
        "by_evidence_correct": {
            str(flag): mean_of([item for item in scored if item["evidence_correct"] == flag], "ground_truth_correct")
            for flag in (0, 1)
        },
    }
    if "faithscore" in scored[0]:
        summary["faithscore"] = mean_of(scored, "faithscore")
    return summary


def bootstrap_difference(pairs, seed=42):
    # Reamostra imagens, não respostas: as 3 perguntas da mesma imagem não são independentes.
    by_image = defaultdict(list)
    for a, b in pairs:
        if a["ground_truth_correct"] is not None and b["ground_truth_correct"] is not None:
            by_image[a["image_id"]].append(b["ground_truth_correct"] - a["ground_truth_correct"])
    if not by_image:
        return None
    groups = list(by_image.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [value for group in rng.choices(groups, k=len(groups)) for value in group]
        estimates.append(float(np.mean(sample)))
    observed = float(np.mean([value for group in groups for value in group]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"difference_b_minus_a": round(observed, 4), "ci95": [round(float(low), 4), round(float(high), 4)], "n_images": len(groups)}


def patchcore_summary(subset, evidences, tests):
    rows = {}
    for category in subset["categories"]:
        qualities = [evidences[i["id"]]["quality"] for i in subset["images"] if i["category"] == category]
        normals = [i for i in subset["images"] if i["category"] == category and i["label"] == "normal"]
        rows[category] = {
            **{key: value for key, value in tests[category].items() if key != "checkpoint"},
            "subset_detection_accuracy": round(float(np.mean([q["detection_correct"] for q in qualities])), 4),
            "subset_false_positives": sum(evidences[i["id"]]["is_anomalous"] for i in normals),
            "subset_false_negatives": sum(
                not evidences[i["id"]]["is_anomalous"]
                for i in subset["images"] if i["category"] == category and i["label"] == "anomalous"
            ),
            "subset_box_iou": summarize([q["box_iou"] for q in qualities]),
            "subset_cell_hit": summarize([binary(q["cell_hit"]) for q in qualities]),
            "subset_evidence_correct": round(float(np.mean([q["evidence_correct"] for q in qualities])), 4),
        }
    return rows


def report():
    subset = load_json(SUBSET_PATH)
    evidences = load_json(EVIDENCES_PATH)
    tests = load_json(PATCHCORE_TEST_PATH)
    scored = {c: load_json(RESULTS_DIR / f"config{c}_pontuado.json")["responses"] for c in ("A", "B")}
    index_b = {(item["image_id"], item["question_id"]): item for item in scored["B"]}
    pairs = [(a, index_b[(a["image_id"], a["question_id"])]) for a in scored["A"]]
    result = {
        "patchcore": patchcore_summary(subset, evidences, tests),
        "overall": {c: configuration_summary(items) for c, items in scored.items()},
        "paired": {
            "overall": {**mcnemar(pairs), "bootstrap": bootstrap_difference(pairs)},
            **{q: {**mcnemar([p for p in pairs if p[0]["question_id"] == q]), "bootstrap": bootstrap_difference([p for p in pairs if p[0]["question_id"] == q])} for q in QUESTIONS},
            "by_evidence_correct": {
                str(flag): mcnemar([p for p in pairs if p[1]["evidence_correct"] == flag]) for flag in (0, 1)
            },
        },
        "by_category": {},
    }
    for category in subset["categories"]:
        selected = [p for p in pairs if p[0]["category"] == category]
        result["by_category"][category] = {
            "A": configuration_summary([a for a, _ in selected]),
            "B": configuration_summary([b for _, b in selected]),
            "paired": mcnemar(selected),
        }
    save_json(result, RESULTS_DIR / "resumo_exp007.json")
    write_table(result)
    print(RESULTS_DIR / "resumo_exp007.json")


def write_table(result):
    path = RESULTS_DIR / "tabela_por_categoria.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "categoria", "image_auroc", "pixel_auroc", "evidencia_correta",
            "acerto_A", "acerto_B", "q1_A", "q1_B", "q2_A", "q2_B", "q3_A", "q3_B",
            "inconsistencia_A", "inconsistencia_B", "tokens_A", "tokens_B", "mcnemar_p",
        ])
        for category, row in result["by_category"].items():
            patchcore = result["patchcore"][category]
            a, b = row["A"], row["B"]
            writer.writerow([
                category, patchcore.get("image_AUROC"), patchcore.get("pixel_AUROC"), patchcore["subset_evidence_correct"],
                a["ground_truth_correct"]["mean"], b["ground_truth_correct"]["mean"],
                *[x["by_question"][q]["mean"] for q in QUESTIONS for x in (a, b)],
                a["inconsistent_images"]["mean"], b["inconsistent_images"]["mean"],
                a["richness"]["new_tokens"]["mean"], b["richness"]["new_tokens"]["mean"],
                row["paired"]["exact_p_value"],
            ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score", "report"))
    parser.add_argument("--config", choices=("A", "B"))
    parser.add_argument("--faithscore", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "score":
        score_configuration(arguments.config, arguments.faithscore)
    else:
        report()
