import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

FACTUALITY_A = Path("experiments/results/exp003_factualidade_configA.json")
FACTUALITY_B = Path("experiments/results/exp005_factualidade_configB.json")
COST_A = Path("experiments/results/custo_configA.json")
COST_B = Path("experiments/results/custo_configB.json")
EVIDENCES = Path("experiments/exp002b_evidencias_subconjunto.json")
TABLE_PATH = Path("experiments/results/tabela_A_vs_B.csv")
PAIRED_PATH = Path("experiments/results/comparacao_pareada_A_vs_B.json")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def pair_responses(first, second):
    index = {(item["image_id"], item["question_id"]): item for item in second["responses"]}
    return [(item, index[(item["image_id"], item["question_id"])]) for item in first["responses"]]


def mcnemar(pairs):
    # Indefinidos ficam fora: o teste exige resultado binário nos dois lados do par.
    known = [
        (a["ground_truth_correct"], b["ground_truth_correct"])
        for a, b in pairs
        if a["ground_truth_correct"] is not None and b["ground_truth_correct"] is not None
    ]
    only_a = sum(1 for a, b in known if a == 1 and b == 0)
    only_b = sum(1 for a, b in known if a == 0 and b == 1)
    discordant = only_a + only_b
    # Versão exata, porque há poucos pares discordantes por grupo.
    p_value = binomtest(only_a, discordant, 0.5).pvalue if discordant else None
    return {
        "n_pairs": len(known),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_correct": sum(1 for a, b in known if a == 1 and b == 1),
        "both_wrong": sum(1 for a, b in known if a == 0 and b == 0),
        "exact_p_value": None if p_value is None else round(float(p_value), 6),
    }


def copied_location(pairs, evidences):
    answers = [
        b
        for _, b in pairs
        if b["question_id"] == "q2_location" and b["label"] == "anomalous"
    ]
    copied = sum(
        1
        for item in answers
        if item["answer"].strip(" .").lower() in {r["position"] for r in evidences[item["image_id"]]["regions"]}
    )
    return {"anomalous_q2": len(answers), "answer_is_report_position": copied}


def table_rows(result_a, result_b, cost_a, cost_b, evidences):
    def correct(result, group, key):
        return result["ground_truth_correct"][group][key]["mean"]

    patchcore_seconds = float(np.mean([item["seconds"] for item in evidences.values()]))
    rows = [
        ("acerto_gabarito_geral", result_a["ground_truth_correct"]["overall"]["mean"], result_b["ground_truth_correct"]["overall"]["mean"]),
        ("acerto_gabarito_n", result_a["ground_truth_correct"]["overall"]["n"], result_b["ground_truth_correct"]["overall"]["n"]),
    ]
    for question in ("q1_presence", "q2_location", "q3_type"):
        rows.append((f"acerto_{question}", correct(result_a, "by_question", question), correct(result_b, "by_question", question)))
    for group in ("broken_large", "broken_small", "contamination", "good"):
        rows.append((f"acerto_{group}", correct(result_a, "by_defect_type", group), correct(result_b, "by_defect_type", group)))
    rows += [
        ("faithscore_medio", result_a["faithscore"]["overall"]["mean"], result_b["faithscore"]["overall"]["mean"]),
        ("faithscore_n", result_a["faithscore"]["overall"]["n"], result_b["faithscore"]["overall"]["n"]),
        ("tokens_gerados_medio", result_a["mean_new_tokens"], result_b["mean_new_tokens"]),
        ("latencia_vlm_s", cost_a["latency_seconds"]["mean"], cost_b["latency_seconds"]["mean"]),
        ("latencia_patchcore_s", 0, round(patchcore_seconds, 2)),
        ("pico_rss_mb", cost_a["peak_memory_megabytes"], cost_b["peak_memory_megabytes"]),
    ]
    return rows


def main():
    result_a, result_b = load(FACTUALITY_A), load(FACTUALITY_B)
    cost_a, cost_b = load(COST_A), load(COST_B)
    evidences = load(EVIDENCES)
    pairs = pair_responses(result_a, result_b)
    with open(TABLE_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["medida", "configuracao_A", "configuracao_B"])
        writer.writerows(table_rows(result_a, result_b, cost_a, cost_b, evidences))
    paired = {"overall": mcnemar(pairs)}
    for question in ("q1_presence", "q2_location", "q3_type"):
        paired[question] = mcnemar([pair for pair in pairs if pair[0]["question_id"] == question])
    paired["location_copy"] = copied_location(pairs, evidences)
    with open(PAIRED_PATH, "w", encoding="utf-8") as handle:
        json.dump(paired, handle, ensure_ascii=False, indent=2)
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
