import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from src.faithscore import VERIFIER_THRESHOLD, decompose, score_claims, verify_claim
from src.vlm import MODEL_ID, load_vlm

MANUAL_PATH = Path("experiments/exp001_avaliacao_manual.json")
OUTPUT_PATH = Path("experiments/exp004_validacao_metrica.json")


def correlation(function, first, second):
    if len(set(first)) < 2 or len(set(second)) < 2:
        return None
    return round(float(function(first, second)[0]), 4)


def claim_agreement(items, threshold):
    truth = [claim["verifiable"] for item in items for claim in item["claims"]]
    predicted = [claim["p_yes"] >= threshold for item in items for claim in item["claims"]]
    matrix = confusion_matrix(truth, predicted, labels=[False, True]).tolist()
    return {
        "n_claims": len(truth),
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "majority_baseline": round(max(np.mean(truth), 1 - np.mean(truth)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(truth, predicted)), 4),
        "confusion_rows_manual_cols_metric": matrix,
    }


def answer_agreement(items, key):
    pairs = [(item["verifiable_ratio"], item[key]) for item in items if item[key] is not None]
    manual = [pair[0] for pair in pairs]
    metric = [pair[1] for pair in pairs]
    return {
        "n_answers": len(pairs),
        "manual_mean": round(float(np.mean(manual)), 4),
        "metric_mean": round(float(np.mean(metric)), 4),
        "mean_absolute_error": round(float(np.mean(np.abs(np.array(manual) - np.array(metric)))), 4),
        "pearson": correlation(pearsonr, manual, metric),
        "spearman": correlation(spearmanr, manual, metric),
    }


def main():
    with open(MANUAL_PATH, encoding="utf-8") as handle:
        manual = json.load(handle)
    processor, model, device, _ = load_vlm()
    items = []
    for item in manual["items"]:
        # Afirmações manuais isolam o erro do verificador.
        claims = [
            {
                "text": claim["text"],
                "verifiable": claim["verifiable"],
                "p_yes": round(verify_claim(processor, model, device, item["image"], claim["text"]), 4),
            }
            for claim in item["claims"]
        ]
        # Afirmações automáticas medem também o erro da decomposição.
        automatic = [
            {"text": text, "p_yes": round(verify_claim(processor, model, device, item["image"], text), 4)}
            for text in decompose(item["answer"])
        ]
        items.append(
            {
                "id": item["id"],
                "image": item["image"],
                "question": item["question"],
                "verifiable_ratio": item["verifiable_ratio"],
                "claims": claims,
                "automatic_claims": automatic,
                "metric_manual_claims": score_claims([claim["p_yes"] for claim in claims]),
                "metric_automatic_claims": score_claims([claim["p_yes"] for claim in automatic]),
            }
        )
        print(item["id"], len(claims), len(automatic), flush=True)
    # Só diagnóstico: escolher o limiar aqui seria ajustar a métrica à própria referência.
    sweep = {
        str(value): claim_agreement(items, value)
        for value in (0.1, 0.3, 0.5, 0.7, 0.9)
    }
    result = {
        "experiment_id": "exp004",
        "week": 10,
        "title": "Validation of simplified FaithScore against exp001 manual evaluation",
        "verifier": {"model": MODEL_ID, "image_splitting": False, "threshold": VERIFIER_THRESHOLD},
        "manual_reference_status": manual["status"],
        "claim_level": claim_agreement(items, VERIFIER_THRESHOLD),
        "threshold_sweep_diagnostic": sweep,
        "answer_level_manual_claims": answer_agreement(items, "metric_manual_claims"),
        "answer_level_automatic_claims": answer_agreement(items, "metric_automatic_claims"),
        "items": items,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("claim_level", "answer_level_manual_claims", "answer_level_automatic_claims")}, indent=2))


if __name__ == "__main__":
    main()
