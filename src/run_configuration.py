import json
import platform
import time
from pathlib import Path

import torch
import transformers

from src.prompts import build_prompt_a
from src.vlm import LONGEST_EDGE, MAX_NEW_TOKENS, MODEL_ID, generate_answers_shared_image, load_vlm

SUBSET_PATH = Path("data/subset_avaliacao.json")
OUTPUT_A = Path("experiments/exp003_configA_completo.json")


def load_subset(path=SUBSET_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_existing(output_path):
    if not Path(output_path).exists():
        return []
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)["responses"]


def build_record(experiment_id, configuration, device, dtype, responses, expected):
    return {
        "experiment_id": experiment_id,
        "configuration": configuration,
        "model": MODEL_ID,
        "generation": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "batch_size": 1,
            "longest_edge": LONGEST_EDGE,
            "image_prefix_kv_cache": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "subset": SUBSET_PATH.as_posix(),
        "n_expected": expected,
        "n_responses": len(responses),
        "responses": responses,
    }


def save_record(record, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Grava em arquivo temporário porque replace é atômico: uma queda não corrompe o JSON.
    temporary = Path(output_path).with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    temporary.replace(output_path)


def run_configuration(experiment_id, configuration, prompt_builder, output_path):
    subset = load_subset()
    processor, model, device, dtype = load_vlm()
    responses = load_existing(output_path)
    # A retomada é por imagem: as 3 respostas compartilham o cache e são salvas juntas.
    finished = {item["image_id"] for item in responses}
    questions = subset["questions"]
    expected = len(subset["images"]) * len(questions)
    for image in subset["images"]:
        if image["id"] in finished:
            continue
        prompts = [prompt_builder(question["text"], image) for question in questions]
        started = time.perf_counter()
        results = generate_answers_shared_image(processor, model, device, image["path"], prompts)
        elapsed = time.perf_counter() - started
        for question, prompt, result in zip(questions, prompts, results):
            responses.append(
                {
                    "image_id": image["id"],
                    "image": image["path"],
                    "label": image["label"],
                    "defect_type": image["defect_type"],
                    "question_id": question["id"],
                    "question": question["text"],
                    "prompt": prompt,
                    "answer": result["answer"],
                    "prompt_tokens": result["prompt_tokens"],
                    "new_tokens": result["new_tokens"],
                    "image_batch_seconds": round(elapsed, 2),
                }
            )
        save_record(
            build_record(experiment_id, configuration, device, dtype, responses, expected),
            output_path,
        )
        print(len(responses), expected, image["id"], round(elapsed, 1), flush=True)
    return responses


def prompt_builder_a(question, image):
    return build_prompt_a(question)


if __name__ == "__main__":
    run_configuration("exp003", "A", prompt_builder_a, OUTPUT_A)
