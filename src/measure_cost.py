import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import torch

from src.prompts import build_prompt_a
from src.run_configuration import load_subset
from src.vlm import MAX_NEW_TOKENS, MODEL_ID, generate_answer, load_vlm

N_IMAGES = 4
N_REPEATS = 3
QUESTION_INDEX = 0


def select_images(subset):
    groups = {}
    for image in subset["images"]:
        groups.setdefault(image["defect_type"], image)
    return list(groups.values())[:N_IMAGES]


def peak_memory_megabytes(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    # No Linux, ru_maxrss vem em KB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def measure(configuration, prompt_builder, output_path):
    subset = load_subset()
    question = subset["questions"][QUESTION_INDEX]
    images = select_images(subset)
    started = time.perf_counter()
    processor, model, device, dtype = load_vlm()
    load_seconds = time.perf_counter() - started
    # Aquecimento fora da medição: a primeira execução inclui alocações iniciais.
    generate_answer(processor, model, device, images[0]["path"], prompt_builder(question["text"], images[0]))
    runs = []
    for image in images:
        for repeat in range(N_REPEATS):
            prompt = prompt_builder(question["text"], image)
            started = time.perf_counter()
            # Sem cache de propósito: mede o custo de uma pergunta isolada.
            result = generate_answer(processor, model, device, image["path"], prompt)
            elapsed = time.perf_counter() - started
            runs.append(
                {
                    "image_id": image["id"],
                    "repeat": repeat,
                    "seconds": round(elapsed, 3),
                    "prompt_tokens": result["prompt_tokens"],
                    "new_tokens": result["new_tokens"],
                }
            )
            print(image["id"], repeat, round(elapsed, 1), flush=True)
    seconds = [run["seconds"] for run in runs]
    report = {
        "configuration": configuration,
        "model": MODEL_ID,
        "precision": str(dtype).replace("torch.", ""),
        "batch_size": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": device,
        "hardware": {"cpu": cpu_name(), "threads": torch.get_num_threads()},
        "vram_note": "No GPU available; peak process RSS is reported instead of VRAM." if device == "cpu" else None,
        "protocol": {
            "warmup_runs": 1,
            "images": len(images),
            "repeats_per_image": N_REPEATS,
            "question_id": question["id"],
        },
        "model_load_seconds": round(load_seconds, 2),
        "latency_seconds": {
            "mean": round(statistics.mean(seconds), 2),
            "std": round(statistics.stdev(seconds), 2),
            "min": min(seconds),
            "max": max(seconds),
        },
        "peak_memory_megabytes": round(peak_memory_megabytes(device), 1),
        "runs": runs,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def prompt_builder_a(question, image):
    return build_prompt_a(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/results/custo_configA.json")
    arguments = parser.parse_args()
    outcome = measure("A", prompt_builder_a, arguments.output)
    print(outcome["latency_seconds"], outcome["peak_memory_megabytes"])
