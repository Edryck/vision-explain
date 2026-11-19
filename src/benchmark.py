import argparse
import hashlib
import json
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.compare_configurations import mcnemar
from src.evaluate_factuality import summarize
from src.faithscore import decompose
from src.multicategory import load_json, save_json, score_response
from src.multicategory_metrics import image_consistency
from src.prompts import build_messages, build_prompt_a

# "env" diz em qual transformers o modelo roda: o código remoto do InternVL quebra no 5.x e o Qwen3.5 só existe nele.
MODELS = {
    "SmolVLM-Instruct": {"repo": "HuggingFaceTB/SmolVLM-Instruct", "backend": "native", "env": "latest", "longest_edge": 1152},
    "SmolVLM2-2.2B-Instruct": {"repo": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "backend": "native", "env": "latest", "longest_edge": 1152},
    "Qwen3.5-0.8B": {"repo": "Qwen/Qwen3.5-0.8B", "backend": "native", "env": "latest", "template_kwargs": {"enable_thinking": False}},
    "Qwen3-VL-2B-Instruct": {"repo": "Qwen/Qwen3-VL-2B-Instruct", "backend": "native", "env": "latest"},
    "Qwen3-VL-4B-Instruct": {"repo": "Qwen/Qwen3-VL-4B-Instruct", "backend": "native", "env": "latest"},
    "InternVL3_5-2B": {"repo": "OpenGVLab/InternVL3_5-2B", "backend": "internvl", "env": "legacy"},
    "InternVL3_5-2B-Instruct": {"repo": "OpenGVLab/InternVL3_5-2B-Instruct", "backend": "internvl", "env": "legacy"},
}
BASELINE = "SmolVLM-Instruct"
LEGACY_TRANSFORMERS = "4.57.1"
SUBSETS = {
    "bottle": Path("data/subset_avaliacao.json"),
    "multiclass": Path("experiments/exp007_multicategoria/subset.json"),
}
BENCH_DIR = Path("experiments/bench")
QUESTIONS = ("q1_presence", "q2_location", "q3_type")
MAX_NEW_TOKENS = 128
INTERNVL_TILE = 448
INTERNVL_MAX_TILES = 12
BOOTSTRAP_RESAMPLES = 2000


def pick_device():
    if not torch.cuda.is_available():
        return "cpu", torch.float32
    # T4 não tem bf16 nativo; emular deixa tudo lento.
    return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(key):
    spec = MODELS[key]
    device, dtype = pick_device()
    loaded = {"key": key, "spec": spec, "device": device, "dtype": dtype}
    if spec["backend"] == "internvl":
        from transformers import AutoModel, AutoTokenizer

        loaded["model"] = AutoModel.from_pretrained(
            spec["repo"], dtype=dtype, low_cpu_mem_usage=True, use_flash_attn=False, trust_remote_code=True
        ).eval().to(device)
        loaded["tokenizer"] = AutoTokenizer.from_pretrained(spec["repo"], trust_remote_code=True, use_fast=False)
        return loaded
    from transformers import AutoModelForImageTextToText, AutoProcessor

    loaded["processor"] = AutoProcessor.from_pretrained(spec["repo"])
    if "longest_edge" in spec:
        # Mesmo ajuste do exp003: o padrão de 1536 só amplia as fotos do MVTec.
        loaded["processor"].image_processor.size = {"longest_edge": spec["longest_edge"]}
    loaded["model"] = AutoModelForImageTextToText.from_pretrained(spec["repo"], dtype=dtype).eval().to(device)
    return loaded


def internvl_tiles(image):
    import torchvision.transforms as T

    # Mesmo recorte dinâmico do model card: escolhe a grade com proporção mais próxima da imagem.
    ratios = sorted(
        {(i, j) for i in range(1, INTERNVL_MAX_TILES + 1) for j in range(1, INTERNVL_MAX_TILES + 1) if i * j <= INTERNVL_MAX_TILES},
        key=lambda ratio: ratio[0] * ratio[1],
    )
    aspect = image.width / image.height
    best, best_diff = (1, 1), float("inf")
    for ratio in ratios:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff:
            best, best_diff = ratio, diff
        elif diff == best_diff and image.width * image.height > 0.5 * INTERNVL_TILE**2 * ratio[0] * ratio[1]:
            best = ratio
    resized = image.resize((INTERNVL_TILE * best[0], INTERNVL_TILE * best[1]))
    tiles = [
        resized.crop((col * INTERNVL_TILE, row * INTERNVL_TILE, (col + 1) * INTERNVL_TILE, (row + 1) * INTERNVL_TILE))
        for row in range(best[1])
        for col in range(best[0])
    ]
    if len(tiles) > 1:
        tiles.append(image.resize((INTERNVL_TILE, INTERNVL_TILE)))
    transform = T.Compose([
        T.Resize((INTERNVL_TILE, INTERNVL_TILE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return torch.stack([transform(tile) for tile in tiles])


def generate(loaded, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    model, device, dtype = loaded["model"], loaded["device"], loaded["dtype"]
    if loaded["spec"]["backend"] == "internvl":
        pixel_values = internvl_tiles(image).to(device, dtype)
        config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
        answer = model.chat(loaded["tokenizer"], pixel_values, f"<image>\n{prompt}", config).strip()
        # chat() não devolve o tamanho do prompt; o custo de imagem fica em n_tiles.
        new_tokens = len(loaded["tokenizer"].encode(answer, add_special_tokens=False))
        return {"answer": answer, "prompt_tokens": None, "new_tokens": new_tokens, "n_tiles": int(pixel_values.shape[0])}
    processor = loaded["processor"]
    text = processor.apply_chat_template(
        build_messages(prompt), add_generation_prompt=True, **loaded["spec"].get("template_kwargs", {})
    )
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_tokens = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_tokens:]
    answer = processor.decode(new_tokens, skip_special_tokens=True)
    # Qwen3.5 sem raciocínio ainda pode fechar um bloco <think> vazio.
    answer = answer.split("</think>")[-1].strip()
    return {"answer": answer, "prompt_tokens": int(prompt_tokens), "new_tokens": int(new_tokens.shape[0]), "n_tiles": None}


def responses_path(subset_name, key):
    return BENCH_DIR / subset_name / f"{key}.json"


def log_line(tag, data):
    print(tag, json.dumps(data, ensure_ascii=False), flush=True)


def log_text(log_path):
    # Aceita o log do Kaggle (lista de {"data": ...}), o .ipynb salvo com saídas ou texto copiado.
    raw = Path(log_path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, list):
        return "".join(item.get("data", "") for item in parsed)
    outputs = [o for cell in parsed.get("cells", []) for o in cell.get("outputs", [])]
    return "".join("".join(o.get("text", [])) for o in outputs)


def recover(log_path):
    rows = defaultdict(list)
    for line in log_text(log_path).splitlines():
        if line.startswith("BENCH_ROW "):
            row = json.loads(line[len("BENCH_ROW "):])
            rows[(row.pop("subset"), row.pop("model"))].append(row)
    for (subset_name, key), found in rows.items():
        path = responses_path(subset_name, key)
        record = load_json(path) if path.exists() else {
            "model": key,
            "repo": MODELS[key]["repo"],
            "configuration": "A",
            "recovered_from_log": True,
            "environment": {},
            "peak_memory_gb": None,
            "subset": SUBSETS[subset_name].as_posix(),
            "n_expected": len(load_json(SUBSETS[subset_name])["images"]) * len(QUESTIONS),
            "responses": [],
        }
        known = {(r["image_id"], r["question_id"]) for r in record["responses"]}
        added = [r for r in found if (r["image_id"], r["question_id"]) not in known]
        record["responses"] += added
        record["n_responses"] = len(record["responses"])
        save_json(record, path)
        print(subset_name, key, "recuperadas:", len(added), "total:", record["n_responses"])


def run(key, subset_name, limit=None):
    import transformers

    subset = load_json(SUBSETS[subset_name])
    images = subset["images"][:limit] if limit else subset["images"]
    path = responses_path(subset_name, key)
    previous = load_json(path) if path.exists() else {}
    responses = previous.get("responses", [])
    done = {(item["image_id"], item["question_id"]) for item in responses}
    loaded = load_model(key)
    if loaded["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Tudo vai também para o stdout: se a sessão cair antes de salvar, o log do Kaggle guarda os dados (ver recover).
    log_line("BENCH_META", {
        "model": key,
        "repo": MODELS[key]["repo"],
        "subset": subset_name,
        "subset_sha256": hashlib.sha256(SUBSETS[subset_name].read_bytes()).hexdigest(),
        "n_images": len(images),
        "already_done": len(responses),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": str(loaded["dtype"]).replace("torch.", ""),
    })
    for index, image in enumerate(images, start=1):
        pending = [q for q in subset["questions"] if (image["id"], q["id"]) not in done]
        for question in pending:
            prompt = build_prompt_a(question["text"])
            started = time.perf_counter()
            result = generate(loaded, image["path"], prompt)
            row = {
                "image_id": image["id"],
                "image": image["path"],
                "category": image["category"],
                "label": image["label"],
                "defect_type": image["defect_type"],
                "question_id": question["id"],
                "question": question["text"],
                "prompt": prompt,
                **result,
                "seconds": round(time.perf_counter() - started, 3),
            }
            responses.append(row)
            log_line("BENCH_ROW", {"model": key, "subset": subset_name, **row})
        if not pending:
            continue
        peak = torch.cuda.max_memory_allocated() / 1024**3 if loaded["device"] == "cuda" else None
        record = {
            "model": key,
            "repo": MODELS[key]["repo"],
            "configuration": "A",
            "generation": {
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "device": loaded["device"],
                "dtype": str(loaded["dtype"]).replace("torch.", ""),
                "longest_edge": MODELS[key].get("longest_edge"),
                "internvl_max_tiles": INTERNVL_MAX_TILES if MODELS[key]["backend"] == "internvl" else None,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            # Com retomada em outra sessão, guarda o maior pico visto.
            "peak_memory_gb": max(filter(None, [peak, previous.get("peak_memory_gb")]), default=None),
            "subset": SUBSETS[subset_name].as_posix(),
            "n_expected": len(subset["images"]) * len(subset["questions"]),
            "n_responses": len(responses),
            "responses": responses,
        }
        save_json(record, path)
        if index % 10 == 0 or index == len(images):
            print(key, len(responses), record["n_expected"], flush=True)


def add_faithscore(subset_name, key):
    from src.faithscore import faithscore
    from src.vlm import load_vlm

    # Juiz fixo (SmolVLM-Instruct) para todos, senão a métrica muda junto com o modelo.
    path = BENCH_DIR / subset_name / "results" / f"faithscore_{key}.json"
    scores = load_json(path) if path.exists() else {}
    processor, model, device, _ = load_vlm()
    for index, response in enumerate(load_json(responses_path(subset_name, key))["responses"], start=1):
        item_key = f"{response['image_id']}|{response['question_id']}"
        if item_key not in scores:
            scores[item_key] = faithscore(processor, model, device, response["image"], response["answer"])["faithscore"]
        if index % 50 == 0:
            save_json(scores, path)
    save_json(scores, path)


def score_model(subset_name, key, images):
    record = load_json(responses_path(subset_name, key))
    faith_path = BENCH_DIR / subset_name / "results" / f"faithscore_{key}.json"
    faith = load_json(faith_path) if faith_path.exists() else None
    items = []
    for response in record["responses"]:
        image = images[response["image_id"]]
        correct = score_response(response, image)
        item = {
            "image_id": response["image_id"],
            "category": image["category"],
            "label": image["label"],
            "defect_type": image["defect_type"] or "good",
            "question_id": response["question_id"],
            "answer": response["answer"],
            "new_tokens": response["new_tokens"],
            "seconds": response["seconds"],
            "n_words": len(response["answer"].split()),
            "n_claims": len(decompose(response["answer"])),
            "ground_truth_correct": None if correct is None else int(correct),
        }
        if faith is not None:
            item["faithscore"] = faith.get(f"{response['image_id']}|{response['question_id']}")
        items.append(item)
    return record, items


def bootstrap_ci(items, seed=42):
    # Reamostra imagens, não respostas: as 3 perguntas da mesma imagem não são independentes.
    by_image = defaultdict(list)
    for item in items:
        if item["ground_truth_correct"] is not None:
            by_image[item["image_id"]].append(item["ground_truth_correct"])
    if not by_image:
        return [None, None]
    groups = list(by_image.values())
    rng = random.Random(seed)
    estimates = [
        np.mean([value for group in rng.choices(groups, k=len(groups)) for value in group])
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def model_summary(record, items, baseline_items):
    complete = defaultdict(set)
    for item in items:
        complete[item["image_id"]].add(item["question_id"])
    # Imagem com resposta faltando (execução interrompida) quebraria a checagem de consistência.
    consistency = image_consistency([item for item in items if len(complete[item["image_id"]]) == len(QUESTIONS)])
    summary = {
        "repo": record["repo"],
        "n_responses": len(items),
        "n_expected": record["n_expected"],
        "accuracy": summarize([item["ground_truth_correct"] for item in items]),
        "accuracy_ci95": bootstrap_ci(items),
        "by_question": {q: summarize([i["ground_truth_correct"] for i in items if i["question_id"] == q]) for q in QUESTIONS},
        "by_category": {
            category: summarize([i["ground_truth_correct"] for i in items if i["category"] == category])
            for category in sorted({i["category"] for i in items})
        },
        "inconsistent_images": summarize(list(consistency.values())),
        "new_tokens": summarize([i["new_tokens"] for i in items]),
        "n_words": summarize([i["n_words"] for i in items]),
        "zero_claim_rate": round(sum(i["n_claims"] == 0 for i in items) / len(items), 4),
        "seconds_per_answer": summarize([i["seconds"] for i in items]),
        "peak_memory_gb": record.get("peak_memory_gb"),
        "environment": record["environment"],
    }
    if items and "faithscore" in items[0]:
        summary["faithscore"] = summarize([i["faithscore"] for i in items])
    if baseline_items is not None:
        index = {(i["image_id"], i["question_id"]): i for i in baseline_items}
        # Par (baseline, modelo): only_b_correct conta acertos só do modelo novo.
        pairs = [(index[(i["image_id"], i["question_id"])], i) for i in items if (i["image_id"], i["question_id"]) in index]
        summary["vs_baseline"] = mcnemar(pairs)
    return summary


def report(subset_name):
    import pandas as pd

    subset = load_json(SUBSETS[subset_name])
    images = {image["id"]: image for image in subset["images"]}
    scored = {key: score_model(subset_name, key, images) for key in MODELS if responses_path(subset_name, key).exists()}
    baseline_items = scored[BASELINE][1] if BASELINE in scored else None
    summaries = {
        key: model_summary(record, items, None if key == BASELINE else baseline_items)
        for key, (record, items) in scored.items()
    }
    results_dir = BENCH_DIR / subset_name / "results"
    save_json({key: items for key, (_, items) in scored.items()}, results_dir / "respostas_pontuadas.json")
    save_json(summaries, results_dir / "resumo.json")
    rows = []
    for key, s in summaries.items():
        rows.append({
            "modelo": key,
            "respostas": f"{s['n_responses']}/{s['n_expected']}",
            "acerto": s["accuracy"]["mean"],
            "ic95_inf": s["accuracy_ci95"][0],
            "ic95_sup": s["accuracy_ci95"][1],
            "indefinidas": s["accuracy"]["n_undefined"],
            **{q: s["by_question"][q]["mean"] for q in QUESTIONS},
            "inconsistencia": s["inconsistent_images"]["mean"],
            "palavras": s["n_words"]["mean"],
            "s_por_resposta": s["seconds_per_answer"]["mean"],
            "pico_memoria_gb": None if s["peak_memory_gb"] is None else round(s["peak_memory_gb"], 2),
            "faithscore": s.get("faithscore", {}).get("mean"),
            "mcnemar_p_vs_baseline": s.get("vs_baseline", {}).get("exact_p_value"),
        })
    table = pd.DataFrame(rows).sort_values("acerto", ascending=False)
    table.to_csv(results_dir / "tabela_modelos.csv", index=False)
    by_category = pd.DataFrame({key: {c: v["mean"] for c, v in s["by_category"].items()} for key, s in summaries.items()})
    by_category.to_csv(results_dir / "tabela_por_categoria.csv")
    return table, by_category


def plot_results(subset_name, table, by_category):
    import matplotlib.pyplot as plt

    figs_dir = BENCH_DIR / subset_name / "results" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    table = table.sort_values("acerto")

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.5))
    errors = [table["acerto"] - table["ic95_inf"], table["ic95_sup"] - table["acerto"]]
    ax.barh(table["modelo"], table["acerto"], xerr=errors, color="#4C72B0", capsize=4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Acerto por gabarito (IC 95%, bootstrap por imagem)")
    ax.set_title(f"Configuração A: acerto geral ({subset_name})")
    fig.tight_layout()
    fig.savefig(figs_dir / "acerto_geral.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.set_index("modelo")[list(QUESTIONS)].plot.bar(ax=ax, rot=30)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Acerto")
    ax.set_title(f"Acerto por pergunta ({subset_name})")
    fig.tight_layout()
    fig.savefig(figs_dir / "acerto_por_pergunta.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = table["pico_memoria_gb"].astype(float).fillna(1) * 60
    ax.scatter(table["s_por_resposta"], table["acerto"], s=sizes, alpha=0.6)
    for _, row in table.iterrows():
        ax.annotate(row["modelo"], (row["s_por_resposta"], row["acerto"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Segundos por resposta (média)")
    ax.set_ylabel("Acerto")
    ax.set_title("Custo x acerto (tamanho do ponto = pico de memória)")
    fig.tight_layout()
    fig.savefig(figs_dir / "custo_acerto.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(table) + 1.5), sharey=True)
    axes[0].barh(table["modelo"], table["inconsistencia"], color="#DD8452")
    axes[0].set_title("Imagens com respostas contraditórias")
    axes[1].barh(table["modelo"], table["palavras"], color="#55A868")
    axes[1].set_title("Palavras por resposta")
    fig.tight_layout()
    fig.savefig(figs_dir / "consistencia_tamanho.png", dpi=150)
    plt.show()

    if len(by_category) > 1:
        ordered = by_category[table["modelo"][::-1]]
        fig, ax = plt.subplots(figsize=(1.2 * len(ordered.columns) + 3, 0.45 * len(ordered) + 2))
        heat = ax.imshow(ordered.values.astype(float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ordered.columns)), ordered.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(ordered)), ordered.index)
        for row in range(len(ordered)):
            for col in range(len(ordered.columns)):
                value = ordered.iat[row, col]
                if value is not None and not np.isnan(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.6 else "black")
        fig.colorbar(heat, ax=ax, label="Acerto")
        ax.set_title("Acerto por categoria e modelo")
        fig.tight_layout()
        fig.savefig(figs_dir / "acerto_por_categoria.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "faithscore", "recover"))
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--subset", choices=sorted(SUBSETS))
    parser.add_argument("--log", help="log do Kaggle, .ipynb com saídas ou texto com linhas BENCH_ROW")
    parser.add_argument("--limit", type=int, help="só as N primeiras imagens, para teste rápido")
    arguments = parser.parse_args()
    if arguments.command == "recover":
        recover(arguments.log)
    elif arguments.command == "run":
        run(arguments.model, arguments.subset, arguments.limit)
    else:
        add_faithscore(arguments.subset, arguments.model)
