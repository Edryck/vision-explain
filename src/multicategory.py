import json
import random
import re
from pathlib import Path

from src.build_subset import QUESTIONS
from src.ground_truth_score import claims_no_defect, score_location, score_presence

MVTEC_ROOT = Path("data/mvtec_ad/mvtec-ad")
EXPERIMENT_ID = "exp007"
EXPERIMENT_DIR = Path("experiments/exp007_multicategoria")
SUBSET_PATH = EXPERIMENT_DIR / "subset.json"
EVIDENCES_PATH = EXPERIMENT_DIR / "evidencias_patchcore.json"
PATCHCORE_TEST_PATH = EXPERIMENT_DIR / "patchcore_teste_completo.json"
RESPONSES_PATHS = {
    "A": EXPERIMENT_DIR / "configA_respostas.json",
    "B": EXPERIMENT_DIR / "configB_respostas.json",
}
RESULTS_DIR = EXPERIMENT_DIR / "results"
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
SEED = 42
IMAGES_PER_CATEGORY = 50
GOOD_PER_CATEGORY = 10

# Dicionário escrito antes de ver as respostas e de propósito permissivo, igual ao do bottle.
TYPE_TERMS = {
    "broken_large": ("broken", "break", "crack", "chip", "shatter", "fractur", "damage", "missing"),
    "broken_small": ("broken", "break", "crack", "chip", "shatter", "fractur", "damage", "missing"),
    "contamination": ("contamin", "dirt", "stain", "foreign", "debris", "particle", "spot", "residue", "smudge"),
    "bent": ("bent", "bend", "deform", "twist", "kink", "warp"),
    "bent_wire": ("bent", "bend", "deform", "twist", "kink"),
    "bent_lead": ("bent", "bend", "deform", "twist", "kink"),
    "cable_swap": ("swap", "switched", "wrong color", "wrong order", "misplac", "mixed"),
    "cut": ("cut", "slice", "slit", "gash", "incision", "tear", "torn", "nick"),
    "cut_inner_insulation": ("cut", "slice", "slit", "gash", "tear", "torn", "nick", "expos"),
    "cut_outer_insulation": ("cut", "slice", "slit", "gash", "tear", "torn", "nick", "expos"),
    "cut_lead": ("cut", "short", "trim", "missing", "clipped"),
    "missing_cable": ("missing", "absent", "lack", "removed", "empty"),
    "missing_wire": ("missing", "absent", "lack", "removed", "empty"),
    "poke": ("poke", "puncture", "hole", "pierc", "dent", "prick"),
    "poke_insulation": ("poke", "puncture", "hole", "pierc", "dent", "prick"),
    "crack": ("crack", "fractur", "break", "broken", "split", "fissure"),
    "faulty_imprint": ("print", "text", "logo", "label", "marking", "letter", "ink", "writing"),
    "print": ("print", "text", "logo", "label", "marking", "letter", "ink", "writing"),
    "scratch": ("scratch", "scrape", "abrasion", "scuff"),
    "scratch_head": ("scratch", "scrape", "abrasion", "scuff"),
    "scratch_neck": ("scratch", "scrape", "abrasion", "scuff"),
    "squeeze": ("squeez", "deform", "dent", "compress", "crush", "squash", "misshap"),
    "squeezed_teeth": ("squeez", "deform", "compress", "crush", "bent", "teeth"),
    "color": ("color", "colour", "discolor", "stain", "spot", "tint", "dye"),
    "hole": ("hole", "puncture", "perforat", "gap", "opening"),
    "metal_contamination": ("metal", "contamin", "foreign", "debris", "particle", "wire"),
    "thread": ("thread", "fiber", "fibre", "hair", "string", "strand", "lint"),
    "broken": ("broken", "break", "crack", "damage", "missing", "gap"),
    "glue": ("glue", "adhesive", "residue", "stain", "smear"),
    "glue_strip": ("glue", "adhesive", "residue", "strip", "stripe", "smear"),
    "fold": ("fold", "crease", "wrinkl"),
    "flip": ("flip", "upside", "inverted", "reversed", "wrong side", "orientation", "turned"),
    "pill_type": ("wrong pill", "different pill", "pill type", "color", "speck", "mismatch"),
    "manipulated_front": ("manipulat", "deform", "damage", "bent", "flatten", "tip"),
    "thread_side": ("thread", "damage", "deform", "worn", "stripped"),
    "thread_top": ("thread", "damage", "deform", "worn", "stripped"),
    "gray_stroke": ("gray", "grey", "stroke", "streak", "stripe", "line"),
    "oil": ("oil", "stain", "liquid", "smear", "grease"),
    "rough": ("rough", "uneven", "texture", "bump", "coarse"),
    # "defect" ficou de fora: casaria com qualquer resposta afirmativa.
    "defective": ("bristle", "damage", "bent", "missing", "fray", "messy", "irregular", "deform", "splay"),
    "damaged_case": ("damage", "crack", "broken", "chip", "dent", "scratch"),
    "misplaced": ("misplac", "missing", "absent", "shift", "misalign", "rotat", "displac"),
    "liquid": ("liquid", "stain", "wet", "spill", "water", "drop"),
    "broken_teeth": ("teeth", "tooth", "broken", "missing", "gap"),
    "split_teeth": ("teeth", "tooth", "split", "separat", "gap", "open"),
    "fabric_border": ("fabric", "cloth", "tear", "torn", "fray", "thread"),
    "fabric_interior": ("fabric", "cloth", "tear", "torn", "fray", "hole", "thread"),
}
NONE_PATTERN = re.compile(r"^\s*none\b", re.IGNORECASE)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def image_record(category, defect_type, path):
    is_good = defect_type == "good"
    mask = MVTEC_ROOT / category / "ground_truth" / defect_type / f"{path.stem}_mask.png"
    return {
        "id": f"{category}_{defect_type}_{path.stem}",
        "path": path.as_posix(),
        "category": category,
        "label": "normal" if is_good else "anomalous",
        "defect_type": None if is_good else defect_type,
        "mask_path": None if is_good else mask.as_posix(),
    }


def sample_category(category, rng):
    test_dir = MVTEC_ROOT / category / "test"
    pools = {d.name: sorted(d.glob("*.png")) for d in sorted(test_dir.iterdir()) if d.is_dir()}
    good = sorted(rng.sample(pools.pop("good"), GOOD_PER_CATEGORY))
    for files in pools.values():
        rng.shuffle(files)
    # Round-robin entre os tipos: divide igual e, quando um tipo acaba, os outros cobrem a vaga.
    wanted = IMAGES_PER_CATEGORY - GOOD_PER_CATEGORY
    chosen = {name: [] for name in pools}
    while sum(map(len, chosen.values())) < wanted and any(pools.values()):
        for name, files in pools.items():
            if files and sum(map(len, chosen.values())) < wanted:
                chosen[name].append(files.pop())
    images = [image_record(category, "good", path) for path in good]
    for name, files in chosen.items():
        images += [image_record(category, name, path) for path in sorted(files)]
    return images


def build_subset():
    rng = random.Random(SEED)
    images = []
    for category in CATEGORIES:
        images += sample_category(category, rng)
    return {
        "dataset": "MVTec AD",
        "experiment_id": EXPERIMENT_ID,
        "categories": CATEGORIES,
        "seed": SEED,
        "images_per_category": IMAGES_PER_CATEGORY,
        "good_per_category": GOOD_PER_CATEGORY,
        "questions": QUESTIONS,
        "images": images,
    }


def region_cells(regions, height, width):
    # Célula da grade 3x3 tocada por cada caixa, no mesmo recorte usado em mask_cells.
    cells = set()
    for x_min, y_min, x_max, y_max in (region["bbox"] for region in regions):
        for row in range(3):
            for col in range(3):
                if x_min < (col + 1) * width // 3 and x_max > col * width // 3 and y_min < (row + 1) * height // 3 and y_max > row * height // 3:
                    cells.add((row, col))
    return cells


def says_no_defect(answer):
    # "None." aparece em q3 e o regex do bottle não pegava.
    return claims_no_defect(answer) or bool(NONE_PATTERN.search(answer))


def type_terms(category, defect_type):
    if defect_type != "combined":
        return TYPE_TERMS[defect_type]
    # "combined" mistura defeitos da categoria, então qualquer tipo dela vale.
    names = [d.name for d in (MVTEC_ROOT / category / "test").iterdir() if d.name not in ("good", "combined")]
    return tuple(term for name in names for term in TYPE_TERMS[name])


def score_type(response, category):
    if response["label"] == "normal":
        return says_no_defect(response["answer"])
    lowered = response["answer"].lower()
    return any(term in lowered for term in type_terms(category, response["defect_type"]))


def score_response(response, image):
    if response["question_id"] == "q1_presence":
        return score_presence(response)
    if response["question_id"] == "q2_location":
        return score_location(response, image["mask_path"])
    return score_type(response, image["category"])


if __name__ == "__main__":
    subset = build_subset()
    save_json(subset, SUBSET_PATH)
    print(len(subset["images"]), SUBSET_PATH)
