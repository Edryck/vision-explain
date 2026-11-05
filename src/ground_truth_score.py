import re

import cv2
import numpy as np

from src.extract_evidence import GRID_NAMES

NEGATIVE_PATTERN = re.compile(r"^\s*no\b|\bno defects?\b|\bnot? (?:any|see|find)\b|\bthere (?:is|are) no\b", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r"^\s*yes\b", re.IGNORECASE)
HORIZONTAL_TERMS = {0: ("left",), 2: ("right",)}
# neck, cap e rim assumem a garrafa de lado, mas as fotos são de cima. Nenhuma resposta do exp003 usa esses termos.
VERTICAL_TERMS = {0: ("top", "upper", "neck", "cap", "rim"), 2: ("bottom", "lower", "base")}
CENTER_TERMS = ("center", "centre", "middle")
TYPE_TERMS = {
    "broken_large": ("broken", "break", "crack", "chip", "shatter", "fractur", "damage", "missing"),
    "broken_small": ("broken", "break", "crack", "chip", "shatter", "fractur", "damage", "missing"),
    "contamination": ("contamin", "dirt", "stain", "foreign", "debris", "particle", "spot", "residue", "smudge"),
}


def claims_no_defect(answer):
    return bool(NEGATIVE_PATTERN.search(answer))


def presence_prediction(answer):
    if POSITIVE_PATTERN.search(answer):
        return True
    if claims_no_defect(answer):
        return False
    return None


def mask_cells(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    height, width = mask.shape
    cells = set()
    for row in range(3):
        for col in range(3):
            block = mask[row * height // 3 : (row + 1) * height // 3, col * width // 3 : (col + 1) * width // 3]
            if np.any(block):
                cells.add((row, col))
    return cells


def answer_cells(answer):
    lowered = answer.lower()
    rows = {row for row, terms in VERTICAL_TERMS.items() if any(term in lowered for term in terms)}
    cols = {col for col, terms in HORIZONTAL_TERMS.items() if any(term in lowered for term in terms)}
    # "center" só preenche o eixo não citado: "upper center" continua na linha de cima.
    if any(term in lowered for term in CENTER_TERMS):
        if not rows:
            rows.add(1)
        if not cols:
            cols.add(1)
    if not rows and not cols:
        return set()
    # Termo em um só eixo, como "left", vale para toda a coluna.
    return {(row, col) for row in (rows or {0, 1, 2}) for col in (cols or {0, 1, 2})}


def score_presence(response):
    predicted = presence_prediction(response["answer"])
    if predicted is None:
        return None
    return predicted == (response["label"] == "anomalous")


def score_location(response, mask_path):
    if response["label"] == "normal":
        return claims_no_defect(response["answer"])
    if claims_no_defect(response["answer"]):
        return False
    cells = answer_cells(response["answer"])
    # Sem posição citada, o resultado é indefinido, não erro.
    if not cells:
        return None
    # Critério permissivo: basta tocar a máscara.
    return bool(cells & mask_cells(mask_path))


def score_type(response):
    if response["label"] == "normal":
        return claims_no_defect(response["answer"])
    lowered = response["answer"].lower()
    return any(term in lowered for term in TYPE_TERMS[response["defect_type"]])


def score_response(response, mask_path):
    scorers = {
        "q1_presence": lambda: score_presence(response),
        "q2_location": lambda: score_location(response, mask_path),
        "q3_type": lambda: score_type(response),
    }
    return scorers[response["question_id"]]()


def cell_name(cell):
    return GRID_NAMES[cell[0]][cell[1]]
