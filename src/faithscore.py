import re

import torch

from src.vlm import prepare_inputs

ANALYTIC_MARKERS = (
    "likely",
    "possibly",
    "probably",
    "perhaps",
    "maybe",
    "might",
    "could be",
    "appears to",
    "seems",
    "suggest",
    "indicat",
    "atmosphere",
    "enjoying",
)
VERIFIER_TEMPLATE = (
    "Statement: {claim}\n"
    "Does the image support this statement? Answer with only yes or no."
)
VERIFIER_THRESHOLD = 0.5
CLAUSE_SPLIT = re.compile(r"(?:;|,\s+and\s+|,\s+but\s+|\s+and\s+(?=there\s|it\s|the\s|a\s|an\s))", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_CLAIM_WORDS = 3


def split_sentences(answer):
    return [part.strip() for part in SENTENCE_SPLIT.split(answer.strip()) if part.strip()]


def is_analytic(text):
    lowered = text.lower()
    return any(marker in lowered for marker in ANALYTIC_MARKERS)


def decompose(answer):
    claims = []
    for sentence in split_sentences(answer):
        if is_analytic(sentence):
            continue
        for clause in CLAUSE_SPLIT.split(sentence):
            clause = clause.strip(" .,!?")
            if len(clause.split()) >= MIN_CLAIM_WORDS:
                claims.append(clause)
    return claims


def token_ids(processor, word):
    tokenizer = processor.tokenizer
    variants = {word, word.capitalize(), f" {word}", f" {word.capitalize()}"}
    return sorted({tokenizer.encode(item, add_special_tokens=False)[0] for item in variants})


def verify_claim(processor, model, device, image_path, claim):
    prompt = VERIFIER_TEMPLATE.format(claim=claim)
    processor.image_processor.do_image_splitting = False
    inputs = prepare_inputs(processor, device, image_path, prompt)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    yes_score = torch.logsumexp(logits[token_ids(processor, "yes")], dim=0)
    no_score = torch.logsumexp(logits[token_ids(processor, "no")], dim=0)
    return float(torch.sigmoid(yes_score - no_score))


def score_claims(probabilities, threshold=VERIFIER_THRESHOLD):
    if not probabilities:
        return None
    return sum(value >= threshold for value in probabilities) / len(probabilities)


def faithscore(processor, model, device, image_path, answer, threshold=VERIFIER_THRESHOLD):
    claims = decompose(answer)
    checked = [
        {"claim": claim, "p_yes": round(verify_claim(processor, model, device, image_path, claim), 4)}
        for claim in claims
    ]
    return {
        "claims": checked,
        "n_claims": len(checked),
        "faithscore": score_claims([item["p_yes"] for item in checked], threshold),
    }
