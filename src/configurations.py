import json
from pathlib import Path

from src.prompts import build_prompt_a, build_prompt_b

EVIDENCES_PATH = Path("experiments/exp002b_evidencias_subconjunto.json")


def load_evidences(path=EVIDENCES_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prompt_builder_a(question, image):
    return build_prompt_a(question)


def make_prompt_builder_b():
    evidences = load_evidences()

    def prompt_builder_b(question, image):
        return build_prompt_b(question, evidences[image["id"]])

    return prompt_builder_b


def get_prompt_builder(configuration):
    return prompt_builder_a if configuration == "A" else make_prompt_builder_b()
