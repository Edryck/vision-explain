from src.multicategory import EVIDENCES_PATH, EXPERIMENT_ID, RESPONSES_PATHS, SUBSET_PATH, load_json
from src.prompts import build_prompt_b
from src.run_configuration import run_configuration

if __name__ == "__main__":
    evidences = load_json(EVIDENCES_PATH)

    def prompt_builder_b(question, image):
        return build_prompt_b(question, evidences[image["id"]])

    run_configuration(EXPERIMENT_ID, "B", prompt_builder_b, RESPONSES_PATHS["B"], SUBSET_PATH)
