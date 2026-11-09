from src.configurations import prompt_builder_a
from src.multicategory import EXPERIMENT_ID, RESPONSES_PATHS, SUBSET_PATH
from src.run_configuration import run_configuration

if __name__ == "__main__":
    run_configuration(EXPERIMENT_ID, "A", prompt_builder_a, RESPONSES_PATHS["A"], SUBSET_PATH)
