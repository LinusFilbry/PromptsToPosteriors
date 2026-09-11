from dataclasses import dataclass
from pathlib import Path

from helpers import BaseConfig, base_config_kwargs

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results-20Q"


@dataclass(kw_only=True)
class Config(BaseConfig):
    method_names: list[str]
    animals: list[str]
    batched_block_size: int
    generation_temperature_simple: float
    answer_temperature: float
    generation_temperature_diverse: float
    target_num_questions: int
    num_mc_samples: int
    max_num_samples: int
    min_num_samples: int
    threshold_rejection_probability: float


def load_config(raw: dict) -> Config:
    return Config(
        **base_config_kwargs(raw),
        method_names = raw["method_names"],
        animals = raw["animals"],
        batched_block_size = raw["batched_block_size"],
        generation_temperature_simple = raw["generation_temperature_simple"],
        answer_temperature = raw["answer_temperature"],
        generation_temperature_diverse = raw["generation_temperature_diverse"],
        target_num_questions = raw["target_num_questions"],
        num_mc_samples = raw["num_mc_samples"],
        max_num_samples = raw["max_num_samples"],
        min_num_samples = raw["min_num_samples"],
        threshold_rejection_probability = raw["threshold_rejection_probability"],
    )
