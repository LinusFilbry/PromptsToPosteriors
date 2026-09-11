from dataclasses import dataclass
from pathlib import Path

from helpers import BaseConfig, base_config_kwargs

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results-proteins"


@dataclass(kw_only=True)
class Config(BaseConfig):
    generation_temperature_simple: float
    answer_temperature: float
    num_worlds_per_run: int
    proteins_per_world: int
    num_protein_classes: int
    num_attr: int
    pre_req_list: list[list[int]]
    num_experiments: int
    do_showcase_world: bool
    exp_type: str
    exp_noise: float
    num_mc_samples: int
    experiment_selection: str
    num_eig_samples: int


def load_config(raw: dict) -> Config:
    return Config(
        **base_config_kwargs(raw),
        generation_temperature_simple = raw["generation_temperature_simple"],
        answer_temperature = raw["answer_temperature"],
        num_worlds_per_run = raw["num_worlds_per_run"],
        proteins_per_world = raw["proteins_per_world"],
        num_protein_classes = raw["num_protein_classes"],
        num_attr = raw["num_attr"],
        pre_req_list = raw["pre_req_list"],
        num_experiments = raw["num_experiments"],
        do_showcase_world = raw["do_showcase_world"],
        exp_type = raw["exp_type"],
        exp_noise = raw["exp_noise"],
        num_mc_samples = raw["num_mc_samples"],
        experiment_selection = raw["experiment_selection"],
        num_eig_samples = raw["num_eig_samples"],
    )
