from dataclasses import dataclass

from helpers import BaseConfig, base_config_kwargs


@dataclass(kw_only=True)
class Config(BaseConfig):
    answer_temperature: float
    scenarios: list[str]
    elicitation_modes: list[str]


def load_config(raw: dict) -> Config:
    return Config(
        **base_config_kwargs(raw),
        answer_temperature = raw["answer_temperature"],
        scenarios = raw["scenarios"],
        elicitation_modes = raw["elicitation_modes"],
    )
