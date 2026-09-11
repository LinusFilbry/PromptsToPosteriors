import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class BaseConfig:
    task: str = ""
    seed: int = 42
    model: str = ""
    model_runtime: dict


# shared Config across tasks
def base_config_kwargs(raw: dict) -> dict[str, Any]:
    return dict(
        task=raw.get("task", ""),
        seed=raw.get("seed", 42),
        model=raw.get("model", ""),
        model_runtime=raw["model_runtime"],
    )


LOG_PATH = "logs/logfile.txt"


def set_log_file(name: str) -> None:
    global LOG_PATH
    LOG_PATH = f"logs/{name}.txt"


def write_to_log(message: str) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message)


# prompts ask to generate collection of entities, one on each line --> convert the returned string to an array
def convert_string_to_array(response):
    return [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]


def convert_to_prompt_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def normalize_prob_dict(probabilities: dict[str, float]) -> dict[str, float]:
    prob_sum = sum(probabilities.values())
    if prob_sum == 0:
        return probabilities
    for belief in probabilities.keys():
        probabilities[belief] /= prob_sum
    return probabilities


def smooth_prob_dict(probabilities: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    return {key: max(value, eps) for key, value in probabilities.items()}


def calculate_kl_divergence(
        probabilities_oracle: dict[str, float],
        probabilities_LLM: dict[str, float],
) -> float:
    posterior_oracle = normalize_prob_dict(smooth_prob_dict(probabilities_oracle))
    posterior_LLM = normalize_prob_dict(smooth_prob_dict(probabilities_LLM))

    KL = 0.0
    for category in posterior_oracle.keys():
        p = posterior_oracle[category]
        q = posterior_LLM[category]
        KL += p * math.log(p / q)

    return KL
