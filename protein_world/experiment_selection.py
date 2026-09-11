import random

import numpy as np

from helpers import write_to_log
from models import Model
from protein_world.attribute import ProteinAttribute, sample_protein_unordered
from protein_world.config import Config
from protein_world.experiment import Experiment


def _entropy(probs: dict[str, float]) -> float:
    return -sum(p * np.log(p) for p in probs.values() if p > 0.0)


def _sample_rollout_instances(
    protein_names: list[str],
    attributes: list[ProteinAttribute],
    prior: dict[str, float],
    rng: np.random.Generator,
    num_samples: int,
) -> list[dict[str, str]]:
    # Normalize to guard against floating-point drift in the prior
    prob_array = np.array([prior[p] for p in protein_names], dtype=float)
    prob_array /= prob_array.sum()

    instances = []
    for _ in range(num_samples):
        # Draw a protein class from the prior, then a full instance from that class
        idx = int(rng.choice(len(protein_names), p=prob_array))
        instances.append(sample_protein_unordered(protein_names[idx], attributes))
    return instances


def _compute_eig(
    experiment: Experiment,
    protein_names: list[str],
    prior: dict[str, float],
    prior_entropy: float,
    instances: list[dict[str, str]],
) -> float:
    total_posterior_entropy = 0.0
    valid_samples = 0

    for instance in instances:
        y = float(experiment.sample(instance))

        # Bayesian update: posterior ∝ prior × likelihood
        unnorm = {p: prior[p] * experiment.likelihood(y, p) for p in protein_names}
        total = sum(unnorm.values())
        if total < 1e-300:
            # Degenerate outcome: effectively impossible under every protein class.
            # Skip rather than divide by zero; rare for well-specified worlds.
            continue

        posterior = {p: v / total for p, v in unnorm.items()}
        total_posterior_entropy += _entropy(posterior)
        valid_samples += 1

    if valid_samples == 0:
        return 0.0

    return prior_entropy - total_posterior_entropy / valid_samples


def select_experiment_eig(
    protein_names: list[str],
    attributes: list[ProteinAttribute],
    prior: dict[str, float],
    history: dict[str, str],
    experiments_by_name: dict[str, Experiment],
    config: Config,
    rng: np.random.Generator,
) -> Experiment:
    available = {k: v for k, v in experiments_by_name.items() if k not in history}

    prior_entropy = _entropy(prior)
    instances = _sample_rollout_instances(protein_names, attributes, prior, rng, config.num_eig_samples)

    eigs = {
        name: _compute_eig(exp, protein_names, prior, prior_entropy, instances)
        for name, exp in available.items()
    }

    write_to_log(
        f"EIG estimates: { {k: f'{v:.4f}' for k, v in eigs.items()} }\n"
    )

    best = max(eigs, key=eigs.get)
    return experiments_by_name[best]


def select_experiment_random(
    model: Model,
    world_building_string: str,
    probabilities_chatcompletion: dict[str, float],
    history_questioner: dict[str, str],
    experiments_by_name: dict[str, Experiment],
    config: Config,
) -> Experiment:
    write_to_log("Randomly determining an experiment to run\n")
    available = [k for k in experiments_by_name.keys() if k not in history_questioner]
    return experiments_by_name[random.choice(available)]