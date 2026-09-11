import numpy as np

from helpers import calculate_kl_divergence, write_to_log
from models import Model
from protein_world.attribute import sample_protein_unordered
from protein_world.config import Config, RESULTS_DIR
from protein_world.experiment_selection import select_experiment_eig, select_experiment_random
from protein_world.probabilities import sample_protein_probabilities_chatcompletion
from protein_world.world_setup import protein_world_setup


def protein_classification(model: Model, config: Config, world_iteration: int):
    rng = np.random.default_rng(config.seed + world_iteration)

    # setup protein world - shared by both the LLM-driven and oracle-driven runs below
    protein_names, attributes, experiments, world_building_string = protein_world_setup(model, config, world_iteration)
    experiments_by_name = {exp.name: exp for exp in experiments}

    selected_proteins = rng.choice(
        protein_names,
        size=config.proteins_per_world,
        replace=False
    )

    protein_vals = [sample_protein_unordered(protein, attributes) for protein in selected_proteins]

    uniform_prior = {prot_name: 1.0 / len(protein_names) for prot_name in protein_names}

    llm_distribution_data, llm_target_prob_by_round = run_llm_driven(
        model, protein_names, attributes, experiments_by_name, protein_vals,
        world_building_string, uniform_prior, config, rng,
    )
    oracle_target_prob_by_round = run_oracle_driven(
        protein_names, attributes, experiments_by_name, protein_vals, uniform_prior, config, rng,
    )

    llm_distribution_accuracy = [float(np.mean(round_probs)) for round_probs in llm_target_prob_by_round]
    oracle_distribution_accuracy = [float(np.mean(round_probs)) for round_probs in oracle_target_prob_by_round]

    file_tag = f"{config.model}_{config.seed}_{world_iteration}_{config.exp_type}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / f"{file_tag}_kl-trace.npy", np.array(llm_distribution_data))
    np.save(RESULTS_DIR / f"{file_tag}_llm-driven_accuracy.npy", np.array(llm_distribution_accuracy))
    np.save(RESULTS_DIR / f"{file_tag}_oracle-driven_accuracy.npy", np.array(oracle_distribution_accuracy))


def run_llm_driven(
        model: Model,
        protein_names: list[str],
        attributes,
        experiments_by_name: dict,
        protein_vals: list[dict[str, str]],
        world_building_string: str,
        uniform_prior: dict[str, float],
        config: Config,
        rng: np.random.Generator,
):
    llm_distribution_data = []
    llm_target_prob_by_round = [[] for _ in range(config.num_experiments)]

    for protein in protein_vals:
        write_to_log(f"[LLM-driven run] Starting on protein: {protein}\n\n\n\n")

        probabilities_oracle = dict(uniform_prior)
        probabilities_chatcompletion = dict(uniform_prior)
        history = {}

        for i in range(config.num_experiments):
            write_to_log(f"Round {i+1}\n\n")

            if config.experiment_selection == "eig":
                executed_experiment = select_experiment_eig(
                    protein_names, attributes, probabilities_chatcompletion, history, experiments_by_name, config, rng
                )
            else:
                executed_experiment = select_experiment_random(
                    model, world_building_string, probabilities_chatcompletion, history, experiments_by_name, config
                )
            experiment_outcome = executed_experiment.sample(protein)
            history = history | {executed_experiment.name: str(experiment_outcome)}
            write_to_log(f"Executed experiment: {executed_experiment.name}, answer: {experiment_outcome}\n\n")

            probabilities_oracle = executed_experiment.posterior(experiment_outcome, protein_names, probabilities_oracle)
            probabilities_chatcompletion = sample_protein_probabilities_chatcompletion(
                world_building_string, protein_names, history, uniform_prior, model, config
            )

            LLM_target_prob = probabilities_chatcompletion[protein["Protein"]]
            oracle_target_prob = probabilities_oracle[protein["Protein"]]
            KL_chatcompletion = calculate_kl_divergence(probabilities_oracle, probabilities_chatcompletion)

            if config.do_showcase_world:
                write_to_log(f"\nFull distribution (LLM): {format_prob_distribution(probabilities_chatcompletion)}\n"
                             f"Full distribution (oracle): {format_prob_distribution(probabilities_oracle)}\n"
                             f"KL: {KL_chatcompletion:.3f}\n\n\n")
            else:
                write_to_log(f"\nTarget probability (LLM): {LLM_target_prob:.3f}\n"
                             f"Target probability (oracle): {oracle_target_prob:.3f}\n"
                             f"KL: {KL_chatcompletion:.3f}\n\n\n")

            llm_distribution_data.append([i, KL_chatcompletion])
            llm_target_prob_by_round[i].append(LLM_target_prob)

    return llm_distribution_data, llm_target_prob_by_round


def run_oracle_driven(
        protein_names: list[str],
        attributes,
        experiments_by_name: dict,
        protein_vals: list[dict[str, str]],
        uniform_prior: dict[str, float],
        config: Config,
        rng: np.random.Generator,
):
    """Experiment selection maximizes EIG against the oracle's current belief p_oracle^k.
    The LLM is not queried at all here; only the oracle's target-probability trajectory
    p_oracle^k(theta*) is tracked, averaged per round across proteins."""
    oracle_target_prob_by_round = [[] for _ in range(config.num_experiments)]

    for protein in protein_vals:
        write_to_log(f"[Oracle-driven run] Starting on protein: {protein}\n\n\n\n")

        probabilities_oracle = dict(uniform_prior)
        history = {}

        for i in range(config.num_experiments):
            write_to_log(f"Round {i+1}\n\n")

            if config.experiment_selection == "eig":
                executed_experiment = select_experiment_eig(
                    protein_names, attributes, probabilities_oracle, history, experiments_by_name, config, rng
                )
            else:
                # random selection doesn't query the LLM, so this stays LLM-free too
                executed_experiment = select_experiment_random(
                    None, "", probabilities_oracle, history, experiments_by_name, config
                )
            experiment_outcome = executed_experiment.sample(protein)
            history = history | {executed_experiment.name: str(experiment_outcome)}

            probabilities_oracle = executed_experiment.posterior(experiment_outcome, protein_names, probabilities_oracle)
            oracle_target_prob = probabilities_oracle[protein["Protein"]]

            if config.do_showcase_world:
                write_to_log(
                    f"Executed experiment: {executed_experiment.name}, answer: {experiment_outcome}, "
                    f"oracle full distribution: {format_prob_distribution(probabilities_oracle)}\n\n"
                )
            else:
                write_to_log(
                    f"Executed experiment: {executed_experiment.name}, answer: {experiment_outcome}, "
                    f"oracle target prob: {oracle_target_prob:.3f}\n\n"
                )

            oracle_target_prob_by_round[i].append(oracle_target_prob)

    return oracle_target_prob_by_round


def format_prob_distribution(probabilities: dict[str, float]) -> str:
    return ", ".join(f"{name}: {prob:.3f}" for name, prob in probabilities.items())


def run(config: Config, model: Model) -> None:
    for i in range(config.num_worlds_per_run):
        write_to_log(f"Starting protein world - {i+1}-th run\n")
        protein_classification(model, config, i)
