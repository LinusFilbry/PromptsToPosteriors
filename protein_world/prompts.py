from helpers import convert_to_prompt_message

sysprompt_content = ("You are a precise text generator. Follow formatting instructions exactly. Output only what is explicitly requested. "
                     "No commentary or extra text. No explanations. No step-by-step reasoning. Just the answer.")


def generate_protein_names(num_protein_classes: int) -> str:
    return f"""Create exactly {num_protein_classes} fictional protein names.
    Output format rules:
    • Output one name per line
    • Do not include any other text, labels, punctuation, headers, blank lines, or commentary.
    Create plausible fictional protein names (no real protein names)."""


def generate_latent_attribute_names(num_latent_attributes: int) -> str:
    return f"""Create exactly {num_latent_attributes} fictional continuous protein attribute names.
    Output format rules:
    • Output one name per line
    • Do not include any other text, labels, punctuation, headers, blank lines, or commentary.
    Create plausible fictional continuous protein attribute names (no real protein names)."""


def create_experiment_names(num_experiments: int) -> str:
    return f"""Create exactly {num_experiments} fictional experiment names to determine protein attributes.
    Output format rules:
    • Output one name per line
    • Do not include any other text, labels, punctuation, headers, blank lines, or commentary.
    Create plausible fictional experiment names (no real names)."""


def world_setup_messages(user_content: str) -> list[dict[str, str]]:
    return [convert_to_prompt_message("system", sysprompt_content),
            convert_to_prompt_message("user", user_content)]


# the system prompt describing the whole fictional world; shared by generated and showcase worlds
def build_world_string(protein_names: list[str], attributes, experiments) -> str:
    protein_string = f"These are all existing proteins: {protein_names}"

    attr_str = "\n\nEvery protein has the following latent attributes:\n\n"
    for attr in attributes:
        attr_str += f"{attr.description}\n\n"

    exp_str = "\n\nThe following experiments to reveal information about the attributes exist:\n\n"
    for exp in experiments:
        exp_str += f"{exp.description}\n\n"

    return f"You are an expert in the following fictional protein world:\n" + f"\n{protein_string} \n {attr_str} \n {exp_str} \n\n"


def protein_probability_messages(world_building_string: str, protein_names: list[str],
                                 history_questioner: dict[str, str],
                                 probs_LLM_initial: dict[str, float]) -> list[dict[str, str]]:
    sys_content = world_building_string + ("The user has found a protein and wants to determine which one it is. "
                                           "Using your knowledge, you will give a current best estimate for the classification.")
    sys_content += "Do not use code simulation techniques like MC sampling, instead reason about the actual distributions and probabilities of each protein to find an answer.\n"

    known_info = ""
    for exp, outcome in history_questioner.items():
        known_info += f"{exp}: {outcome}\n"
    initial_probabilities = ", ".join(f"{k}: {v:.3f}" for k, v in probs_LLM_initial.items()) + "\n"
    user_content = (
        f"You can assume that initially, this probability distribution over the protein classes held:\n"
        f"{initial_probabilities}\n"
        f"I have ran the following experiments with the described outcomes:\n"
        f"{known_info}\n")

    format = ""
    for k, prot_name in enumerate(protein_names):
        format += f"{prot_name}: <probability_{k+1}>, "
    format = format[:-2]
    addon_chatcompletion = (f"\nUsing all of your information: What probability does my protein have to be in each class? "
                            f"Reply in the format:\n"
                            f"{format}\n"
                            f"Make sure the result is a probability distribution: All probabilities have to be between 0 and 1, "
                            f"and they have to sum to 1.\n")

    final_addon = ("Do not include any reasoning, numbering or extra text in the final output. The answer must start "
                   "immediately and contain only the required format.")

    return [convert_to_prompt_message("system", sys_content)] + [convert_to_prompt_message("user", user_content + addon_chatcompletion + final_addon)]
