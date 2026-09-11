from helpers import write_to_log
from models import Model
from protein_world.config import Config
from protein_world.prompts import protein_probability_messages

MAX_FORMAT_RETRIES = 3

def sample_protein_probabilities_chatcompletion(world_building_string: str, protein_names: list[str], history_questioner: dict[str,str],
                                 probs_LLM_initial: dict[str, float], model: Model, config: Config) -> dict[str, float]:
    messages_chatcompletion = protein_probability_messages(
        world_building_string, protein_names, history_questioner, probs_LLM_initial
    )

    # phi-4 tends to drift from the required format unless its answer is forced to already
    # start with the first protein name, so pin the completion's start to match the format
    addon_assistant = f"{protein_names[0]}: " if "phi" in model.model_name.lower() else ""

    last_error = None

    for attempt in range(MAX_FORMAT_RETRIES):

        probabilities_chatcompletion_string = model.chat_complete(
            messages_chatcompletion, temperature=config.generation_temperature_simple, seed=config.seed+attempt,
            addon_assistant=addon_assistant
        )[0]

        try:
            probabilities_chatcompletion = {
                key.strip(): float(value.strip())
                for key, value in (item.split(":") for item in probabilities_chatcompletion_string.split(","))
            }
            missing = [name for name in protein_names if name not in probabilities_chatcompletion]
            if missing:
                raise ValueError(f"Missing protein names in response: {missing}")
            return probabilities_chatcompletion
        except Exception as e:
            last_error = e
            write_to_log(
                f"\n\nFailed to parse probability response (attempt {attempt + 1}/{MAX_FORMAT_RETRIES}): {e}\n"
                f"Raw LLM response: {probabilities_chatcompletion_string}\n"
            )

    raise last_error
