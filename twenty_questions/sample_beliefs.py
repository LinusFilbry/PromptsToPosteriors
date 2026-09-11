from models import Model
from twenty_questions.prompts import greedy_sample_animal_system_prompt, greedy_sample_animal_user_prompt, \
    greedy_sample_animal_system_prompt_naive, greedy_sample_animal_user_prompt_naive


def sample_beliefs(beliefs: list[str], history_questioner: list[dict[str, str]], model: Model,
                   generation_temperature: float) -> str:
    messages = ([greedy_sample_animal_system_prompt()] + history_questioner +
                [greedy_sample_animal_user_prompt(beliefs)])
    return model.chat_complete(messages=messages, temperature=generation_temperature)[0]


def sample_beliefs_naive(history_questioner: list[dict[str, str]], model: Model, generation_temperature: float) -> str:
    messages = ([greedy_sample_animal_system_prompt_naive()] + history_questioner +
                [greedy_sample_animal_user_prompt_naive()])
    return model.chat_complete(messages=messages, temperature=generation_temperature)[0]