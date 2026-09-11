import numpy as np

from helpers import convert_string_to_array, write_to_log
from models import Model
from twenty_questions.config import Config
from twenty_questions.prompts import answer_question_yesnocorrect_system_prompt, generate_original_animals_system_prompt


def _binary_entropy(p_yes: float, p_no: float) -> float:
    p_yes_clipped = max(p_yes, 1e-6)
    p_no_clipped = max(p_no, 1e-6)
    return - (p_yes_clipped * np.log(p_yes_clipped) + p_no_clipped * np.log(p_no_clipped))


# reverses a messages array so that the final question comes first
def reverse_history(history_questioner: list[dict[str,str]]) -> list[dict[str,str]]:
    blocks = [history_questioner[i:i+2] for i in range(0, len(history_questioner), 2)]
    return [x for b in blocks[::-1] for x in b]


def get_question_answered(question: str, goal_object: str, model: Model, answer_temperature: float) -> str:
    user_question = {"role": "user", "content": f"{question}"}
    messages = [answer_question_yesnocorrect_system_prompt(entity=goal_object), user_question]
    return model.chat_complete(messages=messages, temperature=answer_temperature, disable_thinking=True)[0]


def ask_question_and_log(question: str, goal_object: str, model: Model, config: Config) -> str:
    answer = get_question_answered(question, goal_object, model, config.answer_temperature)
    if answer.endswith("."):
        answer = answer[:-1]
    write_to_log(f"Best question: {question}, Answer: {answer}\n")
    return answer


# marks every round from round_index onwards as correctly guessed, in place
def mark_remaining_correct(correct_guess: list, round_index: int) -> None:
    correct_guess[round_index:] = [1] * (len(correct_guess) - round_index)


def generate_original_beliefs(model: Model, config: Config) -> list[str]:
    generation_temperature, max_num_samples, min_num_samples = config.generation_temperature_diverse, config.max_num_samples, config.min_num_samples
    user_question = {"role": "user", "content": f"Let\'s start the game of 20 questions. Generate a diverse "
                                                f"set of animals, at least {min_num_samples}."}
    messages = [generate_original_animals_system_prompt(max_num_samples), user_question]
    new_beliefs = model.chat_complete(messages=messages, temperature=generation_temperature, disable_thinking=True)[0]
    return convert_string_to_array(new_beliefs)
