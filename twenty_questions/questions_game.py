import numpy as np

from helpers import write_to_log
from models import Model
from twenty_questions.config import Config, RESULTS_DIR
from twenty_questions.classification import eval_bayes_categorical_changedfilter, \
    eval_bayes_categorical_oracleonly
from twenty_questions.game_utils import generate_original_beliefs, ask_question_and_log, mark_remaining_correct
from twenty_questions.generate_candidate_questions import generate_candidate_question_naive, select_best_question
from twenty_questions.sample_beliefs import sample_beliefs, sample_beliefs_naive
from twenty_questions.update_beliefs import update_beliefs_batched

NUM_ROUNDS = 20


def twenty_questions_animals_single_EIG(goal_animal: str, model: Model, config: Config) -> list[int]:
    history_questioner = []
    beliefs = generate_original_beliefs(model, config)
    write_to_log(f"Original beliefs: {beliefs}\n")
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n")
        # Generate candidate questions, select the question with best EIG
        best_question = select_best_question(beliefs, history_questioner, model, config)

        # Ask the best question, end game if correct animal was guessed
        answer = ask_question_and_log(best_question, goal_animal, model, config)
        if answer == "Correct!":
            mark_remaining_correct(correct_guess, i)
            return correct_guess

        # update the current beliefs to incorporate new questions
        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]
        beliefs = update_beliefs_batched(history_questioner, beliefs, model, config)
        write_to_log(f"Current beliefs: {beliefs}\n")

        # greedy decoding of current most likely belief
        guess = sample_beliefs(beliefs, history_questioner, model, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower():
          correct_guess[i] = 1
        write_to_log(f"Current best guess: {guess}\n")

    return correct_guess


def twenty_questions_animals_single_naive_eigreasoning(goal_animal: str, model: Model, config: Config) -> list[int]:
    history_questioner = []
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n")
        # prompt to ask a good question
        best_question = generate_candidate_question_naive(history_questioner, model, config.generation_temperature_simple)

        # Ask question, end game if correct animal was guessed
        answer = ask_question_and_log(best_question, goal_animal, model, config)
        if answer == "Correct!":
            mark_remaining_correct(correct_guess, i)
            return correct_guess

        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]

        # greedy decoding of current most likely belief
        guess = sample_beliefs_naive(history_questioner, model, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower():
            correct_guess[i] = 1
        write_to_log(f"Current best guess: {guess}\n")
    return correct_guess


extraction_methods = {
    "naive_eigreasoning": twenty_questions_animals_single_naive_eigreasoning,
    "EIG": twenty_questions_animals_single_EIG,
    "llm-driven": eval_bayes_categorical_changedfilter,
    "oracle-driven": eval_bayes_categorical_oracleonly,
}


def twenty_questions_animals(model: Model, target_animals: list[str], extraction_method_name: str, config: Config) -> list[float]:
    extraction_method = extraction_methods[extraction_method_name]
    accuracies = [0.0]*NUM_ROUNDS
    for goal_animal in target_animals:
        write_to_log(f"\n\nStarting on animal {goal_animal}\n")
        correct_guess = extraction_method(goal_animal, model, config)
        accuracies = [a + c for a, c in zip(accuracies, correct_guess)]
    return [a / len(target_animals) for a in accuracies]


def run(config: Config, model: Model) -> None:
    model_key = config.model
    for method_name in config.method_names:
        write_to_log(f"Starting with model {model_key}, method {method_name}\n\n")
        accuracy = twenty_questions_animals(model, config.animals, method_name, config)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(RESULTS_DIR / f"{model_key}_{config.seed}_{method_name}_accuracy.npy", np.array(accuracy))
