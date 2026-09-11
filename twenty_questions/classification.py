import numpy as np

from helpers import calculate_kl_divergence, normalize_prob_dict, write_to_log
from models import Model
from twenty_questions.config import Config, RESULTS_DIR
from twenty_questions.game_utils import generate_original_beliefs, reverse_history, \
    ask_question_and_log, mark_remaining_correct
from twenty_questions.generate_candidate_questions import select_best_question
from twenty_questions.prompts import generate_animals_system_prompt, answer_question_yesno_system_prompt, \
    sample_animals_independent_system_prompt, sample_animals_independent_user_prompt
from twenty_questions.update_beliefs import generate_new_beliefs

NUM_ROUNDS = 20


def eval_bayes_categorical_changedfilter(goal_animal: str, model: Model, config: Config) -> list[float]:
    history_questioner = []
    bayes_analysis = []
    # belief -> (product probability so far, whether it's still consistent with every answer so far)
    bayes_cache: dict[str, tuple[float, bool]] = {}
    beliefs = generate_original_beliefs(model, config)
    write_to_log(f"Original beliefs: {beliefs}\n")
    # correct_guess[i] = 1 <--> model had it right after i-th question
    correct_guess = [0.0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n")
        # Generate candidate questions, select the question with best EIG
        best_question = select_best_question(beliefs, history_questioner, model, config)

        # Ask the best question, end game if correct animal was guessed
        answer = ask_question_and_log(best_question, goal_animal, model, config)
        if answer == "Correct!":
            mark_remaining_correct(correct_guess, i)
            save_bayes_analysis(config, goal_animal, bayes_analysis)
            return correct_guess

        # update the current beliefs to incorporate new questions + calculate new distribution, Bayesian oracle, BED_LLM filter
        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]
        beliefs_updated, probabilities_new, added_beliefs = update_beliefs_categorical_changedfilter(history_questioner, beliefs, model, config)
        probabilities, filtered_list = compute_Bayesian_posterior(beliefs_updated, history_questioner, model, config, bayes_cache)
        KL_divergence = calculate_kl_divergence(probabilities, probabilities_new)
        num_old = len(beliefs)
        beliefs = [k for k, v in probabilities_new.items() if v > config.threshold_rejection_probability]
        # drop discarded beliefs from the cache so they get fully recomputed if they ever reappear
        beliefs_set = set(beliefs)
        bayes_cache = {k: v for k, v in bayes_cache.items() if k in beliefs_set}

        # log and save relevant data
        num_new, num_valid_changedfilter, num_valid_BEDLLM = len(beliefs), len({k:v for k,v in probabilities_new.items() if v > config.threshold_rejection_probability}.keys()), len(filtered_list)
        write_to_log(f"KL divergence: {KL_divergence:.3f} (old size: {num_old}, new size: {num_new}, valid size: CF {num_valid_changedfilter}/BED-LLM {num_valid_BEDLLM})\n")
        write_to_log(f"Current beliefs: {beliefs}\n")
        bayes_analysis.append((num_old, num_new, num_valid_changedfilter, num_valid_BEDLLM, KL_divergence, i))

        # greedy decoding of current most likely belief
        normalized_probabilities_new = normalize_prob_dict(probabilities_new)
        matching_belief = next((b for b in normalized_probabilities_new if b.lower() == goal_animal.lower()), None)
        if matching_belief is not None:
            correct_guess[i] = normalized_probabilities_new[matching_belief]
        write_to_log(f"Current target probability: {correct_guess[i]}\n")

    save_bayes_analysis(config, goal_animal, bayes_analysis)
    return correct_guess

# same game loop as twenty_questions_animals_single_categorical_changedfilter_bayesiness, but never elicits the
# LLM's own categorical distribution: beliefs are scored and filtered using only the Bayesian oracle posterior
def eval_bayes_categorical_oracleonly(goal_animal: str, model: Model, config: Config) -> list[float]:
    history_questioner = []
    # belief -> (product probability so far, whether it's still consistent with every answer so far)
    bayes_cache: dict[str, tuple[float, bool]] = {}
    beliefs = generate_original_beliefs(model, config)
    write_to_log(f"Original beliefs: {beliefs}\n")
    # correct_guess[i] = 1 <--> model had it right after i-th question
    correct_guess = [0.0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n")
        # Generate candidate questions, select the question with best EIG
        best_question = select_best_question(beliefs, history_questioner, model, config)

        # Ask the best question, end game if correct animal was guessed
        answer = ask_question_and_log(best_question, goal_animal, model, config)
        if answer == "Correct!":
            mark_remaining_correct(correct_guess, i)
            return correct_guess

        # update the current beliefs to incorporate new questions, then score + filter purely via the Bayesian oracle
        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]
        beliefs_updated = generate_updated_beliefs_oracleonly(history_questioner, beliefs, model, config)
        probabilities, filtered_list = compute_Bayesian_posterior(beliefs_updated, history_questioner, model, config, bayes_cache)
        beliefs = filtered_list
        # drop discarded beliefs from the cache so they get fully recomputed if they ever reappear
        beliefs_set = set(beliefs)
        bayes_cache = {k: v for k, v in bayes_cache.items() if k in beliefs_set}

        # log current beliefs
        write_to_log(f"Current beliefs: {beliefs}\n")

        # greedy decoding of current most likely belief, straight from the oracle posterior
        normalized_probabilities = normalize_prob_dict(probabilities)
        matching_belief = next((b for b in normalized_probabilities if b.lower() == goal_animal.lower()), None)
        if matching_belief is not None:
            correct_guess[i] = normalized_probabilities[matching_belief]
        write_to_log(f"Current target probability: {correct_guess[i]}\n")

    return correct_guess


def save_bayes_analysis(config: Config, goal_animal: str, bayes_analysis: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / f"{config.model}_{config.seed}_{goal_animal}_kl-trace.npy", np.array(bayes_analysis))


def generate_updated_beliefs_oracleonly(history: list[dict[str, str]], beliefs: list[str], model: Model, config: Config) -> list[str]:
    system_prompt = generate_animals_system_prompt(config.max_num_samples, config.min_num_samples)
    beliefs_new = generate_new_beliefs(system_prompt, history, model, config.generation_temperature_diverse)
    return list({s.lower(): s for s in beliefs_new + beliefs}.values())


def update_beliefs_categorical_changedfilter(history: list[(str, str)], beliefs: list[str], model: Model, config: Config) -> (list[str], dict[str, float], list[str]):
    # generate new beliefs
    system_prompt = generate_animals_system_prompt(config.max_num_samples, config.min_num_samples)
    beliefs_new = generate_new_beliefs(system_prompt, history, model, config.generation_temperature_diverse)
    beliefs_updated = list({s.lower(): s for s in beliefs_new + beliefs}.values())

    # check which once were truly newly created
    new_unique_beliefs = [b for b in beliefs_new if b.lower() not in {old.lower() for old in beliefs}]

    # build new categorical distribution
    probabilities = sample_belief_probabilities(model, history, config.answer_temperature,
                                                beliefs_updated, config.batched_block_size, normalized=False)

    # keep only the 100 highest-scoring beliefs, drop the rest
    beliefs_updated = sorted(beliefs_updated, key=lambda b: probabilities[b], reverse=True)[:100]
    beliefs_updated_set = set(beliefs_updated)
    probabilities = {b: p for b, p in probabilities.items() if b in beliefs_updated_set}
    new_unique_beliefs = [b for b in new_unique_beliefs if b in beliefs_updated_set]

    return beliefs_updated, probabilities, new_unique_beliefs


# also compute beliefs filtered by BED_LLM to save computation
# `bayes_cache` maps belief -> (product probability, still-consistent flag) accumulated over history so far.
# A belief already in the cache only needs its newest question-answer pair scored, since earlier pairs were
# already folded into the cached product; a belief seen for the first time is scored against the full history.
def compute_Bayesian_posterior(beliefs: list[str], history_questioner: list[dict[str, str]], model: Model,
                               config: Config, bayes_cache: dict[str, tuple[float, bool]]) -> (dict[str, float], list[str]):
    answer_temperature, block_size, BED_LLM_rejection_threshold = config.answer_temperature, config.batched_block_size, config.threshold_rejection_probability

    bayes_posterior = {}
    filtered_beliefs = []

    conversations = []
    answers = []
    belief_qa_counts = []
    # generate one conversation per belief for each question-answer pair not yet folded into the cache
    for new_belief in beliefs:
        if new_belief in bayes_cache:
            question_indices = [len(history_questioner) - 2]
        else:
            question_indices = [i for i, m in enumerate(history_questioner) if m["role"] == "assistant"]
        belief_qa_counts.append(len(question_indices))
        for i in question_indices:
            question = history_questioner[i]["content"]
            answer = history_questioner[i+1]["content"]
            user_question = {"role": "user", "content": question}
            conversations.append([answer_question_yesno_system_prompt(entity=new_belief, use_logits=not model.is_thinking), user_question])
            answers.append(answer)

    # get answers for all conversations in parallel
    probabilities = model.chat_probabilities_messages_batched(conversations, ["Yes", "No"],
                                                                   temperature=answer_temperature, block_size=block_size, disable_thinking=not model.is_thinking)

    # for each belief, fold the new question-answer-pair(s) into its cached product and filter out inconsistent beliefs
    offset = 0
    for i, new_belief in enumerate(beliefs):
        count = belief_qa_counts[i]
        probs_belief = probabilities[offset:offset+count]
        answers_belief = answers[offset:offset+count]
        offset += count

        prob, valid_belief = bayes_cache.get(new_belief, (1.0, True))
        for j, answer in enumerate(probs_belief):
            prob *= answer[answers_belief[j]]
            if ((answer["Yes"] > (1 - BED_LLM_rejection_threshold) and answers_belief[j] == "No")
                    or (answer["Yes"] < BED_LLM_rejection_threshold and answers_belief[j] == "Yes")):
                valid_belief = False

        bayes_cache[new_belief] = (prob, valid_belief)
        if valid_belief:
            filtered_beliefs.append(new_belief)
        bayes_posterior[new_belief] = prob

    return bayes_posterior, filtered_beliefs


def sample_belief_probabilities(model: Model, history: list[dict[str, str]], generation_temperature: float, beliefs: list[str], block_size: int, normalized: bool = True) -> dict[str, float]:
    messages = []
    for belief in beliefs:
        messages.append(([sample_animals_independent_system_prompt()] + reverse_history(history) + [
            sample_animals_independent_user_prompt(belief, use_logits=not model.is_thinking)]))

    # get probability of each belief being a good answer
    probs = model.chat_probabilities_messages_batched(messages, ["Yes", "No"], generation_temperature, block_size, disable_thinking=not model.is_thinking)

    # collect and normalize probabilities
    answer = {}
    for i, belief in enumerate(beliefs):
        answer[belief] = probs[i]["Yes"]
    if normalized:
        answer = normalize_prob_dict(answer)

    return answer


