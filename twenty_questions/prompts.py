from helpers import convert_to_prompt_message


def generate_original_animals_system_prompt(num_samples: int) -> dict[str, str]:
    content = (
        f"You are playing a game of 20 Questions trying to guess an animal. \n\n"
        f"To start the game, generate up to {num_samples} candidate animals which "
        f"might be the answer. Each candidate must be a single, existing animal (e.g., \"Scarlet macaw\", \"White Shark\", "
        f"\"Rhinoceros beetle\"). List each animal on its own line - no numbering, punctuation, or extra text. Generate "
        f"answers in English only. Produce a varied set by diversifying along different kinds of features. Don't count "
        f"out more obscure animals. Do not repeat any animal. Return only the list of animals."
    )
    return convert_to_prompt_message(role="system", content=content)


def candidate_generation_system_message() -> dict[str, str]:
    content = (
            "You are playing a game of 20 Questions to guess an animal. When asked, you must produce Yes/No questions. "
            "You must not add any explanation or extra text: only return the strict format requested by the user."
        )
    return convert_to_prompt_message(role="system", content=content)


def candidate_generation_system_message_naive() -> dict[str, str]:
    content = (
        "You are playing a game of 20 Questions to guess an animal. When asked, you must produce a Yes/No question. "
        "You must not add any explanation or extra text: only return the strict format requested by the user."
    )
    return convert_to_prompt_message(role="system", content=content)


def conditional_question_generation_prompt(beliefs: list[str], num_questions: int) -> dict[str, str]:
    content = (
            f"Using the beliefs list: {beliefs} and all previous questions and answers:\n\n"
            f"Generate up to {num_questions} candidate Yes/No questions that split the list into two roughly equal parts. "
            "Each question should be phrased so the answer is Yes or No. Do not repeat questions. List each question "
            "on its own line - no numbering, punctuation, or extra text."
        )
    return convert_to_prompt_message(role="user", content=content)


def question_generation_prompt_naive() -> dict[str, str]:
    content = (
        f"Generate a Yes/No question in the following way:\n\n"
        f"- In your internal thought process, make a list of up to 10 animals which are still possible based on all the previous "
        f"questions and answers. Do this by varying along features not already fixed from previous questions. \n"
        f"- If you can only come up with 3 or fewer animals fitting all previous questions and answers, ask \"Is it <animal>?\" "
        f"for the animal you think is most likely. \n"
        f"- Otherwise, propose up to 10 questions which you think divide the set of animals reasonably well. \n"
        f"- For each question, calculate the expected information gain across the list of animals. \n"
        f"- Immediately stop your thinking and ask the question with highest EIG. Don't consider new "
        f"questions or other animals than those originally generated.\n\n"
        f"Return only the question, and nothing else."
    )
    return convert_to_prompt_message(role="user", content=content)


def unconditional_question_generation_prompt(candidate_questions: list[str], num_questions_left: int) -> dict[str, str]:
    content = (
            f"Using the the current candidate questions: {candidate_questions} and all previous questions and answers:\n\n"
            f"Generate exactly {num_questions_left} candidate Yes/No questions to identify the target animal. "
            "Each question should be phrased so the answer is Yes or No. Do not repeat questions. List each question on "
            "its own line - no numbering, punctuation, or extra text."
        )
    return convert_to_prompt_message(role="user", content=content)


def answer_question_yesnocorrect_system_prompt(entity: str) -> dict[str, str]:
    content = (
            f"You are playing the answerer in a game of 20 Questions. "
            f"Your chosen entity is: \n\n {entity} \n\n"
            f"When asked a question, you must reply exactly \"Yes\" or \"No\", depending on if your chosen entity "
            f"fulfills the question. If the question correctly guesses your chosen entity, reply exactly \"Correct!\"."
            f"You must not add any explanation, extra text or anything else."
        )
    return convert_to_prompt_message(role="system", content=content)


def answer_question_yesno_system_prompt(entity: str, use_logits: bool) -> dict[str, str]:
    if use_logits:
        content = (
            f"You are playing the answerer in a game of 20 Questions. "
            f"Your chosen entity is: \n\n {entity} \n\n"
            f"When asked a question, you must reply exactly \"Yes\" or \"No\", depending on if your chosen entity "
            f"fulfills the question. You must not add any explanation, extra text or anything else."
        )
    else:
        content = (
            f"You are playing the answerer in a game of 20 Questions. "
            f"Your chosen entity is: \n\n {entity} \n\n"
            f"When asked a Yes/No question, you must give probabilities to the answers \"Yes\" or \"No\", depending on "
            f"how your animal fits to the answer. Your answer must be in the format: \n"
            f"Yes: <prob1>, No: <prob2>\n"
            f"where the probabilities are decimal numbers. You must not add any explanation, extra text or anything else."
        )
    return convert_to_prompt_message(role="system", content=content)



def generate_animals_system_prompt(max_num_samples: int, min_num_samples: int) -> dict[str, str]:
    content = (
            f"You are playing a game of 20 Questions trying to guess an animal. When asked to, using all of the questions "
            f"and answers: Generate up to {max_num_samples} candidate animals that satisfy every clue, aiming for "
            f"at least {min_num_samples}. Each candidate must be an existing animal (e.g., \"Scarlet macaw\", \"White Shark\", \"Rhinoceros beetle\")."
            f"List each animal on its own line - no numbering, punctuation, or extra text. Generate answers in English only. "
            f"Produce a varied set by identifying features not implied by the clues and diversifying "
            f"along them. Don't count out more obscure animals. Do not repeat any animal. Return only the list of animals."
        )
    return convert_to_prompt_message(role="system", content=content)


def generate_animals_user_prompt() -> dict[str, str]:
    content = (
        f"Now generate candidate animals fitting to the questions and answers. Only generate animals that really fit to "
        f"ALL questions and answers. In your final response, don't include any commentary like \'this might be valid\', "
        f"really return only the animal names."
    )
    return convert_to_prompt_message(role="user", content=content)


def generate_more_animals_system_prompt(curr_samples: list[str], target_num_samples: int) -> dict[str, str]:
    content = (
            f"You are playing a game of 20 Questions trying to guess an animal. Using all of the questions "
            f"and answers so far, and a current list of candidates {curr_samples}: If possible under the restrictions, "
            f"generate up to {target_num_samples} additional animals that satisfy every clue. Each candidate must be an "
            f"existing animal (e.g., \"Scarlet macaw\", \"White Shark\", \"Rhinoceros beetle\") not belonging to the "
            f"given list. As output, list each new animal on its own line - no numbering, punctuation, or extra text. "
            f"Generate answers in English only. Produce a varied set by identifying features not implied by the clues "
            f"and diversifying along them. Don't count out more obscure animals. Do not repeat any animal. "
            f"Return only the list of new animals."
        )
    return convert_to_prompt_message(role="system", content=content)


def greedy_sample_animal_system_prompt() -> dict[str, str]:
    content = (
            f"You are playing a game of 20 Questions trying to guess an animal. After asking Yes/No questions and "
            f"receiving answers, and being provided a list of possible beliefs, you will be prompted to choose the "
            f"animal of the possible beliefs you think is the most likely candidate."
        )
    return convert_to_prompt_message(role="system", content=content)


def greedy_sample_animal_user_prompt(beliefs: list[str]) -> dict[str, str]:
    content = (
            f"Using all of the questions and answers so far: Which of the candidates {beliefs} do you think is the "
            f"correct one? List exactly one of the animals from the list. Write nothing else: no numbering, "
            f"punctuation, or extra text."
        )
    return convert_to_prompt_message(role="user", content=content)


def greedy_sample_animal_system_prompt_naive() -> dict[str, str]:
    content = (
        f"You are playing a game of 20 Questions trying to guess an animal. After asking Yes/No questions and "
        f"receiving answers, you will be prompted to choose the animal you think is the most likely candidate."
    )
    return convert_to_prompt_message(role="system", content=content)


def greedy_sample_animal_user_prompt_naive() -> dict[str, str]:
    content = (
        f"Using all of the questions and answers so far: What do you think is the correct animal? "
        f"Write exactly one animal. Write nothing else: no numbering, punctuation, or extra text."
    )
    return convert_to_prompt_message(role="user", content=content)


def sample_animals_independent_system_prompt() -> dict[str, str]:
    content = (
        f"You are playing a game of 20 Questions trying to guess an animal. After asking Yes/No questions and "
        f"receiving answers, you will be prompted to judge whether a given animal fits to the questions and answers."
    )
    return convert_to_prompt_message(role="system", content=content)


def sample_animals_independent_user_prompt(belief: str, use_logits: bool) -> dict[str, str]:
    if use_logits:
        content = (
            f"Does this animal\n"
            f"{belief}\n"
            f"fit all the previous questions and answers? Reply exactly \"Yes\" or \"No\", depending on which best "
            f"describes if the animal fits. Consider every single question-answer-pair and answer \"Yes\" if the animals "
            f"fits all of them."
        )
    else:
        content = (
            f"Does this animal: '{belief}' \n"
            f"fit all the previous questions and answers? Reply in the format:\n"
            f"Yes: <prob1>, No: <prob2>\n"
            f"where the probabilities are decimal numbers. Consider every single question-answer-pair and give positive "
            f"probability to \"Yes\" only if the animals fits all answers. Reply exactly what is listed in the format, and "
            f"nothing else."
        )
    return convert_to_prompt_message(role="user", content=content)
