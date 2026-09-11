import math
import re

from helpers import convert_to_prompt_message, write_to_log
from dutch_books.config import Config
from dutch_books.prompts import DUTCH_BOOK_SYSTEM_PROMPT, SCENARIO_1_PAIRS, SCENARIO_2_SERIES, \
    SCENARIO_3_PAIRS, build_user_prompt
from models import Model


# --- Parsing -------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+\.\d+|\d+")


def parse_probability(response: str) -> float | None:
    if not response:
        return None
    for match in _NUMBER_RE.finditer(response):
        try:
            value = float(match.group())
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None


# --- Elicitation -----------------------------------------------------------

def _elicit_single(
    model: Model, question: str, temperature: float, config: Config, messages: list[dict[str, str]]
) -> tuple[float | None, list[dict[str, str]]]:
    messages = messages + [convert_to_prompt_message("user", build_user_prompt(question))]
    response = model.chat_complete(messages, temperature, 1)[0]
    messages = messages + [convert_to_prompt_message("assistant", response)]
    probability = parse_probability(response)

    write_to_log(f'Question: "{question}"\n')
    write_to_log(f"Raw response: {response!r}\n")
    if probability is None:
        write_to_log(
            "PARSE FAILURE: could not extract a probability in [0, 1] from the response above; "
            "skipping this question in downstream comparisons.\n\n"
        )
    else:
        write_to_log(f"Parsed probability: {probability:.3f}\n\n")

    return probability, messages


def elicit_sequence(
    model: Model, questions: list[str], temperature: float, config: Config, same_chat: bool
) -> list[float | None]:
    probabilities = []
    if same_chat:
        messages = [convert_to_prompt_message("system", DUTCH_BOOK_SYSTEM_PROMPT)]
        for question in questions:
            probability, messages = _elicit_single(model, question, temperature, config, messages)
            probabilities.append(probability)
    else:
        for question in questions:
            messages = [convert_to_prompt_message("system", DUTCH_BOOK_SYSTEM_PROMPT)]
            probability, _ = _elicit_single(model, question, temperature, config, messages)
            probabilities.append(probability)
    return probabilities


# --- Scenario 1: normalization -----------------------------------------------

def run_scenario_1(model: Model, config: Config, same_chat: bool) -> tuple[int, int]:
    label = "same-chat" if same_chat else "separate-chat"
    write_to_log(f"\n=== Scenario 1 ({label}): Normalization coherence (p_yes + p_no should = 1) ===\n\n")
    violations = 0
    for pair in SCENARIO_1_PAIRS:
        write_to_log(f"-- Pair: {pair['name']} --\n")
        p_yes, p_no = elicit_sequence(
            model, [pair["positive"], pair["negation"]], config.answer_temperature, config, same_chat
        )

        if p_yes is None or p_no is None:
            write_to_log(f"Skipping incoherence check for '{pair['name']}' (parse failure).\n\n")
            continue

        total = p_yes + p_no
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            violations += 1
            write_to_log(
                f"DUTCH BOOK FOUND [scenario 1 - normalization, {pair['name']}, {label}]: "
                f"p(yes)={p_yes:.3f}, p(no)={p_no:.3f}, sum={total:.3f} != 1.0\n\n"
            )
        else:
            write_to_log(f"Coherent: p(yes)={p_yes:.3f}, p(no)={p_no:.3f}\n\n")

    total_pairs = len(SCENARIO_1_PAIRS)
    write_to_log(f"Scenario 1 ({label}) violations: {violations}/{total_pairs}\n\n")
    return violations, total_pairs


# --- Scenario 2: monotonicity -----------------------------------------------

def run_scenario_2(model: Model, config: Config, same_chat: bool) -> tuple[int, int]:
    label = "same-chat" if same_chat else "separate-chat"
    write_to_log(
        f"\n=== Scenario 2 ({label}): Monotonicity (subset prob should not exceed superset prob) ===\n\n"
    )
    violations = 0
    for series_name, series in SCENARIO_2_SERIES.items():
        write_to_log(f"-- Series: {series_name} --\n")
        questions = [series["template"].format(x=x) for x in series["values"]]
        probabilities = elicit_sequence(model, questions, config.answer_temperature, config, same_chat)
        results = list(zip(series["values"], probabilities))

        found = False
        for i in range(len(results)):
            x_loose, p_loose = results[i]
            if p_loose is None:
                continue
            for j in range(i + 1, len(results)):
                x_strict, p_strict = results[j]
                if p_strict is None:
                    continue
                if p_strict > p_loose:
                    write_to_log(
                        f"DUTCH BOOK FOUND [scenario 2 - monotonicity, {series_name}, {label}]: "
                        f"stricter threshold {x_strict} has p={p_strict:.3f} > "
                        f"looser threshold {x_loose} has p={p_loose:.3f}\n\n"
                    )
                    found = True
                    break
            if found:
                break
        if found:
            violations += 1

    total_series = len(SCENARIO_2_SERIES)
    write_to_log(f"Scenario 2 ({label}) violations: {violations}/{total_series}\n\n")
    return violations, total_series


# --- Scenario 3: extensionality -----------------------------------------------

def run_scenario_3(model: Model, config: Config, same_chat: bool) -> tuple[int, int]:
    label = "same-chat" if same_chat else "separate-chat"
    write_to_log(f"\n=== Scenario 3 ({label}): Extensionality coherence (p1 should = p2) ===\n\n")
    violations = 0
    for pair in SCENARIO_3_PAIRS:
        write_to_log(f"-- Pair: {pair['name']} --\n")
        p1, p2 = elicit_sequence(model, [pair["q1"], pair["q2"]], config.answer_temperature, config, same_chat)

        if p1 is None or p2 is None:
            write_to_log(f"Skipping incoherence check for '{pair['name']}' (parse failure).\n\n")
            continue

        if not math.isclose(p1, p2, abs_tol=1e-9):
            violations += 1
            write_to_log(
                f"DUTCH BOOK FOUND [scenario 3 - extensionality, {pair['name']}, {label}]: "
                f"p1={p1:.3f}, p2={p2:.3f}, expected p1 = p2 (same event, different positive formulation)\n\n"
            )
        else:
            write_to_log(f"Coherent: p1={p1:.3f}, p2={p2:.3f}\n\n")

    total_pairs = len(SCENARIO_3_PAIRS)
    write_to_log(f"Scenario 3 ({label}) violations: {violations}/{total_pairs}\n\n")
    return violations, total_pairs


# --- Orchestrator ------------------------------------------------------------

ELICITATION_MODES = {
    "same_chat": (True, "Same-chat"),
    "separate_chat": (False, "Separate-chat"),
}


def _run_scenario_with_comparison(scenario_fn, title: str, model: Model, config: Config) -> None:
    write_to_log(f"\n\n##### {title} #####\n\n")
    summary_lines = []
    for mode in config.elicitation_modes:
        same_chat, label = ELICITATION_MODES[mode]
        violations, total = scenario_fn(model, config, same_chat)
        summary_lines.append(f"{label} violations: {violations}/{total}")
    write_to_log(f"--- Summary for {title} ---\n" + "\n".join(summary_lines) + "\n\n")


SCENARIOS = {
    "normalization": (run_scenario_1, "Scenario 1: Normalization coherence"),
    "monotonicity": (run_scenario_2, "Scenario 2: Monotonicity"),
    "extensionality": (run_scenario_3, "Scenario 3: Extensionality coherence"),
}


def run(config: Config, model: Model) -> None:
    write_to_log("\n\n##### Starting Dutch Book experiment #####\n\n")
    selected = set(config.scenarios)
    for name, (scenario_fn, title) in SCENARIOS.items():
        if name in selected:
            _run_scenario_with_comparison(scenario_fn, title, model, config)
    write_to_log("##### Finished Dutch Book experiment #####\n\n")
