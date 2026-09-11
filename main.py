import argparse
import importlib
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent

DEFAULT_MODEL = "gemma-4-31b"
DEFAULT_SEED = 42


def read_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class ListAction(argparse.Action):
    def __init__(self, option_strings, dest, item_type=str, item_choices=None, **kwargs):
        kwargs.setdefault("nargs", "+")
        kwargs.setdefault("metavar", "[A, B]")
        self.item_type = item_type
        self.item_choices = item_choices
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        text = " ".join(values).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                items = yaml.safe_load(text)
            except yaml.YAMLError:
                items = text[1:-1].split(",")
            if not isinstance(items, list):
                raise argparse.ArgumentError(self, f"expected a list, got {text!r}")
        else:
            items = [chunk for chunk in text.replace(",", " ").split() if chunk]

        parsed = []
        for item in items:
            item = self.item_type(item) if isinstance(item, str) else item
            if self.item_choices is not None and item not in self.item_choices:
                raise argparse.ArgumentError(
                    self,
                    f"invalid choice: {item!r} (choose from {', '.join(map(repr, self.item_choices))})",
                )
            parsed.append(item)
        setattr(namespace, self.dest, parsed)


class ListFormatter(argparse.HelpFormatter):
    def _format_args(self, action, default_metavar):
        if isinstance(action, ListAction):
            return self._metavar_formatter(action, default_metavar)(1)[0]
        return super()._format_args(action, default_metavar)



TASK_OVERRIDABLE_FIELDS = {
    "twenty_questions": ["method_names", "animals"],
    "protein_world": [
        "num_worlds_per_run", "proteins_per_world", "num_protein_classes", "num_attr",
        "pre_req_list", "num_experiments", "do_showcase_world", "exp_type",
    ],
    "dutch_books": ["scenarios", "elicitation_modes"],
}


def _common_parent() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, formatter_class=ListFormatter)
    common.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, metavar="KEY",
        help="Which model to run, by its key in the model registry, e.g. --model=gemma-4-31b-reasoning. "
             f"Defaults to '{DEFAULT_MODEL}'.",
    )
    common.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed. Defaults to {DEFAULT_SEED}.",
    )
    common.add_argument(
        "--logfile", default="logfile", metavar="NAME",
        help="Name for the log file (without extension), written to logs/NAME.txt. "
             "Defaults to 'logfile' (logs/logfile.txt).",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=ListFormatter)
    subparsers = parser.add_subparsers(dest="task", required=True, help="Which task to run.")

    twenty_q = subparsers.add_parser(
        "twenty_questions", parents=[_common_parent()], formatter_class=ListFormatter, help="Run the 20 Questions task."
    )
    twenty_q.add_argument(
        "--method-names", action=ListAction, default=None, metavar="[NAME, ...]",
        help="Which methods to run, as an array literal, e.g. --method-names '[llm-driven, oracle-driven]'.",
    )
    twenty_q.add_argument(
        "--animals", action=ListAction, default=None, metavar="[ANIMAL, ...]",
        help="Which animals to run, as an array literal, e.g. --animals '[cat, dog]'.",
    )

    protein = subparsers.add_parser(
        "protein_world", parents=[_common_parent()], formatter_class=ListFormatter, help="Run the ProteinWorld task."
    )
    protein.add_argument("--num-worlds-per-run", type=int, default=None)
    protein.add_argument("--proteins-per-world", type=int, default=None)
    protein.add_argument("--num-protein-classes", type=int, default=None)
    protein.add_argument("--num-attr", type=int, default=None)
    protein.add_argument(
        "--pre-req-list", action=ListAction, default=None, metavar="[[..], ...]",
        help="One dependency list per attribute, as an array literal, "
             "e.g. --pre-req-list '[[], [], [0], [1, 2]]'",
    )
    protein.add_argument("--num-experiments", type=int, default=None)
    protein.add_argument(
        "--do-showcase-world", action=argparse.BooleanOptionalAction, default=None,
        help="Load the fixed showcase world instead of generating a random one "
             "(only supported for exp_type simple or complex).",
    )
    protein.add_argument(
        "--exp-type", default=None,
        choices=["simple", "simple_noisy", "slightly_complex", "complex"],
    )

    dutch = subparsers.add_parser(
        "dutch_books", parents=[_common_parent()], formatter_class=ListFormatter, help="Run the Dutch Books task."
    )
    dutch.add_argument(
        "--scenarios", action=ListAction,
        item_choices=["normalization", "monotonicity", "extensionality"],
        default=None, metavar="[SCENARIO, ...]",
        help="Which of the three scenarios to run, as an array literal, "
             "e.g. --scenarios '[normalization, extensionality]'.",
    )
    dutch.add_argument(
        "--elicitation-modes", action=ListAction,
        item_choices=["same_chat", "separate_chat"],
        default=None, metavar="[MODE, ...]",
        help="Whether to elicit in the same (same_chat) or across (separate_chat) chats, "
             "as an array literal, e.g. --elicitation-modes '[same_chat, separate_chat]'.",
    )

    return parser


def load_task_config(task_name: str, overrides: dict):
    task_config_path = ROOT / task_name / "config.yaml"
    task_raw = read_yaml(task_config_path) if task_config_path.exists() else {}

    raw = merge_dicts(task_raw, overrides)
    raw["task"] = task_name

    package = importlib.import_module(task_name)
    return package, package.load_config(raw)


def main():
    from helpers import set_log_file
    from models import MODEL_REGISTRY_PATH, build_model, load_model_registry

    parser = build_parser()
    args = parser.parse_args()

    set_log_file(args.logfile)

    overrides: dict = {"model": args.model, "seed": args.seed}
    for field_name in TASK_OVERRIDABLE_FIELDS[args.task]:
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value

    package, config = load_task_config(args.task, overrides)

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    if config.model not in registry:
        raise KeyError(f"Unknown model '{config.model}'. Choose one of: {', '.join(registry)}.")

    np.random.seed(config.seed)
    random.seed(config.seed)

    model = build_model(config, registry[config.model])
    package.run(config, model)


if __name__ == "__main__":
    main()
