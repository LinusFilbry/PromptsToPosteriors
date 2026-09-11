from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, TypeAlias, cast

import numpy as np

TransformName: TypeAlias = Literal["square", "tanh"]
CombineOp: TypeAlias = Literal["add", "multiply"]
SampleInput: TypeAlias = Dict[str, str]
SampleFn: TypeAlias = Callable[[SampleInput], str]
ProteinParams: TypeAlias = dict[str, float]

@dataclass
class SimpleProteinAttribute:
    name: str
    pre_reqs: list[str]

@dataclass(frozen=True)
class ProteinAttribute:
    name: str
    pre_req: list[str]
    sample_fn: SampleFn
    description: str
    protein_params: ProteinParams = field(default_factory=dict)
    protein_scales: ProteinParams = field(default_factory=dict)

@dataclass(frozen=True)
class ParentContribution:
    parent_name: str
    combine_op: CombineOp
    transform: TransformName
    weight: float

@dataclass(frozen=True)
class GeneratedAttributeParameters:
    spec: SimpleProteinAttribute
    protein_params: ProteinParams
    protein_scales: ProteinParams
    parent_contributions: dict[str, ParentContribution]


TRANSFORMS: tuple[TransformName, ...] = ("square", "tanh")
COMBINE_OPS: tuple[CombineOp, ...] = ("add", "multiply")


def apply_transform(transform_name: TransformName, x: float) -> float:
    if transform_name == "square":
        return x**2
    if transform_name == "tanh":
        return float(np.tanh(x))
    raise ValueError(f"Unknown transform: {transform_name}")

def transform_expression(transform_name: TransformName, attr_name: str) -> str:
    if transform_name == "square":
        return f"({attr_name})^2"
    if transform_name == "tanh":
        return f"tanh({attr_name})"
    raise ValueError(f"Unknown transform: {transform_name}")

def combine_numeric_terms(terms: list[float], op: CombineOp, contribution: float) -> None:
    if op == "add":
        terms.append(contribution)
    elif op == "multiply":
        if not terms:
            terms.append(contribution)
        else:
            terms[-1] = terms[-1] * contribution
    else:
        raise ValueError(f"Unknown combine operation: {op}")

def combine_formula_terms(terms: list[str], op: CombineOp, contribution: str) -> None:
    if op == "add":
        terms.append(contribution)
    elif op == "multiply":
        if not terms:
            terms.append(contribution)
        else:
            terms[-1] = f"({terms[-1]}) * ({contribution})"
    else:
        raise ValueError(f"Unknown combine operation: {op}")

def _validate_inputs(
    proteins: list[str],
    simple_attributes: list[SimpleProteinAttribute],
) -> None:
    if not proteins:
        raise ValueError("At least one protein class is required.")

    attr_names = [attr.name for attr in simple_attributes]
    if len(set(attr_names)) != len(attr_names):
        raise ValueError("Attribute names must be unique.")

    attr_name_set = set(attr_names)
    for attr in simple_attributes:
        missing = [name for name in attr.pre_reqs if name not in attr_name_set]
        if missing:
            raise ValueError(f"Attribute {attr.name!r} has unknown prerequisites: {missing}.")

def _generate_protein_params(
    proteins: list[str],
    rng: np.random.Generator,
) -> ProteinParams:
    return {protein: float(rng.normal()) for protein in proteins}

def _generate_protein_scales(
        proteins: list[str],
        rng: np.random.Generator,
) -> ProteinParams:
    return {protein: abs(float(rng.normal())) + 0.1 for protein in proteins}

def _choose_combine_op(rng: np.random.Generator) -> CombineOp:
    return cast(CombineOp, str(rng.choice(COMBINE_OPS)))

def _choose_transform(rng: np.random.Generator) -> TransformName:
    return cast(TransformName, str(rng.choice(TRANSFORMS)))

def _generate_parent_contributions(
    spec: SimpleProteinAttribute,
    rng: np.random.Generator,
) -> dict[str, ParentContribution]:
    contributions: dict[str, ParentContribution] = {}

    for parent_name in spec.pre_reqs:
        contributions[parent_name] = ParentContribution(
            parent_name=parent_name,
            combine_op=_choose_combine_op(rng),
            transform=_choose_transform(rng),
            weight=float(rng.normal()),
        )
    return contributions

def _generate_attribute_parameters(
    spec: SimpleProteinAttribute,
    proteins: list[str],
    rng: np.random.Generator,
) -> GeneratedAttributeParameters:
    return GeneratedAttributeParameters(
        spec=spec,
        protein_params=_generate_protein_params(proteins, rng),
        protein_scales=_generate_protein_scales(proteins, rng),
        parent_contributions=_generate_parent_contributions(spec, rng),
    )

def _build_sample_fn(
        params: GeneratedAttributeParameters,
        rng: np.random.Generator,
) -> SampleFn:
    spec = params.spec

    def sample_continuous(dependency_values: SampleInput) -> str:
        protein = dependency_values["Protein"]
        mu_terms: list[float] = [params.protein_params[protein]]
        scale = params.protein_scales[protein]

        for parent_name in spec.pre_reqs:
            config = params.parent_contributions[parent_name]
            parent_val = float(dependency_values[parent_name])

            contribution = config.weight * apply_transform(config.transform, parent_val)

            combine_numeric_terms(
                terms=mu_terms,
                op=config.combine_op,
                contribution=contribution,
            )

        mu = float(sum(mu_terms))
        sample = rng.normal(loc=mu, scale=scale)
        return str(sample)

    return sample_continuous

def _format_float(value: float) -> str:
    return f"{value:.4f}"

def _describe_attribute(params: GeneratedAttributeParameters) -> str:
    spec = params.spec

    formula_terms = ["protein_mean[Protein]"]
    for parent_name in spec.pre_reqs:
        config = params.parent_contributions[parent_name]
        expr = f"{config.weight:.4f} * {transform_expression(config.transform, parent_name)}"
        combine_formula_terms(formula_terms, config.combine_op, expr)

    lines: list[str] = [
        f"Attribute: {spec.name}",
        "Distribution:",
        f"  {spec.name} | Protein{', ' + ', '.join(spec.pre_reqs) if spec.pre_reqs else ''} ~ Normal(mu, protein_variance[Protein])",
        "Formula:",
        "  mu = " + " + ".join(formula_terms),
        "Protein Parameters:"
    ]

    for protein, mean_val in params.protein_params.items():
        scale_val = params.protein_scales[protein]
        variance_val = scale_val ** 2
        lines.append(f"  {protein}: mean = {_format_float(mean_val)}, variance = {_format_float(variance_val)}")

    return "\n".join(lines)

def make_attributes(
        proteins: list[str],
        simple_attributes: list[SimpleProteinAttribute],
        seed: int | None = None,
) -> list[ProteinAttribute]:
    _validate_inputs(proteins=proteins, simple_attributes=simple_attributes)

    rng = np.random.default_rng(seed)
    built_attrs: list[ProteinAttribute] = []

    for spec in simple_attributes:
        params = _generate_attribute_parameters(
            spec=spec,
            proteins=proteins,
            rng=rng,
        )
        sample_fn = _build_sample_fn(params=params, rng=rng)
        description = _describe_attribute(params)

        built_attrs.append(
            ProteinAttribute(
                name=spec.name,
                pre_req=list(spec.pre_reqs),
                sample_fn=sample_fn,
                description=description,
                protein_params=params.protein_params,
                protein_scales=params.protein_scales,
            )
        )

    return built_attrs

def topological_order(attributes: list[ProteinAttribute]) -> list[ProteinAttribute]:
    order: list[ProteinAttribute] = []
    visited: set[str] = set()

    while len(order) < len(attributes):
        curr_len = len(order)
        for attr in attributes:
            if attr.name not in visited and all(req in visited for req in attr.pre_req):
                order.append(attr)
                visited.add(attr.name)
        if len(order) <= curr_len:
            raise ValueError("Attributes don't have a clean topological order.")
    return order

def sample_protein_unordered(protein: str, attributes: list[ProteinAttribute]) -> dict[str, str]:
    known: dict[str, str] = {"Protein": protein}
    for attr in topological_order(attributes):
        known[attr.name] = attr.sample_fn(known)
    return known