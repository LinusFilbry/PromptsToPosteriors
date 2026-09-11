import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np

from helpers import normalize_prob_dict
from protein_world.attribute import (
    CombineOp,
    ProteinAttribute,
    TransformName,
    apply_transform,
    combine_formula_terms,
    combine_numeric_terms,
    topological_order,
    transform_expression,
)


def normal_pdf(x: float, mean: float, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("Normal scale must be positive.")

    z = (x - mean) / scale
    return float(np.exp(-0.5 * z * z) / (scale * np.sqrt(2.0 * np.pi)))


def normal_cdf(x: float, mean: float, scale: float) -> float:
    z = (x - mean) / scale
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@dataclass
class Experiment(ABC):
    name: str
    description: str
    attribute: ProteinAttribute

    @abstractmethod
    def likelihood(self, observed_value: float, protein: str) -> float:
        pass

    def posterior(
            self,
            observed_value: float,
            proteins: list[str],
            prior: dict[str, float],
    ) -> dict[str, float]:
        unnormalized = {
            p: prior[p] * self.likelihood(observed_value, p)
            for p in proteins
        }
        return normalize_prob_dict(unnormalized)

    @abstractmethod
    def sample(self, instance: dict[str, str | float]) -> float:
        pass


@dataclass
class SimpleExperiment(Experiment):
    def __init__(self, name: str, attribute: ProteinAttribute):
        self.name = name
        self.attribute = attribute

        if attribute.pre_req:
            raise ValueError(
                f"SimpleExperiment '{name}' requires a root attribute. "
                f"Attribute '{attribute.name}' has prerequisites: {attribute.pre_req}."
            )

        if not attribute.protein_params:
            raise ValueError(
                f"Attribute '{attribute.name}' does not contain protein parameters."
            )

        self.description = (
            f"Experiment '{name}' reveals the exact value of attribute '{attribute.name}'."
        )

    def likelihood(self, observed_value: float, protein: str) -> float:
        if protein not in self.attribute.protein_params:
            raise KeyError(f"Unknown protein class {protein!r}.")

        protein_param = self.attribute.protein_params[protein]
        protein_scale = self.attribute.protein_scales[protein]

        x = float(observed_value)
        mean = float(protein_param)

        return normal_pdf(x=x, mean=mean, scale=protein_scale)

    def sample(self, instance: dict[str, str | float]) -> float:
        return float(instance[self.attribute.name])


@dataclass
class NoisySimpleExperiment(Experiment):
    noise: float  # std for uniform noise interval

    def __init__(self, name: str, attribute: ProteinAttribute, noise: float):
        self.name = name
        self.attribute = attribute
        self.noise = float(noise)

        if attribute.pre_req:
            raise ValueError(
                f"NoisySimpleExperiment '{name}' requires a root attribute. "
                f"Attribute '{attribute.name}' has prerequisites: {attribute.pre_req}."
            )

        if noise < 0:
            raise ValueError("Noise must be non-negative.")

        self.description = (
            f"Experiment '{name}' measures '{attribute.name}' with uniform noise: "
            f"the reported value is uniformly distributed in "
            f"[true - {noise}, true + {noise}]."
        )

    def likelihood(self, observed_value: float, protein: str) -> float:
        if protein not in self.attribute.protein_params:
            raise KeyError(f"Unknown protein class {protein!r}.")

        param = self.attribute.protein_params[protein]
        scale = self.attribute.protein_scales[protein]

        y = float(observed_value)
        mu = float(param)

        if self.noise == 0:
            return normal_pdf(x=y, mean=mu, scale=scale)

        lower = y - self.noise
        upper = y + self.noise

        cdf_upper = normal_cdf(upper, mean=mu, scale=scale)
        cdf_lower = normal_cdf(lower, mean=mu, scale=scale)

        return (cdf_upper - cdf_lower) / (2 * self.noise)

    def sample(self, instance: dict[str, str | float]) -> float:
        true_value = float(instance[self.attribute.name])
        if self.noise == 0:
            return true_value
        return float(np.random.uniform(true_value - self.noise, true_value + self.noise))


class SlightlyComplexExperiment(Experiment):
    def __init__(
            self,
            name: str,
            attribute: ProteinAttribute,
            all_attributes: list[ProteinAttribute],
            num_mc_samples: int,
            noise: float,
            rng: np.random.Generator
    ):
        self.name = name
        self.attribute = attribute
        self.noise = float(noise)
        self.num_mc_samples = num_mc_samples
        self.rng = rng
        self._sample_cache: dict[str, np.ndarray] = {}

        if not self.attribute.pre_req:
            raise ValueError(
                f"SlightlyComplexExperiment '{name}' requires an attribute with at least one prerequisite. "
                f"Attribute '{attribute.name}' has none."
            )

        if self.noise < 0:
            raise ValueError("Noise must be non-negative.")

        self.topological_order = topological_order(all_attributes)

        desc_noise = (
            f"with uniform noise: reported value is in [true - {self.noise}, true + {self.noise}]."
        ) if self.noise > 0 else "exactly."

        self.description = (
            f"Experiment '{name}' reveals the value of attribute '{attribute.name}' {desc_noise}"
        )

    def sample(self, instance: dict[str, str | float]) -> float:
        true_value = float(instance[self.attribute.name])
        if self.noise == 0:
            return true_value
        return float(self.rng.uniform(true_value - self.noise, true_value + self.noise))

    def _attribute_samples(self, protein: str) -> np.ndarray:
        if protein not in self._sample_cache:
            N = self.num_mc_samples
            samples = np.zeros(N)
            for i in range(N):
                known: dict[str, str] = {"Protein": protein}
                for attr in self.topological_order:
                    known[attr.name] = attr.sample_fn(known)
                samples[i] = float(known[self.attribute.name])
            self._sample_cache[protein] = samples
        return self._sample_cache[protein]

    def likelihood(self, observed_value: float, protein: str) -> float:
        samples = self._attribute_samples(protein)
        N = len(samples)
        y = float(observed_value)

        if self.noise == 0:
            std = np.std(samples) or 1.0
            h = 1.06 * std * (N ** -0.2)
            z = (y - samples) / h
            densities = np.exp(-0.5 * z**2) / (h * np.sqrt(2 * np.pi))
            return float(np.mean(densities))
        else:
            lower, upper = y - self.noise, y + self.noise
            hits = np.sum((samples >= lower) & (samples <= upper))
            return float((hits / N) / (2 * self.noise))


ComplexExperimentType: TypeAlias = Literal["gaussian", "uniform", "sigmoid"]

@dataclass(frozen=True)
class ExperimentContribution:
    attr_name: str
    combine_op: CombineOp
    transform: TransformName
    weight: float

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def sigmoid_array(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (1.0 + z)
    return out


class ComplexExperiment(Experiment):
    def __init__(
            self,
            name: str,
            dependencies: list[str],
            all_attributes: list[ProteinAttribute],
            num_mc_samples: int,
            rng: np.random.Generator,
            exp_type: ComplexExperimentType | None = None,
    ):
        self.name = name
        self.rng = rng
        self.num_mc_samples = num_mc_samples
        self.exp_type = exp_type or cast(ComplexExperimentType, str(self.rng.choice(["gaussian", "uniform", "sigmoid"])))

        self.attr_map = {attr.name: attr for attr in all_attributes}

        self.topological_order = topological_order(all_attributes)

        self.sigma = float(self.rng.uniform(0.3, 0.7))
        self.contributions = self._generate_contributions(dependencies)

        self.attribute = self.attr_map[dependencies[0]] if dependencies else all_attributes[0]
        self.description = self._build_description(dependencies)

        self._mu_cache: dict[str, np.ndarray] = {}

    def _generate_contributions(self, dependencies: list[str]) -> list[ExperimentContribution]:
        contributions = []
        for i, attr_name in enumerate(dependencies):
            combine_op = cast(CombineOp, "add" if i == 0 else str(self.rng.choice(["add", "multiply"])))
            transform = cast(TransformName, str(self.rng.choice(["square", "tanh"])))
            weight = float(self.rng.normal())
            contributions.append(ExperimentContribution(attr_name, combine_op, transform, weight))
        return contributions

    def _build_description(self, dependencies: list[str]) -> str:
        lines = [f"Experiment: {self.name}", f"Dependencies: {', '.join(dependencies)}", "Distribution:"]
        if self.exp_type == "gaussian":
            lines.append(f"  y ~ Normal(mu, {self.sigma:.4f}^2)")
        elif self.exp_type == "uniform":
            lines.append(f"  y ~ Uniform(mu - {self.sigma:.4f}, mu + {self.sigma:.4f})")
        elif self.exp_type == "sigmoid":
            lines.append("  y ~ Bernoulli(sigmoid(mu))")

        formula_terms: list[str] = []
        for c in self.contributions:
            expr = f"{c.weight:.4f} * {transform_expression(c.transform, c.attr_name)}"
            combine_formula_terms(formula_terms, c.combine_op, expr)

        lines.append("  mu = " + " + ".join(formula_terms))
        return "\n".join(lines)

    def _evaluate_formula(self, instance: dict[str, str | float]) -> float:
        terms: list[float] = []
        for contrib in self.contributions:
            raw_val = float(instance[contrib.attr_name])
            transformed = apply_transform(contrib.transform, raw_val)
            term_val = transformed * contrib.weight
            combine_numeric_terms(terms, contrib.combine_op, term_val)
        return float(sum(terms)) if terms else 0.0

    def sample(self, instance: dict[str, str | float]) -> float:
        mu = self._evaluate_formula(instance)
        if self.exp_type == "gaussian":
            return float(self.rng.normal(loc=mu, scale=self.sigma))
        if self.exp_type == "uniform":
            return float(self.rng.uniform(mu - self.sigma, mu + self.sigma))
        if self.exp_type == "sigmoid":
            return 1.0 if self.rng.random() < sigmoid(mu) else 0.0
        raise ValueError(f"Unknown type: {self.exp_type}")

    def _mu_samples(self, protein: str) -> np.ndarray:
        if protein not in self._mu_cache:
            mus = np.empty(self.num_mc_samples)
            for i in range(self.num_mc_samples):
                known: dict[str, str] = {"Protein": protein}
                for attr in self.topological_order:
                    known[attr.name] = attr.sample_fn(known)
                mus[i] = self._evaluate_formula(known)
            self._mu_cache[protein] = mus
        return self._mu_cache[protein]

    def likelihood(self, observed_value: float, protein: str) -> float:
        mus = self._mu_samples(protein)
        y = float(observed_value)
        if self.exp_type == "gaussian":
            z = (y - mus) / self.sigma
            densities = np.exp(-0.5 * z**2) / (self.sigma * np.sqrt(2.0 * np.pi))
            return float(np.mean(densities))
        if self.exp_type == "uniform":
            hits = (mus - self.sigma <= y) & (y <= mus + self.sigma)
            return float(np.mean(hits) / (2.0 * self.sigma))
        if self.exp_type == "sigmoid":
            s = sigmoid_array(mus)
            probs = s if y == 1.0 else (1.0 - s)
            return float(np.mean(probs))
        raise ValueError(f"Unknown type: {self.exp_type}")


def generate_experiments(exp_type: str, num_exp: int, exp_names: list[str], attributes: list[ProteinAttribute],
                         noise: float, seed: int, num_mc_samples: int) -> list[Experiment]:
    rng = np.random.default_rng(seed)
    match exp_type:
        case "simple" | "simple_noisy":
            shuffled_attrs = list(rng.permutation(attributes))
            generated_exps: list[Experiment] = []
            count = 0
            for attr in shuffled_attrs:
                if count == num_exp:
                    break
                if not attr.pre_req:
                    name = exp_names[count]
                    if exp_type == "simple":
                        new_exp = SimpleExperiment(name=name, attribute=attr)
                    else:
                        new_exp = NoisySimpleExperiment(name=name, attribute=attr, noise=noise)
                    generated_exps.append(new_exp)
                    count += 1
            if count < num_exp:
                raise ValueError(f"Needed {num_exp} root attributes, but only found {count}.")
            return generated_exps

        case "slightly_complex":
            shuffled_attrs = list(rng.permutation(attributes))
            generated_exps: list[Experiment] = []
            count = 0
            for attr in shuffled_attrs:
                if count == num_exp:
                    break
                if attr.pre_req:
                    generated_exps.append(SlightlyComplexExperiment(
                        name=exp_names[count],
                        attribute=attr,
                        all_attributes=attributes,
                        num_mc_samples=num_mc_samples,
                        noise=noise,
                        rng=rng
                    ))
                    count += 1
            if count < num_exp:
                raise ValueError(f"Needed {num_exp} dependent attributes, but only found {count}.")
            return generated_exps

        case "complex":
            generated_exps: list[Experiment] = []
            for i in range(num_exp):
                selected_deps = []
                while len(selected_deps) < 2:
                    selected_deps = [
                        attr.name for attr in attributes
                        if rng.random() < 0.6
                    ]

                generated_exps.append(ComplexExperiment(
                    name=exp_names[i],
                    dependencies=selected_deps,
                    all_attributes=attributes,
                    num_mc_samples=num_mc_samples,
                    rng=rng
                ))
            return generated_exps

        case _:
            raise ValueError(f"Unknown experiment type: {exp_type}")


