from __future__ import annotations

import numpy as np

from helpers import write_to_log
from protein_world.attribute import (
    GeneratedAttributeParameters,
    ParentContribution,
    ProteinAttribute,
    SimpleProteinAttribute,
    _build_sample_fn,
    _describe_attribute,
)
from protein_world.config import Config
from protein_world.experiment import ComplexExperiment, Experiment, ExperimentContribution, SimpleExperiment
from protein_world.prompts import build_world_string

SHOWCASE_WORLD_ERROR = "Showcase worlds only exist for exp_type 'simple' or 'complex'."


def _make_attribute(
        name: str,
        pre_reqs: list[str],
        protein_means: dict[str, float],
        protein_variances: dict[str, float],
        parent_contributions: dict[str, ParentContribution],
        rng: np.random.Generator,
) -> ProteinAttribute:
    spec = SimpleProteinAttribute(name=name, pre_reqs=pre_reqs)
    protein_scales = {protein: float(np.sqrt(variance)) for protein, variance in protein_variances.items()}
    params = GeneratedAttributeParameters(
        spec=spec,
        protein_params=protein_means,
        protein_scales=protein_scales,
        parent_contributions=parent_contributions,
    )
    return ProteinAttribute(
        name=name,
        pre_req=list(pre_reqs),
        sample_fn=_build_sample_fn(params, rng),
        description=_describe_attribute(params),
        protein_params=protein_means,
        protein_scales=protein_scales,
    )


def _make_complex_experiment(
        name: str,
        dependencies: list[str],
        all_attributes: list[ProteinAttribute],
        exp_type: str,
        sigma: float,
        contributions: list[ExperimentContribution],
        num_mc_samples: int,
        rng: np.random.Generator,
) -> ComplexExperiment:
    experiment = ComplexExperiment(
        name=name,
        dependencies=dependencies,
        all_attributes=all_attributes,
        num_mc_samples=num_mc_samples,
        rng=rng,
        exp_type=exp_type,
    )
    experiment.sigma = sigma
    experiment.contributions = contributions
    experiment.attribute = experiment.attr_map[dependencies[0]]
    experiment.description = experiment._build_description(dependencies)
    return experiment


def _showcase_world_simple(config: Config, world_iteration: int) -> tuple[list[str], list[ProteinAttribute], list[Experiment], str]:
    rng = np.random.default_rng(config.seed + world_iteration)

    protein_names = ["Xylocorein", "Thalassogen", "Nebulase", "Cryovitin"]

    fold_elasticity = _make_attribute(
        name="Fold-Elasticity Coefficient",
        pre_reqs=[],
        protein_means={"Xylocorein": 0.2442, "Thalassogen": 0.6782, "Nebulase": -0.5855, "Cryovitin": -0.9087},
        protein_variances={"Xylocorein": 4.3758, "Thalassogen": 1.1484, "Nebulase": 0.0136, "Cryovitin": 0.0935},
        parent_contributions={},
        rng=rng,
    )
    helix_torsion = _make_attribute(
        name="Helix-Torsion Index",
        pre_reqs=[],
        protein_means={"Xylocorein": -0.7836, "Thalassogen": 1.2265, "Nebulase": 0.9432, "Cryovitin": -0.1218},
        protein_variances={"Xylocorein": 0.4302, "Thalassogen": 0.2086, "Nebulase": 0.8075, "Cryovitin": 0.0160},
        parent_contributions={},
        rng=rng,
    )
    ligand_permeability = _make_attribute(
        name="Ligand-Permeability Score",
        pre_reqs=[],
        protein_means={"Xylocorein": -0.6162, "Thalassogen": 0.5855, "Nebulase": -0.3595, "Cryovitin": -0.3540},
        protein_variances={"Xylocorein": 2.6806, "Thalassogen": 0.6818, "Nebulase": 2.7819, "Cryovitin": 0.2174},
        parent_contributions={},
        rng=rng,
    )
    attributes = [fold_elasticity, helix_torsion, ligand_permeability]

    experiments: list[Experiment] = [
        SimpleExperiment(name="ProtFold-X Analysis", attribute=helix_torsion),
        SimpleExperiment(name="KinetiBind Protein Assay", attribute=ligand_permeability),
        SimpleExperiment(name="HelixScan Stability Trial", attribute=fold_elasticity),
    ]

    return protein_names, attributes, experiments, build_world_string(protein_names, attributes, experiments)


def _showcase_world_complex(config: Config, world_iteration: int) -> tuple[list[str], list[ProteinAttribute], list[Experiment], str]:
    rng = np.random.default_rng(config.seed + world_iteration)

    protein_names = ["Chronocortin", "Glycorectin", "Luminase", "Somatoplexin"]

    chirotropic_torsion = _make_attribute(
        name="Chirotropic Torsion Index",
        pre_reqs=[],
        protein_means={"Chronocortin": 0.2442, "Glycorectin": 0.6782, "Luminase": -0.5855, "Somatoplexin": -0.9087},
        protein_variances={"Chronocortin": 4.3758, "Glycorectin": 1.1484, "Luminase": 0.0136, "Somatoplexin": 0.0935},
        parent_contributions={},
        rng=rng,
    )
    allosteric_resonance = _make_attribute(
        name="Allosteric Resonance Factor",
        pre_reqs=["Chirotropic Torsion Index"],
        protein_means={"Chronocortin": -0.7836, "Glycorectin": 1.2265, "Luminase": 0.9432, "Somatoplexin": -0.1218},
        protein_variances={"Chronocortin": 0.4302, "Glycorectin": 0.2086, "Luminase": 0.8075, "Somatoplexin": 0.0160},
        parent_contributions={
            "Chirotropic Torsion Index": ParentContribution(
                parent_name="Chirotropic Torsion Index", combine_op="add", transform="square", weight=0.5855,
            ),
        },
        rng=rng,
    )
    zymotic_degradation = _make_attribute(
        name="Zymotic Degradation Propensity",
        pre_reqs=["Chirotropic Torsion Index", "Allosteric Resonance Factor"],
        protein_means={"Chronocortin": -0.3595, "Glycorectin": -0.3540, "Luminase": 1.5373, "Somatoplexin": -0.7257},
        protein_variances={"Chronocortin": 2.7819, "Glycorectin": 0.2174, "Luminase": 4.7431, "Somatoplexin": 0.1328},
        parent_contributions={
            "Chirotropic Torsion Index": ParentContribution(
                parent_name="Chirotropic Torsion Index", combine_op="multiply", transform="square", weight=0.0701,
            ),
            "Allosteric Resonance Factor": ParentContribution(
                parent_name="Allosteric Resonance Factor", combine_op="add", transform="square", weight=-2.0506,
            ),
        },
        rng=rng,
    )
    attributes = [chirotropic_torsion, allosteric_resonance, zymotic_degradation]

    experiments: list[Experiment] = [
        _make_complex_experiment(
            name="Dynamic Refolding Flux Analysis",
            dependencies=["Allosteric Resonance Factor", "Zymotic Degradation Propensity"],
            all_attributes=attributes,
            exp_type="gaussian",
            sigma=0.5349,
            contributions=[
                ExperimentContribution(attr_name="Allosteric Resonance Factor", combine_op="add", transform="tanh", weight=0.9716),
                ExperimentContribution(attr_name="Zymotic Degradation Propensity", combine_op="multiply", transform="tanh", weight=0.2057),
            ],
            num_mc_samples=config.num_mc_samples,
            rng=rng,
        ),
        _make_complex_experiment(
            name="Kinetic Interaction Capture System",
            dependencies=["Chirotropic Torsion Index", "Allosteric Resonance Factor"],
            all_attributes=attributes,
            exp_type="uniform",
            sigma=0.4115,
            contributions=[
                ExperimentContribution(attr_name="Chirotropic Torsion Index", combine_op="add", transform="tanh", weight=-0.3567),
                ExperimentContribution(attr_name="Allosteric Resonance Factor", combine_op="multiply", transform="square", weight=0.0266),
            ],
            num_mc_samples=config.num_mc_samples,
            rng=rng,
        ),
        _make_complex_experiment(
            name="Post-Translational Cascade Mapping",
            dependencies=["Chirotropic Torsion Index", "Allosteric Resonance Factor"],
            all_attributes=attributes,
            exp_type="sigmoid",
            sigma=0.0,
            contributions=[
                ExperimentContribution(attr_name="Chirotropic Torsion Index", combine_op="add", transform="tanh", weight=-2.0779),
                ExperimentContribution(attr_name="Allosteric Resonance Factor", combine_op="add", transform="square", weight=0.9817),
            ],
            num_mc_samples=config.num_mc_samples,
            rng=rng,
        ),
    ]

    return protein_names, attributes, experiments, build_world_string(protein_names, attributes, experiments)


def showcase_world_setup(config: Config, world_iteration: int) -> tuple[list[str], list[ProteinAttribute], list[Experiment], str]:
    if config.exp_type == "simple":
        protein_names, attributes, experiments, world_building_string = _showcase_world_simple(config, world_iteration)
    elif config.exp_type == "complex":
        protein_names, attributes, experiments, world_building_string = _showcase_world_complex(config, world_iteration)
    else:
        write_to_log(SHOWCASE_WORLD_ERROR + "\n")
        raise SystemExit(SHOWCASE_WORLD_ERROR)


    write_to_log(f"Re-Generating {config.exp_type} showcase world\n\n")
    write_to_log(world_building_string)

    return protein_names, attributes, experiments, world_building_string
