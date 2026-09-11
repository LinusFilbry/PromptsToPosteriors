from helpers import convert_string_to_array, write_to_log
from models import Model
from protein_world.attribute import SimpleProteinAttribute, make_attributes
from protein_world.config import Config
from protein_world.experiment import generate_experiments
from protein_world.prompts import build_world_string, create_experiment_names, generate_latent_attribute_names, \
    generate_protein_names, world_setup_messages
from protein_world.showcase_world_setup import showcase_world_setup


def parse_attributes(attr_str: str, pre_req_list) -> list[SimpleProteinAttribute]:
    lines = [ln.strip() for ln in attr_str.splitlines() if ln.strip() != ""]
    if len(pre_req_list) != len(lines):
        raise ValueError(f"Length of pre_req_list {len(pre_req_list)} does not match number of generated protein names {len(lines)}.")

    attributes: list[SimpleProteinAttribute] = []

    for attr_name, pre_reqs_curr in zip(lines, pre_req_list):
        pre_req_names = [attribute.name for i, attribute in enumerate(attributes) if i in pre_reqs_curr]
        attributes.append(SimpleProteinAttribute(name=attr_name, pre_reqs=pre_req_names))

    return attributes


def protein_world_setup(model: Model, config: Config, world_iteration: int):
    if config.do_showcase_world:
        return showcase_world_setup(config, world_iteration)

    # prompt for protein names
    messages = world_setup_messages(generate_protein_names(config.num_protein_classes))
    protein_names = convert_string_to_array(model.chat_complete(messages, config.answer_temperature, 1)[0])

    # prompt for simple attributes, convert to complex and log them
    messages = world_setup_messages(generate_latent_attribute_names(config.num_attr))
    simple_attributes = parse_attributes(model.chat_complete(messages, config.answer_temperature, 1)[0], config.pre_req_list)
    attributes = make_attributes(protein_names, simple_attributes, config.seed + world_iteration)

    messages = world_setup_messages(create_experiment_names(config.num_experiments))
    experiment_names = convert_string_to_array(model.chat_complete(messages, config.answer_temperature, 1)[0])
    experiments = generate_experiments(config.exp_type, config.num_experiments, experiment_names, attributes, config.exp_noise, config.seed + world_iteration, config.num_mc_samples)

    world_building_string = build_world_string(protein_names, attributes, experiments)

    write_to_log(world_building_string)

    return protein_names, attributes, experiments, world_building_string
