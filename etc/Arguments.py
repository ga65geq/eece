import argparse
import os
from os import path

import yaml


def add_args():
    paser = argparse.ArgumentParser(description="Class Exclusion")
    paser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    args, unknown = paser.parse_known_args()
    return args

class Arguments:
    def __init__(self, cmd_args):
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)
        self.configuration = self.load_yaml_config(cmd_args.yaml_config_file)
        self.set_attr_from_config(self.configuration)

    def load_yaml_config(self, yaml_path):
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def set_attr_from_config(self, configuration):
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)


if __name__ == "__main__":
    cmd_args = add_args()
    args = Arguments(cmd_args)
    print(args.yaml_config_file)

