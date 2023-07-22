import yaml


def read_yaml_config(path_to_config: str) -> dict:
    with open(path_to_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print()
        except yaml.YAMLError as exc:
            print(exc)
    return config
