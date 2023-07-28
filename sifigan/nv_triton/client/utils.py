import os
from typing import List

import yaml


def read_yaml_config(path_to_config: str) -> dict:
    with open(path_to_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print()
        except yaml.YAMLError as exc:
            print(exc)
    return config


def prepare_out_folder(out_folder_path: str, force_cleanup: bool):
    if os.path.exists(out_folder_path):
        if force_cleanup:
            # cleanup folder
            all_dir_content = os.listdir(out_folder_path)
            if len(all_dir_content) != 0:
                print("There are", len(all_dir_content), "objects in output dir. Cleaning it.")
            for elem in all_dir_content:
                absolute_path = os.path.join(out_folder_path, elem)
                if os.path.isfile(absolute_path):
                    os.remove(absolute_path)
    else:
        os.makedirs(out_folder_path, exist_ok=True)


def get_all_files_from_dir(dir_path: str) -> List[str]:
    all_dir_content = os.listdir(dir_path)
    all_files = []
    for elem in all_dir_content:
        absolute_path = os.path.join(dir_path, elem)
        if os.path.isfile(absolute_path):
            all_files.append(absolute_path)

    return all_files
