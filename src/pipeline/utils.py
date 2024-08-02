
""" Script for useful functions """

import yaml

def load_yaml(yaml_path: str) -> dict:
    """Function used to load the yaml content. Useful for

    Args:
        yaml_path (str): The absolute path to the yaml

    Returns:
        dict: The yaml content
    """
    with open(yaml_path, 'r') as file:
        try:
            yaml_content = yaml.safe_load(file)
            return yaml_content
        except yaml.YAMLError as e:
            print(e)
            return None