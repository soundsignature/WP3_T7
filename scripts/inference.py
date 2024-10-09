import sys
import os

PROJECT_FOLDER = os.path.dirname(__file__).replace('/src', '/pipeline')
PARENT_PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
sys.path.append(PARENT_PROJECT_FOLDER)

from src.pipeline.passt_model import PasstModel
from src.pipeline.effat_model import EffAtModel
from src.pipeline.vggish_model import VggishModel
from src.pipeline.dataset import EcossDataset
from src.pipeline.utils import create_exp_dir, load_yaml, visualize_inference

import json
from matplotlib import pyplot as plt
from dotenv import load_dotenv


def main():
    load_dotenv()
    YAML_PATH = os.getenv("YAML_PATH")
    MODEL_TYPE = os.getenv("MODEL_TYPE")
    EXP_NAME = os.getenv("EXP_NAME")
    PATH_MODEL_TEST = os.getenv("PATH_MODEL_TEST")
    INFERENCE_DATA_PATH = os.getenv("INFERENCE_DATA_PATH")


    results_folder = create_exp_dir(name=EXP_NAME, model=MODEL_TYPE, task="inference")
    yaml_content = load_yaml(YAML_PATH)

    if MODEL_TYPE.lower() == "passt":
        model = PasstModel(yaml_content=yaml_content,data_path=INFERENCE_DATA_PATH)
    elif MODEL_TYPE.lower() == "effat":
        # Get the number of classes of the model
        with open(PATH_MODEL_TEST.replace("model.pth", "class_dict.json"), 'r') as f:
            class_map = json.load(f)
        model = EffAtModel(yaml_content=yaml_content,data_path=INFERENCE_DATA_PATH, num_classes=len(class_map))
    elif MODEL_TYPE.lower() == "vggish":
        model = VggishModel(yaml_content=yaml_content,data_path=INFERENCE_DATA_PATH)


    model.inference(results_folder=results_folder, path_model=PATH_MODEL_TEST, path_data=INFERENCE_DATA_PATH)
    visualize_inference(path_json=os.path.join(model.results_folder, 'predictions.json'), path_audio=INFERENCE_DATA_PATH, path_yaml=YAML_PATH, model=MODEL_TYPE.lower())

if __name__ == "__main__":
    main()

