import sys
import os

PROJECT_FOLDER = os.path.dirname(__file__).replace('/src', '/pipeline')
PARENT_PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
sys.path.append(PARENT_PROJECT_FOLDER)

from src.pipeline.passt_model import PasstModel
from src.pipeline.effat_model import EffAtModel
from src.pipeline.vggish_model import VggishModel
from src.pipeline.dataset import EcossDataset
from src.pipeline.utils import create_exp_dir, load_yaml
from dotenv import load_dotenv



if __name__ == "__main__":
    load_dotenv()
    ANNOTATIONS_PATHS = os.getenv("ANNOTATIONS_PATHS").split(',')
    YAML_PATH = os.getenv("YAML_PATH")
    MODEL_TYPE = os.getenv("MODEL_TYPE")
    EXP_NAME = os.getenv("EXP_NAME")
    NAME_MODEL = os.getenv("NAME_MODEL")
    NEW_ONTOLOGY = os.getenv("NEW_ONTOLOGY").split(',')
    UNWANTED_LABELS = os.getenv("UNWANTED_LABELS").split(',')
    TEST_SIZE = os.getenv("TEST_SIZE")

    sr =32000
    ecoss_list = []
    yaml_content = load_yaml(YAML_PATH)
    for annot_path in ANNOTATIONS_PATHS:
        print(annot_path)
        ecoss_data1 = EcossDataset(annot_path, 'data/', 'zeros', sr, 1,"wav")
        ecoss_data1.add_file_column()
        ecoss_data1.fix_onthology(labels=NEW_ONTOLOGY)
        ecoss_data1.filter_overlapping()
        ecoss_data1.drop_unwanted_labels(UNWANTED_LABELS)
        ecoss_list.append(ecoss_data1)

    ecoss_data = EcossDataset.concatenate_ecossdataset(ecoss_list)
    length_prior_filter = len(ecoss_data.df)
    ecoss_data.filter_lower_sr()
    ecoss_data.generate_insights()
    ecoss_data.split_train_test_balanced(test_size=TEST_SIZE, random_state=27)
    signals,labels,split_info = ecoss_data.process_all_data()
    data_path = ecoss_data.path_store_data


    results_folder = create_exp_dir(name = EXP_NAME, model=MODEL_TYPE, task= "train")

    num_classes = len(ecoss_data.df["final_source"].unique())
    print(f"THE NUMBER OF CLASSES IS {num_classes}\n")

    if MODEL_TYPE.lower() == "passt":
        model = PasstModel(yaml_content=yaml_content,data_path=data_path)
    elif MODEL_TYPE.lower() == "effat":
        model = EffAtModel(yaml_content=yaml_content,data_path=data_path, name_model=NAME_MODEL, num_classes=num_classes)
    elif MODEL_TYPE.lower() == "vggish":
        model = VggishModel(yaml_content=yaml_content,data_path=data_path, signals=signals, labels=labels, split_info=split_info, sample_rate = sr)

    model.plot_processed_data()
    model.train(results_folder = results_folder)



