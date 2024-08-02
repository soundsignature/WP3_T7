from pipeline.passt_model import PasstModel
from pipeline.effat_model import EffAtModel
from pipeline.vggish_model import VggishModel
from pipeline.dataset import EcossDataset
from pipeline.utils import create_exp_dir, load_yaml
import os
from dotenv import load_dotenv



if __name__ == "__main__":
    load_dotenv()
    ANNOTATIONS_PATH = os.getenv("DATASET_PATH")
    ANNOTATIONS_PATH2 = os.getenv("DATASET_PATH2")
    ANNOTATIONS_PATH3 = os.getenv("DATASET_PATH3")
    YAML_PATH = os.getenv("YAML_PATH")
    MODEL_TYPE = os.getenv("MODEL_TYPE")
    EXP_NAME = os.getenv("EXP_NAME")
    # LABELS =
    ecoss_list = []
    for ANNOT_PATH in [ANNOTATIONS_PATH, ANNOTATIONS_PATH2, ANNOTATIONS_PATH3]:
        ecoss_data1 = EcossDataset(ANNOT_PATH, 'data/', 'zeros', 32000.0, 1,"wav")
        ecoss_data1.add_file_column()
        ecoss_data1.fix_onthology(labels=[])
        ecoss_data1.filter_overlapping()
        ecoss_list.append(ecoss_data1)
        
    ecoss_data = EcossDataset.concatenate_ecossdataset(ecoss_list)
    length_prior_filter = len(ecoss_data.df)
    ecoss_data.filter_lower_sr()
    assert length_prior_filter != len(ecoss_data.df), "The number of rows is the same"
    times = ecoss_data.generate_insights()
    ecoss_data.split_train_test_balanced(test_size=0.3, random_state=27)
    _, _, _  = ecoss_data.process_all_data()
    
    data_path = ecoss_data.path_store_data
    
    yaml_content = load_yaml(YAML_PATH)

    results_folder = create_exp_dir(name = EXP_NAME, model=MODEL_TYPE, task= "train")
    
    if MODEL_TYPE.lower() == "passt":
        model = PasstModel(yaml_content=yaml_content,data_path=data_path)
    elif MODEL_TYPE.lower() == "effat":
        model = EffAtModel(yaml_content=yaml_content,data_path=data_path)
    elif MODEL_TYPE.lower() == "vggish":
        model = VggishModel(yaml_content=yaml_content,data_path=data_path)
    
    model.train(results_folder = results_folder)
    
        
    