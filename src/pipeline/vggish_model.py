"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class VggishModel which is responsible for all the training, testing and inference stuff related with the
    Vggish + SVM model """
    
from src.pipeline.utils import *
import json
from tqdm import tqdm
import os
from sklearn.svm import SVC
import joblib
import logging
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import soundfile as sf
from src.models.VggishFeaturesExtractor import *
import shutil
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class VggishModel():
    def __init__(self, yaml_content: dict, signals: list, labels:list, split_info:list, sample_rate:float, data_path: str = None) -> None:
        self.yaml = yaml_content
        self.sample_rate=sample_rate
        self.signals = signals
        self.labels = labels
        self.split_info = split_info
        self.data_path = data_path
        self.features_extractor = VggishFeaturesExtractor(sample_rate = self.sample_rate)
        self.model  = SVC()


    def train(self, results_folder):
        """
        Compute the training. The data are processed by the feature extractor and the labels are encoded.
        The training is performed using the best parameters found by the gridsearch for the SVM.
        The model and the training plot are saved in the results_folder

        Parameters:
        results_folder (str): path to the folder where the results are saved
        
        Returns:
        None
        """
        train_data = self.get_data(X = self.signals, Y = self.labels, split_info = self.split_info, data_path = self.data_path, split="train")
        X, Y = self.data_preparation(data = train_data, saving_folder = results_folder)
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.yaml.get('grid_search_params'),
            error_score='raise',
            scoring='accuracy',
            cv=5,
            return_train_score=True
        )
        grid = grid_search.fit(X, Y)
        best_params = grid.best_params_
        # Set the best hyperparameters and fit the model
        self.model = self.model.set_params(**best_params)
        self.model = self.model.fit(X,Y)
        Y_pred = self.model.predict(X)
        self.save_weigths(saving_folder = results_folder)
        self.plot_results(set = 'train', saving_folder = results_folder, gridsearch = grid, y_true = Y, y_pred = Y_pred)


    def test(self, results_folder):
        """
        Compute the test. The data are processed by the feature extractor and the labels are encoded.
        The metrics "accuracy" and "macro F1" are computed and saved in the results_folder.
        The test plot are saved in results_folder.

        Parameters:
        results_folder (str): path to the folder where the results are saved
        
        Returns:
        None
        """
        test_data = self.get_data(X = self.signals, Y = self.labels, split_info = self.split_info, data_path = self.data_path, split="test")
        X, Y = self.data_preparation(data = test_data, saving_folder = results_folder)
        if self.yaml.get('model_path'):
            self.model = joblib.load(self.yaml.get('model_path'))
        Y_pred = self.model.predict(X)
        
        accuracy, f1 = self.compute_scores(Y, Y_pred)
        result = {'accuracy':accuracy, 'f1_score' : f1}
        with open(os.path.join(results_folder, 'metrics.json'), 'w') as json_file:
            json.dump(result, json_file)
        self.plot_results(set = 'test', saving_folder = results_folder, y_true = Y, y_pred = Y_pred)
        

    def inference(self,path_data,results_folder):
        sr = self.yaml.get('desired_sr')
        duration = self.yaml.get('desired_duration')
        X = process_data_for_inference(path_audio= path_data, desired_sr=sr, desired_duration=duration)
        if self.yaml.get('model_path'):
            self.model = joblib.load(self.yaml.get('model_path'))
        else:
            logger.error('Error. model_path missing in the yaml configuration file')
            exit()
        predictions = []
        for x in X:
            y = self.model.predict(x)
            labels_mapping_path = self.yaml.get('labels_mapping_path')
            with open(labels_mapping_path, 'r') as file:
                label_mapping = json.load(file)
                y= np.array([label_mapping[str(label)] for label in y])
                predictions.append(y)
        with open(os.path.join(results_folder, 'predictions.json'), "w") as f:
            json.dump(predictions, f)


    def plot_results(self, set, saving_folder, y_true, y_pred, gridsearch = None):
        """
        Compute the plots (Confusion Matrix/Training Curves) based on the procedure(train/test).

        Parameters:
        set (str): 'train' | 'test'
        saving_folder (str): path to the folder where the results are saved
        gridsearch (obj, optional): instance of fitted estimator.Defaults to None.
        y_true (array, optional): ground truth  
        y_pred (array, optional): labels predicted.
        
        Returns:
        None
        """
        if set == 'train':
            cf_matrix = ConfusionMatrix(labels_mapping_path = os.path.join(saving_folder, 'labels_mapping.json'))
            fig1 = cf_matrix.plot(y_true = y_true, y_pred = y_pred)
            cf_matrix.save_plot(plot = fig1, saving_folder = saving_folder)
            
            training_curves = ValidationPlot(gridsearch = gridsearch)
            fig2 = training_curves.plot()
            training_curves.save_plot(plot = fig2, saving_folder = saving_folder)
        else:
            cf_matrix = ConfusionMatrix(labels_mapping_path = os.path.join(saving_folder, 'labels_mapping.json'))
            fig1 = cf_matrix.plot(y_true = y_true, y_pred = y_pred)
            cf_matrix.save_plot(plot = fig1, saving_folder = saving_folder)


    def save_weigths(self,saving_folder):
        """
        Save the weigths of the model

        Parameters:
        saving_folder (str): path to the folder where the results are saved
        
        Returns:
        None
        """
        with open(os.path.join(saving_folder,'model.joblib'), 'wb') as f:
            joblib.dump(self.model, f)


    def plot_processed_data(self):
        pass
    
    
    def get_features(self, x):
        """
        Data preprocessing: 
        - extraction of features using VGGish feature extractor 
        - flatten the features extracted, resulting in a one-dimensional vector

        Parameters:
        x (array): signal

        Returns:
        array
        """
        vggish_features = self.features_extractor(x)
        return flatten(vggish_features)
    
    
    def get_labels_encoding(self, Y, saving_folder):
        """
        Labels encoding. 
        If a labels encoding is provided it is used, otherwhise it is created.
        The labels enconding used is saved in the results_folder

        Parameters:
        Y (list): labels
        saving_folder (str): path to the folder where the results are saved

        Returns:
        list: list of labels encoded
        """
        Y_encoded = []
        if self.yaml.get('labels_mapping_path'):
            with open(self.yaml.get('labels_mapping_path'), 'r') as file:
                data = json.load(file)
                unique_labels = list(data.values())
                encoded_labels = list(data.keys())
                for el in Y:
                    idx = unique_labels.index(el)
                    Y_encoded.append(int(encoded_labels[idx]))
            shutil.copyfile(self.yaml.get('labels_mapping_path'), os.path.join(saving_folder,"labels_mapping.json"))
        else:
            label_encoder = LabelEncoder()
            Y_encoded = label_encoder.fit_transform(Y)
            self.save_labels_encoding(label_encoder, saving_folder)
        return Y_encoded
    
    
    def get_data(self, X = None, Y = None, split_info = None, data_path = None, split = None):
        """
        Get train and test data.

        Parameters:
        X (list, optional): list of signals read. Defaults to None.
        Y (list, optional): list of labels. Defaults to None.
        split_info (list, optional): list of split info. Defaults to None.
        split (str): 'train'/'test'
        data_path (str, optional): path to dataset. Defaults to None.

        Returns:
        list of tuple, list of tuple: list of tuple of signals and the correspondig labels for train and test
        """
        # if we have the signal already read and the information about the train/test split
        if X and Y and split_info:
            data_index = [i for i, s in enumerate(split_info) if s == split]
            x_data = [X[i] for i in data_index]
            y_data = [Y[i] for i in data_index]

            if len(data_index) < 1:
                logger.error(f"Error. N° data data: {len(x_data)}.")
                exit()
        # if we need to read the audio from the data store
        else:
            x_data, y_data = [], []
            unique_data_labels = os.listdir(os.path.join(data_path, split))
            for label in unique_data_labels:
                audio_files = os.listdir(os.path.join(data_path, split, label))
                for audio in audio_files:
                    signal, sr = sf.read(audio)
                    x_data.append(signal)
                    y_data.append(label)
                    
        return [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
                    
        
    def save_labels_encoding(self, label_encoder, saving_folder):
        """
        Saving the labels encoding used
        
        Parameters:
        label_encoder: fitted label encoder.
        saving_folder (str): path to the folder where the results are saved

        Returns:
        None
        """
        label_mapping = {numeric_label: original_label for original_label, numeric_label in
                              zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        labels_mapping_path = os.path.join(saving_folder, "labels_mapping.json")
        with open(labels_mapping_path, 'w') as file:
            json.dump(label_mapping, file)
            
        
    def data_preparation(self, data, saving_folder):
        """
        Data preprocessing.
        
        Parameters:
        data (list): lis of tuple [(x1,y1),(x2,y2),...]
        saving_folder (str): path to the folder where the results are saved

        Returns:
        list, list: list of x and y ready to be fed to the classificator
        """
        X, Y = [], []
        for x,y in tqdm(data, total = len(data), desc = "Features extraction"):
            X.append(self.get_features(x))
            Y.append(y)
        Y = self.get_labels_encoding(Y, saving_folder)
        return X, Y
    
    
    def compute_scores(self, Y, Y_predicted):
        """
        Computing metrics(accuracy/macro F1 score)

        Parameters:
        Y (array): ground truth
        Y_predicted (array): labels predicted

        Returns:
        float, float: accuracy and f1 score
        """
        accuracy = accuracy_score(y_true=Y, y_pred=Y_predicted)
        f1 = f1_score(y_true=Y, y_pred=Y_predicted, average='macro')
        return accuracy, f1