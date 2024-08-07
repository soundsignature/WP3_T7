"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class VggishModel which is responsible for all the training, testing and inference stuff related with the
    Vggish + SVM model """
    
from utils import *
from tools.confusion_matrix import *
from tools.training_curves import *
import json
import pandas as pd
import os
from datetime import datetime
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from models.vggish_modules import VggishFeaturesExtractor


class VggishModel():
    def __init__(self, yaml_content: dict, signals: list, labels:list, split_info:list, data_path: str = None) -> None:
        self.yaml = load_yaml(yaml_content) 
        self.results_folder = './runs/vggish/'
        self.features_extractor = VggishFeaturesExtractor()
        self.model  = SVC()
        self.train_data, self.test_data = self.get_train_test_data(X = signals, Y = labels, split = split_info)


    def train(self):
        current_time = now.strftime("%H:%M:%S:%f")
        now = datetime.now()
        saving_folder = os.path.join(self.results_folder, "train", current_time)
        os.makedirs(saving_folder, exist_ok=True)
        
        X, Y = self.data_preparation(data = self.train_data, saving_folder = saving_folder)
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
        self.save_weigths(saving_folder = saving_folder)
        self.plot_results(set = 'train', saving_folder = saving_folder, gridsearch = grid)


    def test(self):
        current_time = now.strftime("%H:%M:%S:%f")
        now = datetime.now()
        saving_folder = os.path.join(self.results_folder, "test", current_time)
        os.makedirs(saving_folder, exist_ok=True)
        
        X, Y = self.data_preparation(data = self.test_data)
        if self.yaml.get('model_path'):
            self.model = joblib.load(self.yaml.get('model_path'))
        Y_pred = self.model.predict(X)
        
        accuracy, f1 = self.compute_scores(Y, Y_pred)
        result = {'accuracy':accuracy, 'f1_score' : f1}
        with open(os.path.join(saving_folder, 'metrics.json'), 'w') as json_file:
            json.dump(result, json_file)
        self.plot_results(set = 'test', saving_folder = saving_folder, y_true = Y, y_pred = Y_pred)
        

    def inference(self,x):
        if self.yaml.get('model_path'):
            self.model = joblib.load(self.yaml.get('model_path'))
        else:
            print('Error. model_path missing in the yaml configuration file')
            exit()
        y = self.model.predict(x)
        return y


    def plot_results(self, set, saving_folder, gridsearch = None, y_true = None, y_pred = None):
        if set == 'train':
            cf_matrix = ConfusionMatrix(labels_mapping_path = os.path.join(saving_folder, 'labels_mapping.json'))
            fig1 = cf_matrix.plot(y = y_true, y_pred = y_pred)
            cf_matrix.save_plot(plot = fig1, saving_folder = saving_folder)
            
            training_curves = ValidationPlot(gridsearch = gridsearch)
            fig2 = training_curves.plot()
            training_curves.save_plot(plot = fig2, saving_folder = saving_folder)
        else:
            cf_matrix = ConfusionMatrix(labels_mapping_path = os.path.join(saving_folder, 'labels_mapping.json'))
            fig1 = cf_matrix.plot(y = y_true, y_pred = y_pred)
            cf_matrix.save_plot(plot = fig1, saving_folder = saving_folder)


    def save_weigths(self,saving_path):
        with open(saving_path+'model.joblib') as f:
            joblib.dump(self.model)

    def plot_processed_data(self):
        pass
    
    
    def get_features(self, x):
        vggish_features = self.features_extractor(x)
        return flatten(vggish_features)
    
    
    def get_labels_encoding(self, Y):
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        self.save_labels_encoding(label_encoder)
        
        
    def get_train_test_data(self, X, Y, split):
        train_index = [i for i, s in enumerate(split) if s=='train']
        x_train = [X[i] for i in train_index]
        y_train = [[Y[i] for i in train_index]]
        
        test_index = [i for i, s in enumerate(self.split_info) if s=='test']
        x_test = [X[i] for i in test_index]
        y_test = [[Y[i] for i in test_index]]
        
        if  len(train_index)<1 and len(test_index)<1:
            print("Error. Missing data")
        
        return [(x_train[i], y_train[i]) for i in range(0, len(x_train))], [(x_test[i], y_test[i]) for i in range(0, len(x_test))]
       
        
    def save_labels_encoding(self, label_encoder, saving_folder):
        label_mapping = {numeric_label: original_label for original_label, numeric_label in
                              zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        labels_mapping_path = os.path.join(saving_folder, "labels_mapping.json")
        with open(labels_mapping_path, 'w') as file:
            json.dump(label_mapping, file)
            
        
    def data_preparation(self, data):
        X, Y = [], []
        for x,y in data:
            X.append(self.get_features(x))
            Y.append(y)
        Y = self.get_labels_encoding(Y)
        return X, Y
    
    
    def compute_scores(self, Y, Y_predicted):
        accuracy = accuracy_score(y_true=Y, y_pred=Y_predicted)
        f1 = f1_score(y_true=Y, y_pred=Y_predicted, average='macro')
        return accuracy, f1