# -*- coding: utf-8 -*-


"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class PasstModel which is responsible for all the training, testing and inference stuff related with the
    PaSST model """

from pathlib import Path
from copy import deepcopy
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

from hear21passt.base import get_model_passt
from hear21passt.models.preprocess import AugmentMelSTFT
from hear21passt.wrapper import PasstBasicWrapper
from torch.utils.data import Dataset
import torchaudio
from pipeline.utils import save_training_curves, save_confusion_matrix

class StandardDataset(Dataset):
    def __init__(self, root_dir, sr=32000, duration=9.98, label_to_idx = None,time_threshold = 0.6):
        self.root_dir = Path(root_dir)
        self.sr = sr
        self.duration = duration
        self.labels = [item.name for item in self.root_dir.glob('*') if item.is_dir()]
        self.time_threshold = time_threshold
        if not label_to_idx:
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        else:
            self.label_to_idx = label_to_idx
        self.data = []
        for label in self.labels:
            for audio_file in self.root_dir.joinpath(label).rglob('*.wav'):
                self.data.append((audio_file, self.label_to_idx[label]))
        
        self.resamplers = [(None,None)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath, target = self.data[idx]
        # Load the audio file
        waveform, sr = torchaudio.load(str(filepath))
        return waveform, target, str(filepath)


class PasstModel():
    def __init__(self, yaml_content: dict, data_path: str) -> None:
        self.yaml = yaml_content
        self.data_path = data_path
        self.data_path = Path(self.data_path)
        

    
    def train(self,results_folder):
        # Example 
        self.results_folder = Path(results_folder)
        logging.info(f"Training PASST")
        output_config_path = self.results_folder / 'configuration.yaml'
        logging.info(f"Saving configuration in {output_config_path}")
        with open(str(output_config_path), 'w') as outfile:
            yaml.dump(self.yaml, outfile, default_flow_style=False)
        logging.info(f"Config params:\n {self.yaml}")
        train_dataloader, test_dataloader = self.load_train_test_datasets()
        
    
    
    def load_train_test_datasets(self):
        if (self.data_path / "train").exists() and (self.data_path / "test").exists():
            dataset = StandardDataset(self.data_path / "train",duration = self.yaml["duration"],sr=self.yaml["sr"])
            val_dataset = StandardDataset(self.data_path / "test",label_to_idx = dataset.label_to_idx,duration=self.yaml["duration"],sr=self.yaml["sr"])
            
        unique_labels = dataset.labels
        n_classes = len(unique_labels)
        

        # Obtener las etiquetas de todo el conjunto de datos
        targets = [target for _, target, _ in dataset]

        # Crear muestras ponderadas para equilibrar las clases
        class_count = [i for i in np.bincount(targets)]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

        train_targets = targets
        train_weights = [class_weights[t] for t in train_targets]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        
        # Obtener las etiquetas de todo el conjunto de datos
        targets = [target for _, target, _ in val_dataset]
        # Crear muestras ponderadas para equilibrar las clases
        class_count = [i for i in np.bincount(targets)]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
        test_targets = targets
        test_weights = [class_weights[t] for t in test_targets]
        test_sampler = WeightedRandomSampler(test_weights, len(train_weights))

    
        train_loader = DataLoader(dataset, batch_size=self.yaml.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.yaml.batch_size)
        
        self.unique_labels = unique_labels
        self.n_classes = n_classes
            
        return train_loader, val_loader

    def test(self,results_folder):
        pass


    def inference(self,results_folder):
        pass


    def plot_results(self):
        pass


    def save_weights(self):
        pass


    def plot_processed_data(self):
        pass
    
