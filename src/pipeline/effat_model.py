# -*- coding: utf-8 -*-
"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class EffAtModel which is responsible for all the training, testing and inference stuff related with the
    EfficientAT model """

import yaml
import logging
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json

from .utils import AugmentMelSTFT, load_yaml, load_data, EffATWrapper, data_loader
from .effat_repo.models.mn.model import get_model as get_mn
from .effat_repo.models.dymn.model import get_model as get_dymn

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class EffAtModel():
    def __init__(self, yaml_content: dict, data_path: str, name_model: str, num_classes: int) -> None:
        self.yaml = yaml_content
        self.data_path = data_path
        self.mel = AugmentMelSTFT(freqm=self.yaml["freqm"],
                                  timem=self.yaml["freqm"])
        self.name_model = name_model
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if "dy" not in self.name_model:
            model = get_mn(pretrained_name=self.name_model)
        else:
            model = get_dymn(pretrained_name=self.name_model)
        
        # Using the wrapper to modify the last layer and moving to device
        model = EffATWrapper(num_classes=num_classes, model=model, freeze=self.yaml["freeze"])
        model.to(self.device)
        self.model = model


    def train(self, results_folder: str) -> None:
        # Saving the configuration.yaml inside the results folder
        self.results_folder = Path(results_folder)
        logging.info(f"Training EffAT")
        output_config_path = self.results_folder / 'configuration.yaml'
        logging.info(f"Saving configuration in {output_config_path}")
        with open(str(output_config_path), 'w') as outfile:
            yaml.dump(self.yaml, outfile, default_flow_style=False)
        logging.info(f"Config params:\n {self.yaml}")

        # Load the data on suitable way
        train_data, train_label_encoder = load_data(os.path.join(self.data_path, 'train'))
        test_data, _ = load_data(os.path.join(self.data_path, 'test'))
        
        logging.info(f"The encoder is: {train_label_encoder}")

        # # Generate the dataloaders
        # train_dataloader = data_loader(train_data, self.mel, False, self.yaml["batch_size"])
        # test_dataloader = data_loader(test_data, self.mel, True, self.yaml["batch_size"], shuffle=False)
        
        # Begin the training
        self.model.train()
        if self.yaml["optimizer"].lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.yaml["lr"])
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.yaml["lr"])
        
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        epochs_without_improvement = 0

        train_accs, test_accs = [], []
        train_losses, test_losses = [], []

        for i in tqdm(range(self.yaml["n_epochs"]), desc="Epoch"):
            self.model.train()
            running_loss = 0.0
            batch_count = 0
            correct = 0
            total = 0
            
            # Generating dataloaders every epochs because they are generators (can only be iterated once)
            train_dataloader = data_loader(train_data, self.mel, False, self.yaml["batch_size"])
            test_dataloader = data_loader(test_data, self.mel, True, self.yaml["batch_size"], shuffle=False)

            for inputs, labels in tqdm(train_dataloader, desc="Train"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.model(inputs)
                outputs = outputs.squeeze()

                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1

                 # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / batch_count
            train_accuracy = 100 * correct / total
            
            # Evaluation
            self.model.eval()
            test_loss = 0.0
            batch_count = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(test_dataloader, desc="Test"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    outputs = outputs.squeeze()

                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    batch_count += 1

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    
            avg_test_loss = test_loss / batch_count
            test_accuracy = 100 * correct / total
            
            train_losses.append(epoch_loss)
            test_losses.append(avg_test_loss)
            train_accs.append(train_accuracy)
            test_accs.append(test_accuracy)

            logging.info(f"Epoch {i}: Train loss -> {epoch_loss}, test loss -> {avg_test_loss}, train accuracy -> {train_accuracy}, test accuracy -> {test_accuracy}")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                epochs_without_improvement = 0  # Reset counter if we see improvement
                logging.info(f"New best testing accuracy: {best_accuracy}")

                self.plot_results(train_losses, test_losses, train_accs, test_accs)
                self.save_weights(optimizer)
                self.save_results(train_label_encoder, )

            else:
                epochs_without_improvement += 1
                logging.info(f"No improvement for {epochs_without_improvement} epoch(s).")
        
            if epochs_without_improvement >= self.yaml["patience"]:
                logging.info(f"Early stopping triggered after {i+1} epochs.")
                
                break

            
    def test(self,results_folder):
        pass


    def inference(self,results_folder):
        pass


    def plot_results(self, train_loss, test_loss, train_acc, test_acc):
        """This function is used to load the 

        Args:
            train_loss (_type_): _description_
            test_loss (_type_): _description_
            train_acc (_type_): _description_
            test_acc (_type_): _description_
        """
        plt.figure()
        plt.plot(train_loss, label="Train losses")
        plt.plot(test_loss, label="Test losses")
        plt.legend()
        plt.savefig(self.results_folder / 'losses.png')
        plt.close()
        
        plt.figure()
        plt.plot(train_acc, label="Train accuracy")
        plt.plot(test_acc, label="Test accuracy")
        plt.legend()
        plt.savefig(self.results_folder / 'accuracies.png')
        plt.close()


    def save_weights(self, optimizer):
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, self.results_folder / 'model.pth')
        
    
    def save_results(self, label_encoder):
        # Save the class dictionary
        with open(self.results_folder / 'class_dict.json', 'w') as json_file:
            json.dump(label_encoder, json_file)

        # Save the results





    def plot_processed_data(self, augment: bool = True):
        """This function will plot a random mel spectrogram per class available for the training
        """
        path_classes = os.path.join(self.data_path, "train")
        available_classes = os.listdir(path_classes)

        if augment == False:
            self.mel.eval()

        for av_class in available_classes:
            path_wavs = os.path.join(path_classes, av_class)
            wav_to_plot = os.path.join(path_wavs,
                                       np.random.choice(os.listdir(path_wavs)))
            logger.info(f"The file that will be plotted is {wav_to_plot}")

            y, sr = torchaudio.load(wav_to_plot)
            melspec = self.mel(y)
            logger.info(f"The shape of the melspec is {melspec.shape}")

            plt.figure()
            plt.imshow(melspec[0], origin="lower")
            plt.title(av_class)
            plt.show()




# if __name__ == "__main__":
#     load_dotenv()
#     DATASETS_PATH = os.getenv("DATASETS_PATH")
#     YAML_PATH = os.getenv("YAML_PATH")
#     NAME_MODEL = os.getenv("NAME_MODEL")

#     model = EffAtModel(load_yaml(YAML_PATH), DATASETS_PATH, NAME_MODEL, 10)
#     model.plot_processed_data(augment=True)   

            



