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

from utils import AugmentMelSTFT

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class EffAtModel():
    def __init__(self, yaml_content: dict, data_path: str) -> None:
        self.yaml = yaml_content
        self.data_path = data_path
        self.mel = AugmentMelSTFT(freqm=10, timem=10)
        

    def train(self, results_folder: str) -> None:
        # Saving the configuration.yaml inside the results folder
        self.results_folder = Path(results_folder)
        logging.info(f"Training EffAT")
        output_config_path = self.results_folder / 'configuration.yaml'
        logging.info(f"Saving configuration in {output_config_path}")
        with open(str(output_config_path), 'w') as outfile:
            yaml.dump(self.yaml, outfile, default_flow_style=False)
        logging.info(f"Config params:\n {self.yaml}")





    def test(self,results_folder):
        pass


    def inference(self,results_folder):
        pass


    def plot_results(self):
        pass


    def save_weights(self):
        pass


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


if __name__ == "__main__":
    model = EffAtModel({'a': 1}, r"C:\Users\Jose Antonio\Desktop\ScriptsProyectos\cinea\wp3-t7\scripts\data")
    model.plot_processed_data(augment=False)   

            



