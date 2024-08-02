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

class EffAtModel():
    def __init__(self, yaml_content: dict, data_path: str) -> None:
        self.yaml = yaml_content
        self.data_path = data_path

        

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


    def plot_processed_data(self):
        pass


