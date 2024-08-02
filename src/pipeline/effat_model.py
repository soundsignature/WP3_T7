# -*- coding: utf-8 -*-
"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class EffAtModel which is responsible for all the training, testing and inference stuff related with the
    EfficientAT model """

class EffAtModel():
    def __init__(self, yaml_content: dict, data_path: str) -> None:
        self.yaml = yaml_content
        self.data_path = data_path

        

    def train(self,results_folder):
        pass


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


