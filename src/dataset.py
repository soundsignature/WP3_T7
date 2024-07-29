# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:27:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This file contains the class EcossDataset, which is responsible for all the data generation
    and preprocessing steps. """

import pandas as pd
import os
from dotenv import load_dotenv

    
class EcossDataset:
    def __init__(self, annots_path: str, path_store_data: str, pad_mode: str,
                 sr: float, duration: float):
        self.annots_path = annots_path
        self.path_store_data = path_store_data
        self.pad_mode = pad_mode
        self.sr = sr
        self.duration = duration
        self.df = pd.read_csv(self.annots_path, sep=";")
    
    
    def filter_overlapping(self):
        pass
    
    
    def process_audios(self):
        pass
    
    
    def drop_unwanted_labels(self, unwanted_labels: list):
        """
        This function drops the rows that contain an unwanted label.
        It also drops nan rows (e.g the ones in wavec dataset)

        Parameters
        ----------
        unwanted_labels : list
            This list containing the unwanted labels.

        Returns
        -------
        None.

        """
        df = self.df.dropna(subset=['label_source'])
        df.reset_index(drop=True, inplace=True)
        df = df
        
        idxs_to_drop = []
        for i, row in df.iterrows():
            for label in unwanted_labels:
                if label in row["label_source"]:
                    idxs_to_drop.append(i)
        
        df = df.drop(idxs_to_drop)
        df.reset_index(drop=True, inplace=True)
        self.df = df
        
    
    def remap_onthology(self, labels: list[str]):
        """
        This function generates a new column with the final labels on
        the annotations.csv file.

        Parameters
        ----------
        labels : list[str]
            A list with the labels to be modified.

        Returns
        -------
        None.

        """
        self.drop_unwanted_labels(['Undefined'])
        
        for i, row in self.df.iterrows():
            for label in labels:
                idx = row["label_source"].find(label)
                if idx != -1:
                    delimiter = idx + len(label)
                    self.df.loc[i, "label_source"] = self.df.loc[i, "label_source"][:delimiter]
        
        # Once they are defined, we create the final column with the labels
        # Currently not saving, only overwritting the df parameter as this is the first step
        self.df["final_source"] = self.df["label_source"].apply(lambda x: x.split('|')[-1])
        
                    
    def generate_insights(self):
        pass
    

