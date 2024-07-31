# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:27:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This file contains the class EcossDataset, which is responsible for all the data generation
    and preprocessing steps. """

import pandas as pd
import os
from matplotlib import pyplot as plt
from dotenv import load_dotenv

UNWANTED_LABELS = ["Undefined", "Anthropogenic"]

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
        This function drops the rows that contain an unwanted label

        Parameters
        ----------
        unwanted_labels : list
            This list containing the unwanted labels.

        Returns
        -------
        None.

        """
        idxs_to_drop = []
        for i, row in self.df.iterrows():
            for label in unwanted_labels:
                if label in row["final_source"]:
                    idxs_to_drop.append(i)
        
        self.df = self.df.drop(idxs_to_drop)
        # self.df.reset_index(drop=True, inplace=True) Comment because its needed for the overlapping
        
    
    def fix_onthology(self, labels: list[str] = None):
        """
        This function generates a new column with the final labels on
        the annotations.csv file. If a list of labels is used, the function
        not only formats the labels to use the last part, but also fixes the
        onthology to the level of detail requested.
        
        Parameters
        ----------
        labels : list[str], optional
            A list with the labels to be modified.. The default is None.

        Returns
        -------
        None.

        """
        # Dropping rows that contain nan in label_source
        self.df = self.df.dropna(subset=['label_source'])
        self.df.reset_index(drop=True, inplace=True)
        
        if labels != None:
            for i, row in self.df.iterrows():
                for label in labels:
                    idx = row["label_source"].find(label)
                    if idx != -1:
                        delimiter = idx + len(label)
                        self.df.loc[i, "label_source"] = self.df.loc[i, "label_source"][:delimiter]
        
        # Once they are defined, we create the final column with the labels
        # Currently not saving, only overwritting the df parameter as this is the first step
        self.df["final_source"] = self.df["label_source"].apply(lambda x: x.split('|')[-1])
        
        # Now, we can proceed to eliminate the unwanted labels
        self.drop_unwanted_labels(UNWANTED_LABELS)
        
        
    # TODO: When the splitting is performed, go for metrics per split
    def generate_insights(self):
        """
        This function is used to generate insights on the data. It generates plots
        for the number of sound signatures per class, and the time per class.
        
        IMPORTANT: It needs to be used right after the first step (the remapping and reformating of classes).
        However, you dont need to remap the labels if you don't want to, as this parameter
        is optional.

        Returns
        -------
        None.

        """
        # Plot for number of sound signatures (time independent)
        count_signatures = self.df["final_source"].value_counts()
        plt.figure(figsize=(8,6))
        plt.bar(range(0, len(count_signatures)), count_signatures)
        plt.xticks(range(0, len(count_signatures)),
                   count_signatures.index.to_list(),
                   horizontalalignment='center')
        plt.xlabel("Source")
        plt.ylabel("# of sound signatures")
        plt.show()
        
        # Plot for time per class of sound signature
        times = dict()
        for i, row in self.df.iterrows():
            if row["final_source"] not in times.keys():
                times[row["final_source"]] = row["tmax"] - row["tmin"]
            else:
                times[row["final_source"]] += row["tmax"] - row["tmin"]
        plt.figure(figsize=(8,6))
        plt.bar(range(0, len(times)), times.values())
        plt.xticks(range(0, len(times)),
                   list(times.keys()),
                   horizontalalignment='center')
        plt.xlabel("Source")
        plt.ylabel("Time (s)")
        plt.show()
        

    
if __name__ == "__main__":
    load_dotenv()
    ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
    # LABELS = 
    ecoss_data = EcossDataset(ANNOTATIONS_PATH, '.', '.', '.', '.')
    ecoss_data.fix_onthology(labels=["Odontocetes", "Ship"])
    times = ecoss_data.generate_insights()
    
    
    
    

