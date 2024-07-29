# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:27:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This file contains the class EcossDataset, which is responsible for all the data generation
    and preprocessing steps. """
    
class EcossDataset:
    def __init__(self, annots_path: str, path_store_data: str, pad_mode: str,
                 sr: float, duration: float, labels: list):
        self.annots_path = annots_path
        self.path_store_data = path_store_data
        self.pad_mode = pad_mode
        self.sr = sr
        self.duration = duration
        self.labels = labels
    
    
    def filter_overlapping(self):
        pass
    
    
    def process_audios(self):
        pass
    
    
    def remap_onthology(self):
        pass
    
    
    def generate_insights(self):
        pass
    
    
    