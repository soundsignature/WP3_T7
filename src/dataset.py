# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:27:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This file contains the class EcossDataset, which is responsible for all the data generation
    and preprocessing steps. """

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import librosa
import pickle
import os

UNWANTED_LABELS = ["Undefined", "Anthropogenic"]

class EcossDataset:
    def __init__(self, annots_path: str, path_store_data: str, pad_mode: str,
                 sr: float, duration: float, saving_on_disk: bool):
        self.annots_path = annots_path
        self.path_store_data = path_store_data
        self.pad_mode = pad_mode
        self.sr = sr
        self.duration = duration
        self.segment_length = int(self.duration * self.sr)
        self.saving_on_disk = saving_on_disk
        self.df = pd.read_csv(self.annots_path, sep=";")
        
    
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
        
    def process_all_data(self, signals_list, original_sr_list, paths_list, labels_list):
        """
        Process the signals and return processed signals and labels, according to the sample rate, duration and pad_mode chosen.

        Parameters:
        signals_list (list): List of signal arrays.
        original_sr_list (list): List of original sampling rates.
        paths_list (list): List of paths corresponding to each signal.
        labels_list (list): List of labels corresponding to each signal.

        Returns:
        tuple: list of processed signals and  list of the corresponding labels.
        """
        processed_signals = []
        processed_labels = []
        # Iterate over all signals,sr,paths,labels
        for signal, original_sr, path, label in zip(signals_list, original_sr_list, paths_list, labels_list):
            # Process the signal
            segments = self.process_data(signal, original_sr)
            if self.saving_on_disk:
                try:
                    # Save the processed segments to disk
                    self.save_data(segments, path)
                except:
                    print('Error : data could not be saved')
                    
            # Extend the lists of processed signals and labels 
            processed_signals.extend(segments)
            processed_labels.extend(label * len(segments))
        
        # Ensure the lengths of signals and labels match  
        assert len(processed_signals)==len(processed_labels),'Error : signals and labels processed have different length'
        return processed_signals, processed_labels
                        
            
    def process_data(self, signal, original_sr):
        """
        Process a single signal by resampling and segmenting or padding it.

        Parameters:
        signal (np.array): Signal array.
        original_sr (float): Original sampling rate of the signal.

        Returns:
        list: List of processed segments.
        """
        # Resample the signal if the original sampling rate is different from the target
        if original_sr != self.sr:
            signal = librosa.resample(y=signal, orig_sr=original_sr, target_sr=self.sr)
         
        # Pad the signal if it is shorter than the segment length   
        if len(signal) < self.segment_length:
            segments = self.make_padding(signal)
         # Segment the signal if it is longer or equal to the segment length
        elif len(signal) >= self.segment_length:
            segments = self.make_segments(signal)
            
        return segments
    
    
    def make_segments(self, signal):
        """
        Segment a signal into equal parts based on segment length (duration).

        Parameters:
        signal (np.array): Signal array.

        Returns:
        list: List of signal segments.
        """
        segments = []
        # Calculate the number of full segments in the signal
        n_segments = len(signal)//(self.segment_length)
        # Extract each segment and append to the list
        for i in range(n_segments):
            segment = signal[(i*self.segment_length):(i+1*self.segment_length)]
            segments.append(segment)
        return segments
            
            
    def make_padding(self, signal):
        """
        Pad a signal to match a fixed segment length using the specified pad mode.

        Parameters:
        signal (np.array): Signal array.
        Returns:
        list: List containing the padded signal.
        """
        # Calculate the amount of padding needed
        delta = self.segment_length - len(signal)
        delta_start = delta // 2
        delta_end = delta_start if delta%2 == 0 else (delta // 2) + 1 
        
        # Pad the signal according to the specified mode
        if self.pad_mode == 'zeros':
           segment = self.zero_padding(signal, delta_start, delta_end)
        elif self.pad_mode == 'white_noise':
            segment = self.white_noise_padding(signal, delta_start, delta_end)
        else:
            print('Error : pad_mode not valid')
            exit(1)
            
        return [segment]
    

    def zero_padding(self, signal, delta_start, delta_end):
        """
        Pad the signal with zeros.

        Parameters:
        signal (np.array): Signal array.
        delta_start (int): Number of zeros to add at the start.
        delta_end (int): Number of zeros to add at the end.

        Returns:
        np.array: Zero-padded signal.
        """
        segment = np.pad(signal, (delta_start, delta_end), 'constant', constant_values=(0, 0))
        
        return segment
    
    
    def white_noise_padding(self, signal, delta_start, delta_end):
        """
        Pad the signal with white noise.

        Parameters:
        signal (np.array): Signal array.
        delta_start (int): Number of padding values to add at the start.
        delta_end (int): Number of padding values to add at the end.

        Returns:
        np.array: White-noise padded signal.
        """
        # TODO we should decide is std needs to be a parameter to set or not 
        # Generate white noise with standard deviation scaled to the signal
        std = np.std(signal)/10
        white_noise_start = np.random.normal(loc=0, scale=std, size=delta_start)
        white_noise_end = np.random.normal(loc=0, scale=std, size=delta_end)

        # Concatenate white noise segments with the original signal
        segment = np.concatenate((white_noise_start, signal, white_noise_end))

        return segment
    
    
    def save_data(self, segments, path):
        """
        Save the processed segments to disk  in the folder "././cache" as pickle files.

        Parameters:
        segments (list): List of processed segments.
        path (str): Path to save the segments.
        """
        # Create the cache directory if it does not exist
        os.makedirs(self.path_store_data, exist_ok=True)
        # Extract the base filename from the path
        filename = path.split('.')[0].replace('/', '_')[1:]
        
        # Save each segment as a separate pickle file
        for idx, segment in enumerate(segments):
            saving_filename = filename + '-' + str(idx) + '.pickle'
            with open(os.path.join(self.path_store_data, saving_filename), 'wb') as f:
                pickle.dump(segment, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    
    load_dotenv()
    ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
    # LABELS = 
    ecoss_data = EcossDataset(ANNOTATIONS_PATH, '.', '.', '.', '.')
    ecoss_data.fix_onthology(labels=["Odontocetes", "Ship"])
    times = ecoss_data.generate_insights()
    signals, sr, paths, labels = ...
    signals_processed, labels_processed = ecoss_data.process_all_data(signals_list=signals, original_sr_list=sr, paths_list=paths, labels_list=labels)
    
