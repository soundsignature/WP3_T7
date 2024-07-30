# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:27:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This file contains the class EcossDataset, which is responsible for all the data generation
    and preprocessing steps. """

import pandas as pd
import numpy as np
import librosa
import pickle
import os

class EcossDataset:
    def __init__(self, annots_path: str, pad_mode: str,
                 sr: float, duration: float, saving_on_disk: bool):
        self.annots_path = annots_path
        self.pad_mode = pad_mode
        self.sr = sr
        self.duration = duration
        self.segment_length = int(self.duration * self.sr)
        self.saving_on_disk = saving_on_disk
        # self.df = pd.read_csv(self.annots_path, sep=";")
        self.cache_path = os.path.join(os.getcwd(), "cache")
        
        
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
        white_noise = np.random.normal( loc = 0, scale = std, size=self.segment_length)
        # Pad the signal with zeros and add white noise
        segment = np.pad(signal, (delta_start, delta_end), 'constant', constant_values=(0, 0))
        segment = segment + white_noise
        
        return segment
    
    
    def save_data(self, segments, path):
        """
        Save the processed segments to disk  in the folder "././cache" as pickle files.

        Parameters:
        segments (list): List of processed segments.
        path (str): Path to save the segments.
        """
        # Create the cache directory if it does not exist
        os.makedirs(self.cache_path, exist_ok=True)
        # Extract the base filename from the path
        filename = path.split('.')[0].replace('/', '_')[1:]
        
        # Save each segment as a separate pickle file
        for idx, segment in enumerate(segments):
            saving_filename = filename + '-' + str(idx) + '.pickle'
            with open(os.path.join(self.cache_path, saving_filename), 'wb') as f:
                pickle.dump(segment, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ecoss_data = EcossDataset(...)
    signals = ...
    sr = ...
    paths = ...
    labels = ...
    signals_processed, labels_processed = ecoss_data.process_all_data(signals_list=signals, original_sr_list=sr, paths_list=paths, labels_list=labels)