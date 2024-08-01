# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:27:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This file contains the class EcossDataset, which is responsible for all the data generation
    and preprocessing steps. """

import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import numpy as np
from enum import Enum
import librosa
import pickle
import os

UNWANTED_LABELS = ["Undefined"]

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
    
    @staticmethod
    def concatenate_ecossdataset(dataset_list):
        """
        Checks the EcossDataset object provided in a list have the same variables and then generates a new object with
        a concatenated dataframe. annots_path and path_store_data taken from the first EcossDataset in the list.

        Inputs
        -------
        dataset_list : List with EcossDataset to be concatenated

        Outputs
        -------
        EcossDataset with concatenated DataFrame
        """

        #Extract values to compare
        annot_path0 = dataset_list[0].annots_path
        sr0 = dataset_list[0].sr
        duration0 = dataset_list[0].duration
        padding0 = dataset_list[0].pad_mode
        save0 = dataset_list[0].saving_on_disk
        path_store0 = dataset_list[0].path_store_data
        #Start populatinf DataFrame list
        df_list = [dataset_list[0].df]
        #Iterate over list to check appropiate values, exiting function it variables do not match
        for dataset in dataset_list[1:]:
            if dataset.sr != sr0 or dataset.duration != duration0 or dataset.pad_mode != padding0 or dataset.saving_on_disk != save0:
                print("The datasets selected do not have the same characteristics")
                return
            else:
                df_list.append(dataset.df)
        #Create EcossDataset object with concatenated info
        ConcatenatedEcoss = EcossDataset(annots_path=annot_path0, path_store_data=path_store0,
                                         pad_mode=padding0, sr=sr0, duration=duration0, saving_on_disk=save0)
        ConcatenatedEcoss.df = pd.concat(df_list)
        return ConcatenatedEcoss

    def split_train_test_balanced(self, test_size=0.2, random_state=None):
        """
        Divides the dataframe in train and test at file level, ensuring a balanced class distribution.
        Adds a 'split' column with values 'test' or 'train' 

        Parameters:
       
        test_size (float): Ratio of test dataset.
        random_state (int): Random seed.
 
        Returns:
        None (updates pd.DataFrame: original DataFrame with an extra columnn named 'split')
        """
        # Creates a DataFrame with unique files and their labels
        file_labels = self.df.groupby('parent_file')['final_source'].apply(
            lambda x: x.mode()[0]).reset_index()

        # Initialize column split in the original DataFrame
        self.df['split'] = ''

        # Handle classes with only one instance
        single_instance_files = file_labels[file_labels.duplicated('final_source', keep=False) == False]
        multiple_instance_files = file_labels[file_labels.duplicated('final_source', keep=False) == True]

        # Create StratifiedShuffleSplit for files with multiple instances
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        # Get indexes for train and test
        for train_idx, test_idx in sss.split(multiple_instance_files['parent_file'], multiple_instance_files['final_source']):
            train_files = multiple_instance_files['parent_file'].iloc[train_idx]
            test_files = multiple_instance_files['parent_file'].iloc[test_idx]
        # Add single instance files to the train set
        train_files = pd.concat([train_files, single_instance_files['parent_file']])

        # Assign in the original DataFrame 'train' or 'test' in columns 'split'
        self.df.loc[self.df['parent_file'].isin(train_files), 'split'] = 'train'
        self.df.loc[self.df['parent_file'].isin(test_files), 'split'] = 'test'
        
    def filter_overlapping(self, visualize_overlap = False):
        """
        Filters overlapping segments in the dataset and optionally generates a representation of the timeline of labels before and after filtering.

        Parameters:
        visualize_overlap (bool): If True, visualizes the timeline of labels before and after processing.

        Returns:
        None
        """
        overlap_info_processed = self._extract_overlapping_info()
        self.df["overlap_info_processed"] = overlap_info_processed
        # self.df.dropna(subset=["final_source"],inplace=True)
        self.df["to_delete"] = False
        if visualize_overlap:
            self._visualize_overlappping(self.df)
        # Iterate through the DataFrame to handle overlapping segments
        for eval_idx,_ in self.df.iterrows():
            not_count = False
            if not eval_idx in self.df.index:
                continue
            if np.isnan(self.df.loc[eval_idx]["overlapping"]):
                continue
            segments_to_delete = []
            for overlap_idx,tmin,tmax in self.df.loc[eval_idx]["overlap_info_processed"]:
                if overlap_idx not in self.df.index:
                    continue
                if self.df.loc[eval_idx]["final_source"] != self.df.loc[overlap_idx]["final_source"]:
                    # Add to segments_to_delete everytime there is overlapping different class sources
                    segments_to_delete.append([tmin,tmax])
                else:
                    # Handle when the two overlapping segments are from the same class
                    t_eval = [self.df.loc[eval_idx]['tmin'],self.df.loc[eval_idx]["tmax"]]
                    t_overlap = [self.df.loc[overlap_idx]['tmin'],self.df.loc[overlap_idx]["tmax"]]
                    
                    superpos = self._check_superposition(t_eval,t_overlap)
                    not_count = self._handle_superposition(eval_idx, overlap_idx, superpos)
                    if not_count:
                        break
            if not_count:
                continue
            # Divide the event into subevents excluding the segments_to_delete
            final_segments = self._divide_labels([self.df.loc[eval_idx]["tmin"],self.df.loc[eval_idx]["tmax"]],segments_to_delete)
            for tmin,tmax in final_segments:
                new_row = self.df.loc[eval_idx].copy()
                new_row["tmin"] = tmin
                new_row["tmax"] = tmax
                new_df = pd.DataFrame([new_row])
                new_df.index = [np.max(self.df.index)+1]
                self.df = pd.concat([self.df,new_df], axis=0)
            
            self.df.at[eval_idx,'to_delete'] = True
            
        # Remove rows marked for deletion
        self.df.drop(self.df[self.df["to_delete"]==True].index,inplace=True)
        self.df.drop(columns=['to_delete'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        if visualize_overlap:
            self._visualize_overlappping(self.df,"_postprocessed") 
 
        
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
        # self.df.reset_index(drop=True, inplace=True) Comment because its needed for the overlapping
        
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
        
    def _extract_overlapping_info(self):
        """
        Extracts and processes overlapping information from the DataFrame.

        Returns:
        list: A list of processed overlapping information for each row in the DataFrame.
        """
        overlap_info_processed = []
        # Process each row to handle overlapping information
        for eval_idx,row in self.df.iterrows():
            if pd.isna(row["overlapping"]):
                overlap_info_processed.append([])
                continue
            overlap_info_processed.append(self._parse_overlapping_field(row["overlap_info"]))
        return overlap_info_processed   
    
    def _visualize_overlappping(self,df,append = ""):
        """
        Visualizes overlapping segments for each unique parent file in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the segments to visualize.
        append (str): A suffix to append to the visualization filename.

        Returns:
        None
        """
        # Get unique parent files
        files = df["parent_file"].unique()
        
        for file in files:
            segments = []
            labels = []
            
            # Iterate through rows corresponding to the current parent file
            for eval_idx, row in df[df["parent_file"] == file].iterrows():
                segments.append([row['tmin'], row["tmax"]])
                labels.append(row['final_source'])
            
            # Plot segments for the current file
            self._plot_segments(segments, labels, file, append=append)
    
    @staticmethod
    def _plot_segments(segments, labels, filename, append=""):
        """
        Plots the segments with their corresponding labels and saves the plot as a PNG file.

        Parameters:
        segments (list of lists): A list of [t_min, t_max] pairs representing the segments.
        labels (list): A list of labels corresponding to each segment.
        filename (str): The base filename for the saved plot.
        append (str): A suffix to append to the filename.

        Returns:
        None
        """
        try:
            fig, ax = plt.subplots()
            labels_unique = list(np.unique(labels))
            # Plot each segment
            for i, (t_min, t_max) in enumerate(segments):
                ax.plot([t_min, t_max], [labels_unique.index(labels[i]), labels_unique.index(labels[i])], marker='o')
            ax.set_yticks(range(len(labels_unique)))
            ax.set_yticklabels([f'{x}' for x in labels_unique])
            ax.set_xlabel('Time')
            ax.set_title(f'{filename}')
            ax.set_xlim([0, np.max(segments) + 10])
            # Save the plot
            plt.savefig(filename + append + ".png")
        except Exception as e:
            print(f"An error occurred while plotting overlapping segments: {e}")
        finally:
            plt.close(fig) 
    @staticmethod
    def _parse_overlapping_field(overlapping_str):
        """
        Parses a string containing overlapping segment information and returns a list of tuples.

        Parameters:
        overlapping_str (str): A string containing overlapping segment information.

        Returns:
        list of tuples: A list where each tuple contains (index, start, stop) for each overlapping segment.
        """
        # Split the string by commas and remove the last empty element
        overlapping_str = overlapping_str.split(",")[:-1]
        
        # Group the elements into sublists of three elements each
        divided_list = [overlapping_str[i:i + 3] for i in range(0, len(overlapping_str), 3)]
        
        overlap_info = []
        
        # Parse each sublist to extract index, start, and stop values
        for x in divided_list:
            index = [int(s) for s in x[0].split() if s.isdigit()][0]
            start = float(x[1].split(':')[-1])
            stop = float(x[2].split(':')[-1])
            overlap_info.append((index, start, stop))
        
        return overlap_info
    @staticmethod
    def _check_superposition(segment1, segment2):
        """
        Checks the superposition between two time intervals.

        Parameters:
        segment1 (list): The time interval of the first segment.
        segment2 (list): The time interval of the second segment.

        Returns:
        SuperpositionType: An enum indicating the type of superposition.
        """
        t_min1, t_max1 = segment1
        t_min2, t_max2 = segment2

        if t_min1 <= t_min2 <= t_max1 < t_max2:
            return SuperpositionType.STARTS_BEFORE_AND_OVERLAPS
        elif t_min2 <= t_min1 <= t_max2 < t_max1:
            return SuperpositionType.STARTS_AFTER_AND_OVERLAPS
        elif t_min1 <= t_min2 and t_max1 >= t_max2:
            return SuperpositionType.CONTAINS
        elif t_min2 <= t_min1 and t_max2 >= t_max1:
            return SuperpositionType.IS_CONTAINED
        else:
            return SuperpositionType.NO_SUPERPOSITION

    @staticmethod
    def _divide_labels(event, segments):
        """
        Divides an event into subevents by excluding specified segments.

        Parameters:
        event (list): A list containing the start and end times of the event [tmin, tmax].
        segments (list of lists): A list of [tmin, tmax] pairs representing the segments to exclude.

        Returns:
        list of lists: A list of [tmin, tmax] pairs representing the subevents.
        """
        tmin, tmax = event
        subevents = []
        start = tmin

        # Sort the segments by their start time
        segments.sort()

        for segment in segments:
            if segment[0] > start:
                subevents.append([start, segment[0]])
            # Update the start to the maximum between the end of the current segment and the current start
            start = max(start, segment[1])

        if start < tmax:
            subevents.append([start, tmax])

        return subevents
    
    def _handle_superposition(self, eval_idx, overlap_idx, superpos):
        """
        Handles the superposition between two segments.

        Parameters:
        eval_idx (int): The index of the evaluated segment.
        overlap_idx (int): The index of the overlapping segment.
        superpos (SuperpositionType): The type of superposition.

        Returns:
        bool: True if the evaluated segment should not be counted, False otherwise.
        """
        if superpos == SuperpositionType.STARTS_BEFORE_AND_OVERLAPS:
            self.df.at[eval_idx, 'tmax'] = self.df.loc[overlap_idx]["tmin"]
            self.df.at[overlap_idx, 'tmin'] = self.df.loc[eval_idx]["tmax"]
            return False
        elif superpos == SuperpositionType.STARTS_AFTER_AND_OVERLAPS:
            self.df.at[eval_idx, 'tmin'] = self.df.loc[overlap_idx]["tmax"]
            self.df.at[overlap_idx, 'tmax'] = self.df.loc[eval_idx]["tmin"]
            return False
        elif superpos == SuperpositionType.IS_CONTAINED:
            self.df.at[eval_idx, 'to_delete'] = True
            return True
        else:
            return False



class SuperpositionType(Enum):
    NO_SUPERPOSITION = 0
    STARTS_BEFORE_AND_OVERLAPS = 1
    STARTS_AFTER_AND_OVERLAPS = 2
    CONTAINS = 3
    IS_CONTAINED = 4        

if __name__ == "__main__":
    load_dotenv()
    ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
    ANNOTATIONS_PATH2 = os.getenv("ANNOTATIONS_PATH2")
    ANNOTATIONS_PATH3 = os.getenv("ANNOTATIONS_PATH3")
    # LABELS = 
    ecoss_data1 = EcossDataset(ANNOTATIONS_PATH, '.', 'zeros', 32000.0, 1,False)
    ecoss_data1.fix_onthology(labels=[])
    ecoss_data1.filter_overlapping()
    ecoss_data2 = EcossDataset(ANNOTATIONS_PATH2, '.', 'zeros', 32000.0, 1,False)
    ecoss_data2.fix_onthology(labels=[])
    ecoss_data2.filter_overlapping()
    ecoss_data3 = EcossDataset(ANNOTATIONS_PATH3, '.', 'zeros', 32000.0, 1,False)
    ecoss_data3.fix_onthology(labels=[])
    ecoss_data3.filter_overlapping()
    ecoss_list = [ecoss_data1, ecoss_data2, ecoss_data3]
    times = ecoss_data1.generate_insights()
    ecoss_data = EcossDataset.concatenate_ecossdataset(ecoss_list)
    times = ecoss_data.generate_insights()
    # ecoss_data.split_train_test_balanced(test_size=0.3, random_state=27)
    # signals, sr, paths, labels = ...
    # signals_processed, labels_processed = ecoss_data.process_all_data(signals_list=signals, original_sr_list=sr, paths_list=paths, labels_list=labels)
    