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
import numpy as np
from enum import Enum

    
class EcossDataset:
    def __init__(self, annots_path: str, path_store_data: str, pad_mode: str,
                 sr: float, duration: float):
        self.annots_path = annots_path
        self.path_store_data = path_store_data
        self.pad_mode = pad_mode
        self.sr = sr
        self.duration = duration
        self.df = pd.read_csv(self.annots_path, sep=";", index_col=0)
    
    
    def filter_overlapping(self, visualize_overlap = True):
        """
        Filters overlapping segments in the dataset and optionally generates a representation of the timeline of labels before and after filtering.

        Parameters:
        visualize_overlap (bool): If True, visualizes the timeline of labels before and after processing.

        Returns:
        None
        """
        overlap_info_processed = self._extract_overlapping_info()
            
        self.df["overlap_info_processed"] = overlap_info_processed
        self.df.dropna(subset=["label_source"],inplace=True)
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
                if self.df.loc[eval_idx]["label_source"] != self.df.loc[overlap_idx]["label_source"]:
                    # Add to segments_to_delete everytime there is overlapping different class sources
                    segments_to_delete.append([tmin,tmax])
                else:
                    # Handle when the two overlapping segments are from the same class
                    t_eval = [self.df.loc[eval_idx]['tmin'],self.df.loc[eval_idx]["tmax"]]
                    t_overlap = [self.df.loc[overlap_idx]['tmin'],self.df.loc[overlap_idx]["tmax"]]
                    superpos = self._check_superposition(t_eval,t_overlap)
                    not_count = self._handle_superposition(eval_idx, overlap_idx, superpos)
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
        if visualize_overlap:
            self._visualize_overlappping(self.df,"_postprocessed") 
 
        
    def process_audios(self):
        pass
    
    
    def remap_onthology(self, labels: list):
        pass
    
    
    def generate_insights(self):
        pass
    
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
                labels.append(row['label_source'])
            
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
        elif superpos == SuperpositionType.STARTS_AFTER_AND_OVERLAPS:
            self.df.at[eval_idx, 'tmin'] = self.df.loc[overlap_idx]["tmax"]
            self.df.at[overlap_idx, 'tmax'] = self.df.loc[eval_idx]["tmin"]
        elif superpos == SuperpositionType.IS_CONTAINED:
            self.df.at[eval_idx, 'to_delete'] = True
            return True
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
    # LABELS = 
    ecoss_data = EcossDataset(ANNOTATIONS_PATH, '.', '.', '.', '.')
    ecoss_data.filter_overlapping()
    times = ecoss_data.generate_insights()

    