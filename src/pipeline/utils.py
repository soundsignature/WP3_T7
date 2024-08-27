""" Script for useful functions """
    
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import pandas as pd
import numpy as np
import json
import librosa
import soundfile as sf
import yaml
from pathlib import Path
import torch.nn as nn
import torchaudio
import torch
import logging

UNWANTED_LABELS = ["Undefined"]
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_yaml(yaml_path: str) -> dict:
    """Function used to load the yaml content. Useful for reading configuration files or any other data stored in YAML format.

    Args:
        yaml_path (str): The absolute path to the yaml

    Returns:
        dict: The yaml content
    """
    with open(yaml_path, 'r') as file:
        try:
            yaml_content = yaml.safe_load(file)
            return yaml_content
        except yaml.YAMLError as e:
            logger.error(e)
            return None
        

def create_exp_dir(name: str, model: str, task: str) -> str:
    """
    Function to create a unique experiment directory. Useful for organizing experiment results by model and task.

    Args:
        name (str): The base name for the experiment directory.
        task (str): The name of the task associated with the experiment.
        model (str): The name of the model associated with the experiment.

    Returns:
        str: The path to the created experiment directory.
    """
    parent_path = Path(f'runs/{model}/{task}')
    parent_path.mkdir(exist_ok=True, parents=True)
    exp_path = str(parent_path / name) + "_{:02d}"
    i = 0
    while Path(exp_path.format(i)) in list(parent_path.glob("*")):
        i += 1
    exp_path = Path(exp_path.format(i))
    exp_path.mkdir(exist_ok=True)
    return str(exp_path)


def flatten(array):
    """
    Flatten a NumPy array.

    Parameters:
    - array (numpy.ndarray): The input array to be flattened.

    Returns:
    - flatten_array (numpy.ndarray): The flattened array.
    """
    flatten_array = array.flatten()
    return flatten_array

def process_data_for_inference(path_audio: str, desired_sr: float, desired_duration: float):
    """
    Process a single signal by resampling and segmenting or padding it.

    Parameters:
    signal (np.array): Signal array.
    original_sr (float): Original sampling rate of the signal.

    Returns:
    list: List of processed segments.
    """
    signal, original_sr = sf.read(path_audio)
    # Resample the signal if the original sampling rate is different from the target
    if original_sr != desired_sr:
        signal = librosa.resample(y=signal, orig_sr=original_sr, target_sr=desired_sr)
        
    # Pad the signal if it is shorter than the segment length   
    segment_length = (desired_duration*desired_sr)
    if len(signal) < segment_length:
        delta = segment_length - len(signal)
        delta_start = delta // 2
        delta_end = delta_start if delta%2 == 0 else (delta // 2) + 1 
        segments = np.pad(signal, (delta_start, delta_end), 'constant', constant_values=(0, 0))
        # Segment the signal if it is longer or equal to the segment length
    elif len(signal) >= segment_length:
        segments = []
        # Calculate the number of full segments in the signal
        n_segments = len(signal)//(segment_length)
        # Extract each segment and append to the list
        for i in range(n_segments):
            segment = signal[(i*segment_length):((i+1)*segment_length)]
            segments.append(segment)
        
    return segments

class AugmentMelSTFT(nn.Module):
    """ This class is used in order to generate the mel spectrograms for the EffAT and PaSST models
    """
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            logger.warning(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        # x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)  # Makes the mels look bad
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        # GOOD ONES
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec
    

class ConfusionMatrix:
    def __init__(self, labels_mapping_path: str) -> None:
        """
        Initialize the ConfusionMatrix class.
        """
        self.labels_mapping_path = labels_mapping_path

    def plot(self, y_true, y_pred):
        """
        Plot the confusion matrix.

        Parameters:
        - y (numpy.ndarray): True labels.
        - y_pred (numpy.ndarray): Predicted labels.

        Returns:
        - fig (matplotlib.figure.Figure): The generated matplotlib figure.
        """
        # Compute confusion matrix and normalize
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        
   
        with open(self.labels_mapping_path, 'r') as file:
            label_mapping = json.load(file)
            y_true = np.array([label_mapping[str(label)] for label in y_true])
            y_pred = np.array([label_mapping[str(label)] for label in y_pred])
        
        if len(list(np.unique(y_true)))>len(list(np.unique(y_pred))):
            labels = np.unique(y_true)
        else:
            labels = np.unique(y_pred)
        
        cm_percent = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))

        # Plot the normalized confusion matrix
        cm_display_percent = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
        cm_display_percent.plot(ax=ax1, cmap='Blues', values_format='.2%')

        # Plot the absolute confusion matrix
        cm_display_absolute = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
        cm_display_absolute.plot(ax=ax2, cmap='Blues', values_format='d')
        
        # Rotate labels in the second plot
        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_yticklabels(labels, rotation=45)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.set_yticklabels(labels, rotation=45)
        # Set a title for the entire figure
        
        fig.suptitle('Confusion Matrix', fontsize=11)

        return fig
    
    def save_plot(self, plot, saving_folder):
        plot.savefig(os.path.join(saving_folder, "ConfusionMatrix.png"))
        


class ValidationPlot:
    def __init__(self, gridsearch):
        self.gridsearch = gridsearch
        self.grid_results = gridsearch.cv_results_
        self.best_parameters = gridsearch.best_params_
        self.grid_params = self.grid_results['param_C']
        
    def get_filtered_dataset(self, df, filters):
        idx_to_drop = []
        for key, value in filters.items():
            for idx, row in df.iterrows():
                grid_key = 'param_' + key
                if row[grid_key] != value:
                    idx_to_drop.append(idx)
        unique_idx_to_drop = list(set(idx_to_drop))
        df_filtered = df.drop(unique_idx_to_drop)
        return df_filtered

    def plot(self):
        metric = 'accuracy'
        best_params_show = self.best_parameters['C']

        del self.best_parameters['C']
        df_grid = pd.DataFrame(self.grid_results)

        if len(self.best_parameters) != 0:
            df_grid = self.get_filtered_dataset(df=df_grid, filters=self.best_parameters)

        fig, ax = plt.subplots(figsize=(8, 6))

        validation_metric = df_grid['mean_test_score'].to_numpy()
        training_metric = df_grid['mean_train_score'].to_numpy()
 
        param_values = np.unique(np.array(list(self.grid_results['param_C'])))
        
        lim0_val = df_grid['mean_test_score'].to_numpy()-df_grid['std_test_score'].to_numpy()
        lim1_val = df_grid['mean_test_score'].to_numpy()+df_grid['std_test_score'].to_numpy()
    
        ax.fill_between(param_values, lim0_val, lim1_val, alpha=0.4, color = 'skyblue')
        ax.plot(param_values, validation_metric, marker='o', color='b', label='Validation ' + metric)
        
        lim0_tr = df_grid['mean_train_score'].to_numpy()-df_grid['std_train_score'].to_numpy()
        lim1_tr = df_grid['mean_train_score'].to_numpy()+df_grid['std_train_score'].to_numpy()
        
        ax.fill_between(param_values, lim0_tr, lim1_tr, alpha=0.4, color = 'lightcoral')
        ax.plot(param_values, training_metric, marker='o', color='r', label='Training ' + metric)

        ax.axvline(best_params_show, color ='green', linestyle= '--', alpha = 0.75, label = 'Best C') 
        
        ax.set_xlabel('C', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        fig.suptitle(f'Training and Validation {metric} for VGGish-SVM', fontsize=14, fontweight='bold' )
        ax.legend(loc = 'best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig

    def save_plot(self, plot, saving_folder):
        plot.savefig(os.path.join(saving_folder, "TrainigCurves.png"))
        
     
