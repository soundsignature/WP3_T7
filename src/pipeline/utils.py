""" Script for useful functions """

import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns
import os
import torch.nn as nn
import torchaudio
import torch
import numpy as np

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
            print(e)
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

def process_audio_for_inference(path_audio: str, desired_sr: float, desired_duration: float):
    """It processes audios for inference purposes

    Args:
        path_audio (str): Path to the audio that needs to be processed
        desired_sr (float): The desired sampling rate
        desired_duration (float): The desired duration

    Raises:
        ValueError: In case the sampling rate of a signal is lower than the desired one.

    Returns:
        torch.Tensor: The processed signal
    """
    y, sr = torchaudio.load(path_audio)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sr)
    
    # Check sampling rate
    if sr < desired_sr:
        raise ValueError(f"Sampling rate of {sr} Hz is lower than the desired sampling rate of {desired_sr} Hz.")
    if sr > desired_sr:
        y = resampler(y)
        sr = desired_sr 

    # Check length
    length = int(desired_duration * desired_sr)
    if y.size(1) < length:
        y = torch.nn.functional.pad(y, (0, length - y.size(1)))
        y = y.unsqueeze(0)
    elif y.size(1) > length:
        chunk_size = desired_sr * desired_duration
        y = y.unfold(dimension=1, size=int(chunk_size), step=int(chunk_size))
    else:
        y = y.unsqueeze(0)

    return y, sr

def save_confusion_matrix(unique_labels, exp_folder, true_labels, predicted_labels, title = "confusion_matrix"):
    """
    Saves the confusion matrix and the normalized confusion matrix as SVG files in the specified folder.

    Parameters:
    unique_labels (list): List of unique labels used in the classification.
    exp_folder (str): Path to the folder where the files will be saved.
    true_labels (list or array): True labels of the data.
    predicted_labels (list or array): Labels predicted by the model.
    title (str, optional): Title for the confusion matrix plots. Default is "confusion_matrix".

    Returns:
    None
    """
    plt.rcParams.update({'font.size': 22})
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, f'{title}.png'))

    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} - Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, f'{title}-normalized.png'))
    plt.close()


def save_training_curves(exp_folder, train_losses, val_losses, val_accs):
    """
    Saves the training and validation curves as a PNG file in the specified folder.

    Parameters:
    exp_folder (str): Path to the folder where the file will be saved.
    train_losses (list or array): Training losses per epoch.
    val_losses (list or array): Validation losses per epoch.
    val_accs (list or array): Validation accuracies per epoch.

    Returns:
    None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(exp_folder, f'training_cuves.png'))





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
            print(f"Warning: FMAX is None setting to {fmax} ")
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
    