"""This script is used to run some checks on annotations, specially pulse types. It should be run every time new annotations come from W+B.
    If mistakes are seen with this script, then contact W+B and SHOM.
"""

import librosa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import logging
import os
import sys
import torch
from dotenv import load_dotenv

PROJECT_FOLDER = os.path.dirname(__file__).replace('/src', '/pipeline')
PARENT_PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
sys.path.append(PARENT_PROJECT_FOLDER)

from src.pipeline.utils import load_yaml, AugmentMelSTFT

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

handler = logging.FileHandler("log.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AnnotationsChecker():
    def __init__(self, yaml_content: dict, dataset_path: str, source: str):
        self.yaml = yaml_content
        self.source = source
        self.dataset_path = dataset_path
        # self.mel = AugmentMelSTFT(freqm=self.yaml["freqm"],
        #                           timem=self.yaml["freqm"],
        #                           n_mels=self.yaml["n_mels"],
        #                           sr=self.yaml["sr"],
        #                           win_length=self.yaml["win_length"],
        #                           hopsize=self.yaml["hopsize"],
        #                           n_fft=self.yaml["n_fft"],
        #                           fmin=self.yaml["fmin"],
        #                           fmax=self.yaml["fmax"],
        #                           fmax_aug_range=self.yaml["fmax_aug_range"],
        #                           fmin_aug_range=self.yaml["fmin_aug_range"])
        self.threshold = yaml_content["threshold"]
        self.desired_sr = float(yaml_content["sr"])
        self.df = pd.read_csv(os.path.join(dataset_path, 'samples for training', 'annotations.csv'), sep=";")


    def filter_pulses(self):
        types = ["Pulse", "FrequencyModulation", "Transient"]
        indexes = []
        for i, row in self.df.iterrows():
            for t in types:
                if (t in row["label_type"]) & (self.source in row["label_source"]):
                    indexes.append(i)
        indexes = list(set(indexes))
        self.df = self.df.loc[indexes, :]
        self.df["duration"] = self.df["tmax"] - self.df["tmin"]


    def filter_by_threshold(self):
        self.df = self.df[self.df["duration"] > self.threshold]

    def get_sorted_durations(self):
        durations = self.df["duration"].sort_values(ascending = False)
        plt.figure()
        plt.hist(durations, bins=100)
        plt.xlabel("duration (s)")
        plt.ylabel("number of labels")
        plt.show()
        return durations      


    def plot_filtered_signals(self, path_store: str, cut_signal: float = 0, resample: bool = False):
        for i, row in tqdm(self.df.iterrows()):
            filename = row["reference"]
            filepath = os.path.join(self.dataset_path, 'samples for training', filename)

            y, sr = librosa.load(filepath, sr=None)
            print(f"SR: {sr}")
            
            if resample == True:
                if sr >= self.desired_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.desired_sr)
                    sr = self.desired_sr
                else:
                    continue
                
            y = y[int(row["tmin"]*sr):int(row["tmax"]*sr)]

            if cut_signal != 0:
                y = torch.Tensor(y)
                signal_length = y.size(0)  # Longitud total de la señal
                window_size = int(cut_signal * signal_length)
                y = y.unfold(0, window_size, window_size)
                y = y.numpy()
                for w in tqdm(range(y.shape[0])):
                    window = y[w, :]
                    S = np.abs(librosa.stft(window))
                    S_dB = librosa.amplitude_to_db(S, ref=np.max)

                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='linear')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(filename + '; ' + 'tmin: ' + str(row["tmin"]) + '; tmax:' + str(row["tmax"]) + f'; fmin: {row["fmin"]}; fmax: {row["fmax"]}')
                    plt.tight_layout()
                    ax = plt.gca()
                    rect = Rectangle((0.1, row["fmin"]), row["tmax"]-1, row["fmax"]-row["fmin"], linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.savefig(os.path.join(path_store, f"{filename}_{row['tmin']}_{row['tmax']}_chunk_{w}.png"))
                    plt.close()

            else:
                S = np.abs(librosa.stft(y))
                S_dB = librosa.amplitude_to_db(S, ref=np.max)

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='linear')
                plt.colorbar(format='%+2.0f dB')
                plt.title(filename + '; ' + 'tmin: ' + str(row["tmin"]) + '; tmax:' + str(row["tmax"]) + f'; fmin: {row["fmin"]}; fmax: {row["fmax"]}')
                plt.tight_layout()
                ax = plt.gca()
                rect = Rectangle((0.1, row["fmin"]), row["tmax"]-1, row["fmax"]-row["fmin"], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.savefig(os.path.join(path_store, f"{filename}_{row['tmin']}_{row['tmax']}.png"))
                plt.close()


if __name__ == "__main__":
    load_dotenv()

    dataset_path = os.getenv("DATASET_PATH_CHECK")
    yaml_content = load_yaml(os.getenv("YAML_LABELS_CHECK"))

    path_store = os.path.join(os.getenv("STORE_PATH_CHECK"), os.path.basename(dataset_path), yaml_content["source"])
    os.makedirs(path_store, exist_ok=True)

    checker = AnnotationsChecker(yaml_content=yaml_content, dataset_path=dataset_path, source=yaml_content["source"])
    print(len(checker.df))
    checker.filter_pulses()
    print(len(checker.df))
    top_durations = checker.get_sorted_durations()
    print(f"The top five durations are {top_durations}")
    checker.filter_by_threshold()
    checker.plot_filtered_signals(path_store=path_store, cut_signal=0, resample=True)
