# -*- coding: utf-8 -*-


"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar

This script implements the class PasstModel which is responsible for all the training, testing and inference stuff related with the
    PaSST model
"""

from pathlib import Path
from copy import deepcopy
import logging
import os
import json
from argparse import Namespace
import time
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
import torchaudio

import sys
PROJECT_FOLDER = os.path.dirname(__file__).replace('/pipeline', '/models')
PARENT_PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
sys.path.append(PARENT_PROJECT_FOLDER)

from models.passt.base import get_model
from models.passt.preprocess import AugmentMelSTFT
from models.passt.wrapper import PasstBasicWrapper
from pipeline.utils import save_training_curves, save_confusion_matrix, process_audio_for_inference

class HelperDataset(Dataset):
    """
    Helper dataset for loading audio files and their corresponding labels.

    Args:
        root_dir (str or Path): Root directory containing subdirectories of labels with audio files.
        sr (int, optional): Sampling rate for the audio files. Default is 32000.
        duration (float, optional): Duration of the audio clips in seconds. Default is 9.98.
        label_to_idx (dict, optional): Dictionary mapping labels to indices. If not provided, it will be generated automatically.
        time_threshold (float, optional): Time threshold for a specific operation. Default is 0.6.

    Attributes:
        root_dir (Path): Root directory converted to a Path object.
        sr (int): Sampling rate.
        duration (float): Duration of the audio clips.
        labels (list): List of labels obtained from the subdirectories in root_dir.
        label_to_idx (dict): Dictionary mapping labels to indices.
        data (list): List of tuples (audio file path, label index).
        resamplers (list): List of resamplers, initialized with a default value.
    """
    def __init__(self, root_dir, sr=32000, duration=9.98, label_to_idx = None):
        self.root_dir = Path(root_dir)
        self.sr = sr
        self.duration = duration
        self.labels = [item.name for item in self.root_dir.glob('*') if item.is_dir()]
        if not label_to_idx:
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        else:
            self.label_to_idx = label_to_idx
        self.data = []
        for label in self.labels:
            for audio_file in self.root_dir.joinpath(label).rglob('*.wav'):
                self.data.append((audio_file, self.label_to_idx[label]))

        self.resamplers = [(None,None)]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio waveform, the label index, and the file path.
        """
        filepath, target = self.data[idx]
        # Load the audio file
        waveform, _ = torchaudio.load(str(filepath))
        waveform = waveform[0]
        return waveform, target, str(filepath)


class PasstModel():
    """
    PASST Model for training and testing audio classification tasks.

    Args:
        yaml_content (dict): Configuration parameters for the model.
        data_path (str): Path to the dataset.

    Attributes:
        yaml (dict): Configuration parameters.
        data_path (Path): Path to the dataset as a Path object.
        opt (Namespace): Configuration parameters as a Namespace object.
    """
    def __init__(self, yaml_content: dict, data_path: str) -> None:
        self.yaml = yaml_content
        self.data_path = Path(data_path)
        self.opt = Namespace(**self.yaml)
        self.loss_fn = CrossEntropyLoss()
        self.n_classes = None # Should be calculated later and overwritted
        self.unique_labels = None # Should be calculated later and overwritted
        self.optimizer = None # Should be calculated later and overwritted


    def train(self,results_folder):
        """
        Train the PASST model.

        Args:
            results_folder (str): Path to the folder where results will be saved.
        """
        results_folder = Path(results_folder)
        logging.info("Training PASST")
        output_config_path = results_folder / 'configuration.yaml'
        logging.info(f"Saving configuration in {output_config_path}")
        with open(str(output_config_path), 'w', encoding="utf-8") as outfile:
            yaml.dump(self.yaml, outfile, default_flow_style=False)
        logging.info(f"Config params:\n {self.yaml}")

        train_dataloader, test_dataloader = self.load_train_test_datasets()
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        model = self.train_model(model,train_dataloader,test_dataloader,results_folder)

        self.test_model(model,train_dataloader,results_folder,title = "train_result")
        self.test_model(model,test_dataloader,results_folder,title = "val_result")

    def test(self,results_folder, path_model, path_data = None):
        """
        Test the PASST model.

        Args:
            results_folder (str): Path to the folder where results will be saved.
            path_model (str): Path to the trained model weights.
            path_data (str): Path to the dataset.
        """
        logging.info("Testing PASST")
        results_folder = Path(results_folder)
        path_model = Path(path_model)
        self.opt.weights_path = path_model
        with open(str(path_model.parent / "class_dict.json"),"r", encoding="utf-8") as f:
            class_dict = json.load(f)
        self.n_classes = len(class_dict)
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        model.eval()
        logging.info("Weights succesfully loaded into the model")
        test_dataloader = DataLoader(HelperDataset(self.data_path / "train",duration = self.opt.duration,sr=self.opt.sr))
        self.unique_labels = test_dataloader.dataset.labels
        self.test_model(model,test_dataloader,results_folder,title = "test_result")

    def inference(self,results_folder, path_model, path_data):
        """
        Perform inference using the PASST model.

        Args:
            results_folder (str): Path to the folder where results will be saved.
            path_model (str): Path to the trained model weights.
            path_data (str): Path to the dataset.
        """
        logging.info("Infering with PASST")
        results_folder = Path(results_folder)
        path_model = Path(path_model)
        self.opt.weights_path = path_model
        with open(str(path_model.parent / "class_dict.json"),"r",encoding="utf-8") as f:
            class_dict = json.load(f)
        self.n_classes = len(class_dict)
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        model.eval()
        logging.info("Weights succesfully loaded into the model")

        y_list, sr = process_audio_for_inference(path_audio=self.data_path,
                            desired_sr=self.yaml["sr"],
                            desired_duration=self.yaml["duration"])

        assert sr == self.yaml["sr"], "inconsistent sampling rate"
        class_list =  [key for key, value in sorted(class_dict.items(), key=lambda item: item[1])]
        results = []

        with torch.no_grad():
            for audio_wave in tqdm(y_list, desc="predict:"):
                if audio_wave.shape[1] == 0:
                    continue
                start = time.time()
                if self.opt.gpu:
                    audio_wave = audio_wave.cuda()
                    target = target.cuda()
                logits = model(audio_wave)
                precentage = torch.nn.Softmax(dim=1)(logits)
                _, predicted = torch.max(logits.data, 1)
                inferece_time = time.time()- start

                for i in range(len(predicted.cpu().numpy())):
                    confidence_by_class = dict(zip(class_list,precentage.cpu().numpy()[i]))
                    results.append({"Predicted Class" : class_list[predicted.cpu().numpy()[i]],
                                    "Confidence by class" : confidence_by_class,
                                    "Inference time" : inferece_time
                                    }
                        )

        logging.info(results)
        with open(str(Path(results_folder) / "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, default=str)


    def test_model(self, model, dataloader, results_folder, title=""):
        """
        Evaluate the performance of a given model on a validation dataset.

        Args:
        -----------
        model : torch.nn.Module
            The PyTorch model to be evaluated.
        dataloader : torch.utils.data.DataLoader
            DataLoader providing the validation data.
        title : str, optional
            Title for saving results and confusion matrix (default is "").

        Returns:
        --------
        tuple
            A tuple containing the validation loss, accuracy, F1 score, precision, and recall.
        """
        # Initialize metrics
        val_loss = 0
        pred_labels = []
        true_labels = []

        # Set model to evaluation mode
        model.eval()

        # Disable gradient computation
        with torch.no_grad():
            for audio_wave, target, _ in tqdm(dataloader, desc="Final val:"):
                if audio_wave.shape[1] == 0:
                    continue
                if self.opt.gpu:
                    audio_wave = audio_wave.cuda()
                    target = target.cuda()

                # Forward pass
                logits = model(audio_wave)
                loss = self.loss_fn(logits, target)
                val_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(logits.data, 1)
                pred_labels.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        # Calculate average loss
        val_loss /= len(dataloader)

        # Calculate metrics
        val_acc = accuracy_score(true_labels, pred_labels)
        val_f1 = f1_score(true_labels, pred_labels, average='macro', labels=np.unique(pred_labels))
        val_precision = precision_score(true_labels, pred_labels, average='macro', labels=np.unique(pred_labels))
        val_recall = recall_score(true_labels, pred_labels, average='macro', labels=np.unique(pred_labels))

        # Compile results
        result = {
            "Accuracy": val_acc,
            "Loss": val_loss,
            "F1 Score": val_f1,
            "Precision": val_precision,
            "Recall": val_recall
        }

        # Log results
        logging.info(result)

        # Save results and confusion matrix if title and necessary attributes are provided
        if title and self.unique_labels and results_folder:
            with open(str(results_folder / title) + ".json", "w", encoding = "utf-8") as f:
                json.dump(result, f)
            save_confusion_matrix(self.unique_labels, results_folder, true_labels, pred_labels, title=title)
            report = classification_report(true_labels, pred_labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(str(results_folder / ("classification_report" + title + ".csv")))

        return val_loss, val_acc, val_f1, val_precision, val_recall



    def load_train_test_datasets(self):
        """
        Load and prepare the training and testing datasets.

        Returns:
            tuple: A tuple containing the training DataLoader and the validation DataLoader.
        """
        if (self.data_path / "train").exists() and (self.data_path / "test").exists():
            dataset = HelperDataset(self.data_path / "train",duration = self.opt.duration,sr=self.opt.sr)
            test_dataset = HelperDataset(self.data_path / "test",label_to_idx = dataset.label_to_idx,duration=self.opt.duration,sr=self.opt.sr)

        unique_labels = dataset.labels
        n_classes = len(unique_labels)


        # Create weighted samples to balance classes
        train_sampler = self.balance_dataset(dataset)

        train_loader = DataLoader(dataset, batch_size=self.opt.batch_size, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=self.opt.batch_size)

        self.unique_labels = unique_labels
        self.n_classes = n_classes

        return train_loader, test_loader

    def balance_dataset(self, dataset):
        """
        Create a weighted sampler to balance the classes in the dataset.

        Args:
            dataset (Dataset): The dataset to be balanced.

        Returns:
            tuple: A tuple containing the list of weights for each sample and the WeightedRandomSampler.
        """
        targets = [target for _, target, _ in dataset]

        # Create weighted samples to balance classes
        class_count = [i for i in np.bincount(targets)]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler


    def plot_results(self):
        pass


    def save_weights(self):
        pass


    def plot_processed_data(self,n_of_samples_to_plot = 1):
        """
        Plot the processed data (mel spectrograms) for a given number of samples from the training and test datasets.

        Parameters:
        -----------
        n_of_samples_to_plot : int, optional
            The number of samples to plot from each dataset (default is 1).

        Returns:
        --------
        None
        """

        train_dataloader, test_dataloader = self.load_train_test_datasets()
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        for dataset in [train_dataloader,test_dataloader]:
            dataset = dataset.dataset
            mel = model.mel
            mels=[]
            labels = []
            filenames = []
            idx_to_label= {v: k for k, v in dataset.label_to_idx.items()}

            if n_of_samples_to_plot > len(dataset):
                n_of_samples_to_plot = len(dataset)
            for i,(audio_wave, target, filename) in tqdm(enumerate(dataset)):

                if audio_wave.shape[0] == 0:
                    continue
                if self.opt.gpu:
                    audio_wave = audio_wave.cuda()
                mels.append(mel(audio_wave[np.newaxis,:]).cpu().numpy())

                label = target
                filenames.append(str(filename))
                if np.isnan(label):
                    labels.append("")
                else:
                    labels.append(idx_to_label[label])

                if i >= n_of_samples_to_plot:
                    break

            for i,x in enumerate(mels):
                plt.figure()
                plt.imshow(x[0],origin="lower")
                plt.title(f"{str(Path(filenames[i]).stem)}-{labels[i]}")
                plt.show()

    def create_model(self):
        """
        Creates the model based on the provided configuration.

        Returns:
        torch.nn.Module: Created model.

        """
        if self.opt.preprocess_type == "mel":
            mel = AugmentMelSTFT(n_mels=self.opt.n_mels,
                    sr=self.opt.sr,
                    win_length=self.opt.win_length,
                    hopsize=self.opt.hopsize,
                    n_fft=self.opt.n_fft,
                    freqm=self.opt.freqm,
                    timem=self.opt.timem,
                    htk=self.opt.htk,
                    fmin=self.opt.fmin,
                    fmax=self.opt.fmax,
                    norm=self.opt.norm,
                    fmin_aug_range=self.opt.fmin_aug_range,
                    fmax_aug_range=self.opt.fmax_aug_range
                    )
        else:
            raise ValueError("preprocess_type should be 'mel'")


        # Define the transformer
        net = get_model(arch=self.opt.model_architecture,
                            n_classes=self.n_classes,
                            pretrained=self.opt.pretrained,
                            in_channels=self.opt.in_channels,
                            fstride=self.opt.fstride,
                            tstride=self.opt.tstride,
                            input_fdim=self.opt.input_fdim,
                            input_tdim=self.opt.input_tdim,
                            u_patchout=self.opt.u_patchout,
                            s_patchout_t=self.opt.s_patchout_t,
                            s_patchout_f=self.opt.s_patchout_f)

        # Wrap preprocess and transformer
        model = PasstBasicWrapper(mel=mel, net=net, mode = "logits")


        if self.opt.weights_path:
            # load the custom weights model state dict
            state_dict = torch.load(self.opt.weights_path)

            # I had to add this because apparently toch.save is adding somo prefix to configurations
            remove_prefix = 'net.'
            state_dict = {k[len(remove_prefix):] if k.startswith(
                remove_prefix) else k: v for k, v in state_dict.items()}
            state_dict. pop('device_proxy', None)

            # load the weights into the transformer
            model.net.load_state_dict(state_dict)
        return model

    def train_model(self,model,train_dataloader,test_dataloader,results_folder):
        """
        Trains the model.

        Parameters:
        model (torch.nn.Module): Model to be trained.
        train_dataloader (DataLoader): Training dataloader.
        test_dataloader (DataLoader): Validation dataloader.

        Returns:
        torch.nn.Module: Trained model.

        """
        torch.cuda.empty_cache()

        logging.info(f"train samples: {len(train_dataloader) * self.opt.batch_size}")
        logging.info(f"val samples: {len(test_dataloader) * self.opt.batch_size}")
        logging.info(f"number of classes: {len(self.unique_labels)}")
        logging.info(f"classes dict: {train_dataloader.dataset.label_to_idx}")

        with open(Path(results_folder) / "class_dict.json", "w", encoding="utf-8") as f:
            json.dump(train_dataloader.dataset.label_to_idx, f)

        logging.info(f"saving results to {results_folder}")

        result_path = Path(results_folder / "results.csv")
        create_train_csv(result_path, ["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_precision", "val_recall"])

        if self.opt.freeze_layers:
            # Freeze all layers except the last ones
            for name, param in model.named_parameters():
                if 'head' not in name and 'head_dist' not in name:
                    param.requires_grad = False
            model_params = filter(lambda p: p.requires_grad, model.parameters())
        else:
            model_params = model.parameters()

        if self.opt.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=self.opt.lr0, weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=self.opt.lr0, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        else:
            raise NotImplementedError("Optimizer not defined")

        if self.opt.scheduler == "OneCycleLR":
            total_steps = int(self.opt.num_epochs * len(train_dataloader))
            scheduler = OneCycleLR(self.optimizer, max_lr=self.opt.lr0, total_steps=total_steps, pct_start=self.opt.warmup_epochs / self.opt.num_epochs, anneal_strategy='cos', final_div_factor=1 / self.opt.lrf)
        else:
            scheduler = None

        # Early Stopping configuration
        min_delta = 0.001  # Minimum change in loss
        early_stopping_counter = 0

        train_losses = []
        val_losses = []
        val_accs = []

        # Train the model
        best_val_loss = float('inf')
        for epoch in range(self.opt.num_epochs):
            train_loss = 0
            model.train()
            for audio_wave, target, _ in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
                if audio_wave.shape[1] == 0:
                    continue
                self.optimizer.zero_grad()
                if self.opt.gpu:
                    audio_wave = audio_wave.cuda()
                    target = target.cuda()
                logits = model(audio_wave)
                loss = self.loss_fn(logits, target)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if scheduler:
                    scheduler.step()

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            logging.info(f"Epoch {epoch + 1} completed, Train Loss: {train_loss}")

            # Test on the validation dataset
            val_loss, val_acc, val_f1, val_precision, val_recall = self.test_model(model, test_dataloader,results_folder)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Save the model if the validation loss improves
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model_state_dict = deepcopy(model.state_dict())
                torch.save(best_model_state_dict, os.path.join(results_folder, f'checkpoint_{epoch + 1}.pth'))
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.opt.patience:
                    logging.info(f"Early stopping triggered after epoch {epoch + 1}")
                    break  # Exit the training loop

            update_train_csv(result_path, epoch, [train_loss, val_loss, val_acc, val_f1, val_precision, val_recall])

        torch.save(deepcopy(model.state_dict()), os.path.join(results_folder, 'last.pth'))

        save_training_curves(results_folder, train_losses, val_losses, val_accs)
        if best_model_state_dict is not None:  # Ensure a better model was found
            torch.save(best_model_state_dict, os.path.join(results_folder, 'best.pth'))
        model.load_state_dict(best_model_state_dict)
        return model

def create_train_csv(path, columns=None):
    """
    Create a CSV file with the specified columns for logging training statistics.

    Parameters:
    -----------
    path : str or Path
        The file path where the CSV file will be created.
    columns : list of str, optional
        A list of column names to be written as the header of the CSV file (default None).

    Returns:
    --------
    None
    """
    if columns is None:
        columns = []
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(','.join(columns))


def update_train_csv(path, epoch, stats=None):
    """
    Append a new row of training statistics to the CSV file.

    Parameters:
    -----------
    path : str or Path
        The file path of the CSV file to be updated.
    epoch : int
        The current epoch number.
    stats : list of float, optional
        A list of training statistics to be appended to the CSV file (default to Nonet).

    Returns:
    --------
    None
    """
    if stats is None:
        stats = []
    stats = [f"{x:.4f}" for x in stats]
    with open(str(path), "a", encoding="utf-8") as f:
        f.write(f"\n{int(epoch)}," + ','.join(stats))