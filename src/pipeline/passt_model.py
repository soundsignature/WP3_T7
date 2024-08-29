# -*- coding: utf-8 -*-


"""
Created on Fri Aug 02 10:10:48 2024

@authors: Pablo Aguirre, Isabel Carozzo, Jose Antonio García, Mario Vilar
"""

""" This script implements the class PasstModel which is responsible for all the training, testing and inference stuff related with the
    PaSST model """

from pathlib import Path
from copy import deepcopy
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from argparse import Namespace

from models.passt.base import get_model
from models.passt.preprocess import AugmentMelSTFT
from models.passt.wrapper import PasstBasicWrapper
from torch.utils.data import Dataset
import torchaudio
from pipeline.utils import save_training_curves, save_confusion_matrix, process_audio_for_inference
import time

class StandardDataset(Dataset):
    def __init__(self, root_dir, sr=32000, duration=9.98, label_to_idx = None,time_threshold = 0.6):
        self.root_dir = Path(root_dir)
        self.sr = sr
        self.duration = duration
        self.labels = [item.name for item in self.root_dir.glob('*') if item.is_dir()]
        self.time_threshold = time_threshold
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
        return len(self.data)

    def __getitem__(self, idx):
        filepath, target = self.data[idx]
        # Load the audio file
        waveform, sr = torchaudio.load(str(filepath))
        waveform = waveform[0]
        return waveform, target, str(filepath)


class PasstModel():
    def __init__(self, yaml_content: dict, data_path: str) -> None:
        self.yaml = yaml_content
        self.data_path = Path(data_path)
        self.opt =Namespace(**self.yaml)
        

    
    def train(self,results_folder):
        self.results_folder = Path(results_folder)
        logging.info(f"Training PASST")
        output_config_path = self.results_folder / 'configuration.yaml'
        logging.info(f"Saving configuration in {output_config_path}")
        with open(str(output_config_path), 'w') as outfile:
            yaml.dump(self.yaml, outfile, default_flow_style=False)
        logging.info(f"Config params:\n {self.yaml}")
        
        train_dataloader, test_dataloader = self.load_train_test_datasets()
        self.loss_fn = CrossEntropyLoss()
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        model = self.train_model(model,train_dataloader,test_dataloader)
        
        self.test_model(model,train_dataloader,title = "train_result")
        self.test_model(model,test_dataloader,title = "val_result")
        
    def test(self,results_folder, path_model, path_data):
        logging.info(f"Testing PASST")
        self.results_folder = Path(results_folder) 
        path_model = Path(path_model)
        self.opt.weights_path = path_model
        with open(str(path_model.parent / "class_dict.json"),"r") as f:
            class_dict = json.load(f)
        self.n_classes = len(class_dict)
        self.loss_fn = CrossEntropyLoss()
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        model.eval()
        logging.info(f"Weights succesfully loaded into the model")
        test_dataloader = DataLoader(StandardDataset(self.data_path / "train",duration = self.opt.duration,sr=self.opt.sr))
        self.unique_labels = test_dataloader.dataset.labels
        self.test_model(model,test_dataloader,title = "test_result")
        
    
    
    def load_train_test_datasets(self):
        if (self.data_path / "train").exists() and (self.data_path / "test").exists():
            dataset = StandardDataset(self.data_path / "train",duration = self.opt.duration,sr=self.opt.sr)
            val_dataset = StandardDataset(self.data_path / "test",label_to_idx = dataset.label_to_idx,duration=self.opt.duration,sr=self.opt.sr)
            
        unique_labels = dataset.labels
        n_classes = len(unique_labels)
        

        # Obtener las etiquetas de todo el conjunto de datos
        targets = [target for _, target, _ in dataset]

        # Crear muestras ponderadas para equilibrar las clases
        class_count = [i for i in np.bincount(targets)]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

        train_targets = targets
        train_weights = [class_weights[t] for t in train_targets]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        
        # Obtener las etiquetas de todo el conjunto de datos
        targets = [target for _, target, _ in val_dataset]
        # Crear muestras ponderadas para equilibrar las clases
        class_count = [i for i in np.bincount(targets)]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
        test_targets = targets
        test_weights = [class_weights[t] for t in test_targets]
        test_sampler = WeightedRandomSampler(test_weights, len(train_weights))

    
        train_loader = DataLoader(dataset, batch_size=self.opt.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.opt.batch_size)
        
        self.unique_labels = unique_labels
        self.n_classes = n_classes
            
        return train_loader, val_loader

    def test_model(self,model,dataloader,title=""):
        # Test on val data
        val_loss = 0
        pred_labels = []
        true_labels = []
        model.eval()
        with torch.no_grad():
            for audio_wave, target, _ in tqdm(dataloader, desc=f"Final val:"):
                if audio_wave.shape[1] == 0:
                    continue
                if self.opt.gpu:
                    audio_wave = audio_wave.cuda()
                    target = target.cuda()
                logits = model(audio_wave)
                loss = self.loss_fn(logits, target)
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                pred_labels.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        val_loss /= len(dataloader)
        val_acc = accuracy_score(true_labels, pred_labels)
        val_f1 = f1_score(true_labels, pred_labels, average='macro',labels=np.unique(pred_labels))
        val_precision = precision_score(true_labels, pred_labels, average='macro',labels=np.unique(pred_labels))
        val_recall = recall_score(true_labels, pred_labels, average='macro',labels=np.unique(pred_labels))
        result = {
            "Accuracy" : val_acc,
            "Loss" : val_loss,
            "F1 Score" : val_f1,
            "Precision" : val_precision,
            "Recall ": val_recall
        }
        
        logging.info(result)
        if title and self.unique_labels and self.results_folder:
            with open(str(self.results_folder / title) + ".json", "w") as f:
                json.dump(result,f)
            save_confusion_matrix(self.unique_labels, self.results_folder,
                                    true_labels, pred_labels, title = title)
            report = classification_report(true_labels, pred_labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(str(self.results_folder / ("classification_report" + title + ".csv")))

        return val_loss,val_acc,val_f1,val_precision,val_recall


    def inference(self,results_folder, path_model, path_data):
        logging.info(f"Infering with PASST")
        self.results_folder = Path(results_folder) 
        path_model = Path(path_model)
        self.opt.weights_path = path_model
        with open(str(path_model.parent / "class_dict.json"),"r") as f:
            class_dict = json.load(f)
        self.n_classes = len(class_dict)
        model = self.create_model()
        if self.opt.gpu:
            model = model.cuda()
        model.eval()
        logging.info(f"Weights succesfully loaded into the model")

        y_list, sr = process_audio_for_inference(path_audio=self.data_path,
                            desired_sr=self.yaml["sr"],
                            desired_duration=self.yaml["duration"])
        class_list =  [key for key, value in sorted(class_dict.items(), key=lambda item: item[1])]
        results = []

        with torch.no_grad():
            for audio_wave in tqdm(y_list, desc=f"predict:"):
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
                                    "Connfidence by class" : confidence_by_class,
                                    "Inference time" : inferece_time
                                    } 
                        )
                

        logging.info(results)
        with open(str(Path(self.results_folder) / "results.json"),"w") as f:
            json.dump(results,f,default=str)




    def plot_results(self):
        pass


    def save_weights(self):
        pass


    def plot_processed_data(self,n = 1):
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

            if n > len(dataset):
                n = len(dataset)
            for i,(audio_wave, target, filename) in tqdm(enumerate(dataset)):

                if audio_wave.shape[0] == 0:
                    continue
                audio_wave = audio_wave
                if self.opt.gpu: 
                    audio_wave = audio_wave.cuda()
                mels.append(mel(audio_wave[np.newaxis,:]).cpu().numpy())

                label = target
                filenames.append(str(filename))
                if np.isnan(label):
                    labels.append("")
                else:
                    labels.append(idx_to_label[label])
                
                if i >= n:
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

        Example:
        >>> model = create_model(self.opt, 10)
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

    def train_model(self,model,train_dataloader,test_dataloader):
        """
        Trains the model.

        Parameters:
        model (torch.nn.Module): Model to be trained.
        train_dataloader (DataLoader): Training dataloader.
        test_dataloader (DataLoader): Validation dataloader.

        Returns:
        torch.nn.Module: Trained model.

        Example:
        >>> trained_model = train(self.opt, model, train_dataloader, test_dataloader, unique_labels, 10, exp_folder, loss_fn, cuda)
        """
        torch.cuda.empty_cache()

        logging.info(f"train samples: {len(train_dataloader) * self.opt.batch_size}")
        logging.info(f"val samples: {len(test_dataloader) * self.opt.batch_size}")
        logging.info(f"number of classes: {len(self.unique_labels)}")
        logging.info(f"classes dict: {train_dataloader.dataset.label_to_idx}")

        with open(Path(self.results_folder) / "class_dict.json", "w") as f:
            json.dump(train_dataloader.dataset.label_to_idx, f)

        logging.info(f"saving results to {self.results_folder}")

        result_path = Path(self.results_folder / "results.txt")
        create_train_txt(result_path, ["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_precision", "val_recall"])

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
            val_loss, val_acc, val_f1, val_precision, val_recall = self.test_model(model, test_dataloader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Save the model if the validation loss improves
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model_state_dict = deepcopy(model.state_dict())
                torch.save(best_model_state_dict, os.path.join(self.results_folder, f'checkpoint_{epoch + 1}.pth'))
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.opt.patience:
                    logging.info(f"Early stopping triggered after epoch {epoch + 1}")
                    break  # Exit the training loop

            update_train_txt(result_path, epoch, [train_loss, val_loss, val_acc, val_f1, val_precision, val_recall])

        torch.save(deepcopy(model.state_dict()), os.path.join(self.results_folder, f'last.pth'))

        save_training_curves(self.results_folder, train_losses, val_losses, val_accs)
        if best_model_state_dict is not None:  # Ensure a better model was found
            torch.save(best_model_state_dict, os.path.join(self.results_folder, 'best.pth'))
        model.load_state_dict(best_model_state_dict)
        return model
    
def create_train_txt(path,columns=[]):
    with open(str(path),"w") as f:
        f.write(','.join(columns))

def update_train_txt(path,epoch,stats=[]):
    stats = [f"{x:.4f}" for x in stats]
    with open(str(path),"a") as f:
        f.write(f"\n{int(epoch)}," + ','.join(stats))