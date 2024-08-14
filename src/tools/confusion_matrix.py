import matplotlib.pyplot as plt
from sklearn import metrics
import os
import numpy as np
import json

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