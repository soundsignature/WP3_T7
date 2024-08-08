import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

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

        # del self.best_parameters[self.grid_params]
        df_grid = pd.DataFrame(self.grid_results)

        # if len(self.best_parameters) != 0:
            # df_grid = self.get_filtered_dataset(df=df_grid, filters=self.best_parameters)

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

        ax.axvline(best_params_show, color ='green', linestyle= '--', alpha = 0.75, label = f'Best {str(self.grid_params)}') 
        
        ax.set_xlabel(str(self.grid_params), fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        fig.suptitle(f'Training and Validation {metric} for VGGish-SVM', fontsize=14, fontweight='bold' )
        ax.legend(loc = 'best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig

    def save_plot(self, plot, saving_folder):
        plot.savefig(os.path.join(saving_folder, "TrainigCurves.png"))