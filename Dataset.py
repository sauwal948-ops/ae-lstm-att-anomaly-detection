import os
import pandas as pd
import random
import numpy as np

class Dataset:
    DEFAULT_COLUMNS = [f'{key}_{i}' for key in ['position', 'velocity', 'effort'] for i in range(7)]
    CORRUPTION_TYPES = ['freeze_zero', 'freeze_last_value', 'spike', 'step']

    def __init__(self, name, dataset=None, window_size=None, scaler=None):
        self.name = name
        self.window_size = window_size
        self.scaler = scaler
        self.is_windowed = False
        self.is_normalized = False
        self.dataset = dataset
        self.dataset_processed = self.dataset.copy() if self.dataset is not None else None
        self.errors = None
        self.anomalies = {}

    def corrupt(self, corruption_type='random', save_path=None):
        if corruption_type == 'random' or corruption_type not in Dataset.CORRUPTION_TYPES:
            corruption_type = random.choice(Dataset.CORRUPTION_TYPES)
        self.name += "_" + corruption_type
        eval(f"self.{corruption_type}")()

    def freeze_zero(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")
        col = random.choice(self.dataset.columns.tolist())
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])
        self.dataset.loc[len(self.dataset) // 2:, col] = 0.0
        self.anomalies[col] = np.unique(np.append(self.anomalies[col],
            np.arange(len(self.dataset) // 2, len(self.dataset))))
        self.dataset_processed = self.dataset.copy()
        self.is_windowed = False
        self.is_normalized = False
        return self.anomalies

    def freeze_last_value(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")
        col = random.choice(self.dataset.columns.tolist())
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])
        self.dataset.loc[len(self.dataset) // 2:, col] = self.dataset.loc[len(self.dataset) // 2, col]
        self.anomalies[col] = np.unique(np.append(self.anomalies[col],
            np.arange(len(self.dataset) // 2, len(self.dataset))))
        self.dataset_processed = self.dataset.copy()
        self.is_windowed = False
        self.is_normalized = False
        return self.anomalies

    def spike(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")
        col = random.choice(self.dataset.columns.tolist())
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])
        for i in range(0, 1, 1):
            error = 500
            index = random.randint(0, len(self.dataset) - 1)
            self.dataset.loc[index, col] += error
            self.anomalies[col] = np.unique(np.append(self.anomalies[col], index))
        self.dataset_processed = self.dataset.copy()
        self.is_windowed = False
        self.is_normalized = False
        return self.anomalies

    def step(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")
        col = random.choice(self.dataset.columns.tolist())
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])
        error = random.randint(int(max(self.dataset[col])*5+1), int(20 * max(self.dataset[col])+1))
        self.dataset.loc[len(self.dataset) // 2:, col] += error
        self.anomalies[col] = np.unique(np.append(self.anomalies[col],
            np.arange(len(self.dataset) // 2, len(self.dataset))))
        self.dataset_processed = self.dataset.copy()
        self.is_windowed = False
        self.is_normalized = False
        return self.anomalies
