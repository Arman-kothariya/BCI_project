# data_loader.py

# I created this file to handle all data loading responsibilities.
# It supports both tabular EEG datasets (like CSV) and raw .npy EEG files.
# This way I can switch datasets easily without changing main code.

import os
import pandas as pd
import numpy as np

class DatasetLoader:
    def __init__(self, config):
        # First, get which dataset I'm using from the config file (e.g., 'default')
        use_dataset = config['use_dataset']
        
        # Then get all config related to that dataset (path, format, channels, etc.)
        self.dataset_config = config['datasets'][use_dataset]

        # Path where my EEG data is stored
        self.data_path = self.dataset_config['path']
        
        # Which EEG channels to use (important for selecting features)
        self.channels = self.dataset_config.get('channels', [])
        
        # The column name where my emotion/class labels are stored
        self.label_column = self.dataset_config.get('label_column')
        
        # Parameters for splitting data into train/test
        self.test_size = self.dataset_config.get('test_size', 0.2)
        self.random_state = self.dataset_config.get('random_state', 42)
        
        # The format of the dataset: 'tabular' = CSV/Excel, 'raw' = numpy arrays
        # NEW: Supports dynamic switching via config.yaml
        self.data_format = self.dataset_config.get('format', 'tabular')

    def load_data(self):
        """
        Load EEG data based on its format (tabular or raw)
        This function acts like a switch that calls the correct loader method.
        """
        if self.data_format == 'tabular':
            return self._load_tabular()
        elif self.data_format == 'raw':
            return self._load_raw()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

    def _load_tabular(self):
        """
        This method loads tabular EEG data from a CSV file.
        """
        data_file = None
        
        # Loop through files in the directory and find the first .csv file
        for f in os.listdir(self.data_path):
            if f.endswith('.csv'):
                data_file = os.path.join(self.data_path, f)
                break

        # If no CSV found, throw an error
        if data_file is None:
            raise FileNotFoundError(f"No CSV file found in {self.data_path}")

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(data_file)

        # Extract only the EEG channel data and the labels
        eeg_data = df[self.channels].values
        labels = df[self.label_column].values

        # Print shapes to make sure things look correct
        print(f"Loaded tabular EEG data shape: {eeg_data.shape}")
        print(f"Loaded labels shape: {labels.shape}")

        return eeg_data, labels

    def _load_raw(self):
        """
        This method loads raw EEG data and labels from .npy files.
        Useful for real-time or continuous EEG datasets.
        """
        eeg_data_file = os.path.join(self.data_path, "eeg_data.npy")
        labels_file = os.path.join(self.data_path, "labels.npy")

        # Check if both data and labels exist
        if not os.path.isfile(eeg_data_file) or not os.path.isfile(labels_file):
            raise FileNotFoundError("Raw EEG data or labels .npy files not found.")

        # Load the EEG data and labels
        eeg_data = np.load(eeg_data_file)
        labels = np.load(labels_file)

        # Print to verify shape and check if it loaded correctly
        print(f"Loaded raw EEG data shape: {eeg_data.shape}")
        print(f"Loaded labels shape: {labels.shape}")

        return eeg_data, labels
