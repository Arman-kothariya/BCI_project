# modules/feature_extraction.py

import numpy as np
from scipy.stats import skew, kurtosis

# This function extracts a variety of basic statistical features from tabular EEG data.
# These features are useful for traditional ML models and give us a rough understanding of signal behavior.
# It works sample-wise (i.e., row-wise) assuming each row is one EEG sample with multiple channels.
def extract_extended_stats_features(eeg_data):
    """
    Extract extended statistical features per EEG sample.
    
    Args:
        eeg_data: np.ndarray of shape (samples, features) — e.g., shape (1000, 32)
    
    Returns:
        features: np.ndarray of shape (samples, n_features) — shape (1000, 7)
    """
    means = np.mean(eeg_data, axis=1)       # Average signal per sample
    variances = np.var(eeg_data, axis=1)    # Variability/spread
    skews = skew(eeg_data, axis=1)          # Asymmetry of signal
    kurtoses = kurtosis(eeg_data, axis=1)   # Peakedness / flatness
    medians = np.median(eeg_data, axis=1)   # Midpoint of the values
    ranges = np.ptp(eeg_data, axis=1)       # Peak-to-peak (max - min)
    stds = np.std(eeg_data, axis=1)         # Standard deviation

    # Stack all features into a single feature vector for each sample
    # Final shape: (samples, 7)
    features = np.vstack([means, variances, skews, kurtoses, medians, ranges, stds]).T
    return features


# This function is a placeholder for future use.
# It is designed for raw EEG signal (like .npy with shape: samples x time x channels).
# We might later use this for extracting features like power spectral density (PSD), band power, etc.
def extract_raw_eeg_features(eeg_data):
    """
    Placeholder for raw EEG time series feature extraction.
    You can implement PSD, wavelets, band power, etc.

    Args:
        eeg_data: np.ndarray of shape (samples, timepoints, channels) or similar

    Returns:
        features: np.ndarray of shape (samples, n_features)
    """
    # Not implemented yet. This will be used for time-series EEG features.
    raise NotImplementedError("Raw EEG feature extraction not implemented yet.")


# This function decides which feature extraction method to apply based on the format of EEG data.
# If the config format is 'tabular' or 'custom', we use the statistical summary method.
# If 'raw', we call the time-series placeholder (to be built later).
def extract_features(eeg_data, dataset_format):
    """
    Dispatch feature extraction based on dataset format.

    Args:
        eeg_data: np.ndarray
        dataset_format: str, 'tabular', 'custom' or 'raw'

    Returns:
        features: np.ndarray
    """
    if dataset_format in ['tabular', 'custom']:
        return extract_extended_stats_features(eeg_data)
    elif dataset_format == 'raw':
        return extract_raw_eeg_features(eeg_data)
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")
# This function can be used to extract features from EEG data in various formats.
# It will call the appropriate feature extraction function based on the dataset format specified.