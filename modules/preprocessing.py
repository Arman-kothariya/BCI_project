# modules/preprocessing.py

import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.

    Args:
        lowcut (float): Low frequency cutoff in Hz.
        highcut (float): High frequency cutoff in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order.

    Returns:
        b, a (tuple): Filter coefficients.
    """
    nyq = 0.5 * fs  # Nyquist frequency (half the sampling rate)
    low = lowcut / nyq  # Normalized low cutoff
    high = highcut / nyq  # Normalized high cutoff
    b, a = butter(order, [low, high], btype='band')  # Get filter coefficients
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to EEG data.

    Args:
        data (np.ndarray): EEG data of shape (samples, channels).
        lowcut (float): Low frequency cutoff in Hz.
        highcut (float): High frequency cutoff in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.

    Returns:
        filtered_data (np.ndarray): Filtered EEG data, same shape as input.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # Get filter coefficients
    filtered_data = lfilter(b, a, data, axis=0)  # Apply filter to all channels
    return filtered_data

def preprocess(data, config):
    """
    Apply preprocessing steps to EEG data based on config.

    Args:
        data (np.ndarray): EEG data of shape (samples x channels).
        config (dict): Dictionary containing preprocessing parameters.

    Returns:
        data (np.ndarray): Preprocessed EEG data.
    """
    # If bandpass filter config is provided, apply it
    if 'bandpass' in config:
        bp = config['bandpass']
        data = bandpass_filter(
            data,
            lowcut=bp['lowcut'],
            highcut=bp['highcut'],
            fs=bp['fs'],
            order=bp.get('order', 5)  # Default order = 5
        )

    # Future preprocessing options to be added:
    # - Notch filter to remove powerline noise
    # - Z-score normalization
    # - Baseline correction
    # - Artifact rejection (e.g., eye blink, muscle noise)

    return data
