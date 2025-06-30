__all__ = ['load_dataset']

# import libraries
import os
import wfdb
import numpy as np
from scipy.signal import resample_poly

# Constants
SAMPLE_LIMIT = 2800  # Limit the number of samples to load from each record
DAT_EXTENSION = '.dat'  # Extension for WFDB data files

def _parse_header_comments(header):
    """ Parse the comments from the WFDB header and return them as a dictionary. """
    comments_dict = {}
    if header.comments:
        for comment in header.comments:
            comment = comment.strip()
            if ':' in comment:
                key, value = comment.split(':', 1)
                comments_dict[key.strip()] = value.strip()
                if value.strip().lower() == 'true':
                    comments_dict[key.strip()] = True
                elif value.strip().lower() == 'false':
                    comments_dict[key.strip()] = False
                elif value.strip().isdigit():
                    comments_dict[key.strip()] = int(value.strip())
                elif value.strip().replace('.', '', 1).isdigit():
                    comments_dict[key.strip()] = float(value.strip())
            else:
                comments_dict[comment] = None
    return comments_dict

def _fix_sampling_rate(signal, labels, limit=SAMPLE_LIMIT):
    if labels.get('Source') == 'PTB-XL':
        signal = resample_poly(signal, up=4, down=5, axis=0)
    signal = signal[:limit, :]  # Limit the number of samples
    if signal.shape[0] < limit:
        # If the signal has fewer samples than the limit, pad with zeros
        padding = np.zeros((limit - signal.shape[0], signal.shape[1]))
        signal = np.vstack((signal, padding))
    
    # Normalize each channel independently between -1 and 1
    for i in range(signal.shape[1]):
        channel = signal[:, i]
        if np.ptp(channel) != 0:
            signal[:, i] = np.clip((channel - np.min(channel)) / np.ptp(channel) * 2 - 1, -1, 1)
    return signal

def _load_wfdb_record(record_path):
    """ Load a WFDB record and return the signal data, header, and labels."""
    try:
        print("Loading record:", record_path)
        record = wfdb.rdrecord(record_path)
        header = wfdb.rdheader(record_path)
        print("Parsing header for record:", record_path)
        labels = _parse_header_comments(header)
        print("Fixing sample rate:")
        return _fix_sampling_rate(record.p_signal, labels), labels
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return None, None

def load_dataset(dataset_path, n=None, randomness=False, verbose=False):
    """ Load all WFDB records from a dataset directory and return the data and labels. """
    all_data = []
    all_labels = []
    filenames = [f for f in os.listdir(dataset_path) if f.endswith(DAT_EXTENSION)]
    if randomness:
        np.random.shuffle(filenames)
    for filename in filenames[:n if n is not None else None]:
        record_path = os.path.join(dataset_path, filename[:-4])  # Remove .dat extension
        data, labels = _load_wfdb_record(record_path)
        if data is not None:
            if verbose:
                print(f"Loaded {filename}: {labels.get('Chagas label')} - {labels}")
            all_data.append(data)
            all_labels.append(labels.get('Chagas label'))
    
    # Stack the data
    X_all = np.stack(all_data, axis=0)  # Shape: (n_files, n_samples, n_channels)
    y_all = np.array(all_labels)        # Shape: (n_files,)
    return X_all, y_all

def concatenate_and_shuffle(positives, negatives):
    """
    Efficiently concatenate and shuffle positive and negative samples.
    
    This function combines positive and negative datasets and shuffles them randomly
    while minimizing memory usage by avoiding unnecessary array copies.
    
    Args:
        positives (tuple): A tuple (X_pos, y_pos) containing positive samples and labels
        negatives (tuple): A tuple (X_neg, y_neg) containing negative samples and labels
    
    Returns:
        tuple: A tuple (X_all, y_all) containing the shuffled combined dataset
            - X_all: Combined and shuffled feature arrays
            - y_all: Combined and shuffled label arrays
    """
    X_pos, y_pos = positives
    X_neg, y_neg = negatives
    
    # Get total size for pre-allocation
    total_size = len(y_pos) + len(y_neg)
    
    # Generate shuffle indices once
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    
    # Pre-allocate arrays with correct shape
    X_all = np.empty((total_size,) + X_pos.shape[1:], dtype=X_pos.dtype)
    y_all = np.empty(total_size, dtype=y_pos.dtype)
    
    # Fill arrays directly in shuffled order to avoid intermediate concatenation
    pos_indices = indices < len(y_pos)
    neg_indices = ~pos_indices
    
    X_all[pos_indices] = X_pos[indices[pos_indices]]
    X_all[neg_indices] = X_neg[indices[neg_indices] - len(y_pos)]
    y_all[pos_indices] = y_pos[indices[pos_indices]]
    y_all[neg_indices] = y_neg[indices[neg_indices] - len(y_pos)]
    
    return X_all, y_all