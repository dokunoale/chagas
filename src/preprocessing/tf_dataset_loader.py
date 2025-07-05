__all__ = ['load_dataset', 'concatenate_and_shuffle', 'WfdbLoader']

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
                    comments_dict[key.strip()] = 1
                elif value.strip().lower() == 'false':
                    comments_dict[key.strip()] = 0
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
        record = wfdb.rdrecord(record_path)
        header = wfdb.rdheader(record_path)
        labels = _parse_header_comments(header)
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

class WfdbLoader():
    """
    A class to load and preprocess WFDB datasets for machine learning tasks.
    
    This class allows adding multiple datasets, applying filters, and loading the data
    as NumPy arrays with optional shuffling and limiting the number of samples.
    """
    
    def __init__(self, label):
        self._records = []
        self._filters = []
        self._metadata = []
        self._label = label
        
    def add_dataset(self, dataset_path):
        """ Add a dataset path to the loader. """
        self._records.extend([os.path.join(dataset_path, f[:-4]) for f in os.listdir(dataset_path) if f.endswith(DAT_EXTENSION)])
    
    def add_filter(self, filter_func):
        """ Add a filter function to be applied to the data. """
        self._filters.append(filter_func)

    def get_metadata(self):
        """ Get metadata from the loaded records. """
        return self._metadata
    
    def load(self, shuffle=False, limit=None, verbose=False):
        """ Load all datasets and apply filters. """
        all_data = []
        all_labels = []

        if shuffle:
            np.random.shuffle(self._records)
        
        records_to_process = self._records[:limit if limit is not None else None]
        
        if verbose:
            from tqdm import tqdm
            records_to_process = tqdm(records_to_process, desc="Loading records", unit="record")
            
        for record in records_to_process:
            data, self._metadata = _load_wfdb_record(record)

            if data is not None:              
                all_data.append(data)
                all_labels.append(self._metadata.get(self._label))
        
        # Stack the data
        X_all = np.stack(all_data, axis=0)  # Shape: (n_files, n_samples, n_channels)
        y_all = np.array(all_labels)        # Shape: (n_files,)

        for filter_func in self._filters:
            X_all = filter_func(X_all)
            
        return X_all, y_all
                

"""
# Example usage:
# loader = WfdbLoader()
# loader.add_filter(NoiseFilter())
# loader.add_dataset('path/to/positive/dataset')
# loader.add_dataset('path/to/negative/dataset')
# loader.load(shuffle=True, limit=1000)
# """