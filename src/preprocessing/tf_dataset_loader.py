__all__ = ['load_dataset']

# import libraries
import os
import wfdb
import numpy as np

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
                # Try to convert value to int, float, or bool if possible
                if value.strip().lower() in ['true', 'false']:
                    comments_dict[key.strip()] = value.strip().lower() == 'true'
                else:
                    try:
                        num = int(value.strip())
                        comments_dict[key.strip()] = num
                    except ValueError:
                        try:
                            num = float(value.strip())
                            comments_dict[key.strip()] = num
                        except ValueError:
                            pass
            else:
                comments_dict[comment] = None
    return comments_dict

def _load_wfdb_record(record_path):
    """ Load a WFDB record and return the signal data, header, and labels."""
    try:
        record = wfdb.rdrecord(record_path)
        header = wfdb.rdheader(record_path)
        labels = _parse_header_comments(header)
        return record.p_signal[:SAMPLE_LIMIT, :], header, labels
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return None, None, None

def load_dataset(dataset_path):
    """ Load all WFDB records from a dataset directory and return the data and labels. """
    all_data = []
    all_labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(DAT_EXTENSION):
            record_path = os.path.join(dataset_path, filename[:-4])  # Remove .dat extension
            data, _, labels = _load_wfdb_record(record_path)
            if data is not None:
                all_data.append(data)
                all_labels.append(labels)
    
    # Stack the data
    X_all = np.stack(all_data, axis=0)  # Shape: (n_files, n_samples, n_channels)
    y_all = np.array(all_labels)        # Shape: (n_files,)
    return X_all, y_all
