import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.tf_dataset_loader import load_dataset

def explore_dataset(dataset_path):
    """ Explore the dataset by loading and printing basic information. """
    print("loading dataset from:", dataset_path)
    data, labels = load_dataset(dataset_path, n=6, randomness=True, verbose=True)
    
    # Plot all signals
    print(f"Total samples loaded: {len(data)}")
    
    import matplotlib.pyplot as plt

    # Plot all 6 signals
    _, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i in range(len(data)):
        row = i // 3
        col = i % 3
        # Create time axis in seconds (sampling rate = 400 Hz)
        time_axis = [j / 400.0 for j in range(len(data[i]))]
        axes[row, col].plot(time_axis, data[i][:,6], label='Channel 1')
        axes[row, col].set_title(f'Signal {i+1} (Label: {labels[i]})')
        axes[row, col].set_xlabel('Time (seconds)')
        axes[row, col].set_ylabel('Amplitude')
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python explore.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    explore_dataset(dataset_path)