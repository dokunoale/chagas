from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def find_optimal_threshold(y_true, y_pred_proba) -> float:
    """
    Computes the ROC curve and returns the optimal threshold that maximizes TPR - FPR.

    Args:
        y_true: array-like, true labels (0/1)
        y_pred_proba: array-like, predicted probabilities for the positive class

    Returns:
        optimal_threshold: float, optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    return optimal_threshold


def compute_predictions(model, X_test, y_test, threshold) -> tuple:
    """
    Computes predictions from the model and evaluates them against the true labels.
    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: True labels for the test set.
        threshold: Threshold for binary classification.
    Returns:
        y_pred: Predicted probabilities.
        y_pred_class: Predicted classes (0 or 1).
        correct: Boolean array indicating if predictions are correct.
    """
    y_pred = model.predict(X_test).flatten()
    y_pred_class = (y_pred >= threshold).astype(int)
    correct = (y_pred_class == y_test)
    return y_pred, y_pred_class, correct


def show_spectrogram_from_sample(model, sample, index=1):
    """
    Displays the spectrogram output from the LogSpectrogram layer using a single sample.
    This function works only if the model has a LogSpectrogram layer.

    Parameters:
        model: The full CRNN model that includes the LogSpectrogram layer.
        sample: One ECG sample of shape (2800, 12).
        index: Index of the LogSpectrogram layer in the model. Default is 1.
    """

    # Get the LogSpectrogram layer output
    spectrogram_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=index).output)

    # Prepare sample: add batch dimension
    sample = tf.expand_dims(sample, axis=0)  # shape: (1, 2800, 12)

    # Get spectrogram output
    spectrogram = spectrogram_model(sample)  # shape: [1, time, freq, 12]
    spectrogram = spectrogram[0]  # remove batch dimension → [time, freq, 12]

    # Display each lead's spectrogram
    num_leads = spectrogram.shape[-1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    fig.suptitle("LogSpectrogram Output for Each ECG Lead")

    for i in range(num_leads):
        ax = axes[i // 4, i % 4]
        ax.imshow(spectrogram[..., i].numpy().T, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f"Lead {i+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def show_raw_ecg_from_sample(sample, title: str ="Raw ECG Signal for Each Lead", lead_names=None):
    """
    Displays the raw ECG signal for each lead using a single sample.

    Args:
        sample: One ECG sample of shape (2800, 12).
        title: Title for the entire figure. Default is "Raw ECG Signal for Each Lead".
        lead_names: Names for each lead. If None, uses "Lead 1", "Lead 2", etc.
    """
    # Convert to numpy if it's a tensor
    if hasattr(sample, 'numpy'):
        sample = sample.numpy()

    # Ensure we have the right shape
    if len(sample.shape) == 3 and sample.shape[0] == 1:
        sample = sample[0]  # Remove batch dimension if present

    # Default lead names
    if lead_names is None:
        lead_names = [f"Lead {i+1}" for i in range(12)]

    # Create time axis (assuming 500 Hz sampling rate for 2800 samples = 5.6 seconds)
    time = np.linspace(0, len(sample) / 500, len(sample))

    # Display each lead's raw signal
    num_leads = sample.shape[-1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(num_leads):
        ax = axes[i // 4, i % 4]
        ax.plot(time, sample[:, i], linewidth=0.8, color='blue')
        ax.set_title(lead_names[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)  # Assuming normalized signals

    plt.tight_layout()
    plt.show()