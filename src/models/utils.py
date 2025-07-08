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


def make_all_gradcam_heatmaps_gru(ecg_sample, model, gru_layer_name="gru", pred_index=None):
    """
    Generates Grad-CAM heatmaps for a GRU layer in a Keras model.

    Args:
        ecg_sample: A single ECG sample of shape (2800, 12).
        model: The trained Keras model containing the GRU layer.
        gru_layer_name: Name of the GRU layer in the model. Default is "gru".
        pred_index: Index of the class for which to compute the heatmap. \
            If None, it uses the class with the highest predicted probability.
    
    Returns:
        heatmaps: List of heatmaps for each GRU unit.
    """
    # Find the GRU layer in the model
    gru_layer = model.get_layer(gru_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            gru_layer.output,
            model.output
        ]
    )

    ecg_input = tf.expand_dims(ecg_sample, axis=0)  # (1, 2800, 12)

    with tf.GradientTape() as tape:
        gru_outputs, predictions = grad_model(ecg_input)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, gru_outputs)

    # Reshape outputs and grads to match the expected dimensions
    if len(gru_outputs.shape) == 3:  # (batch, timesteps, units)
        gru_outputs = gru_outputs[0]  # (timesteps, units)
        grads = grads[0]              # (timesteps, units)
    elif len(gru_outputs.shape) == 2:  # (batch, units) - solo ultimo output
        gru_outputs = gru_outputs[0]  # (units,)
        grads = grads[0]              # (units,)
        raise Warning("GRU layer returns only last output, not full sequence")

    # Heatmap calculation
    heatmaps = []
    if len(gru_outputs.shape) == 2:  # Sequenza completa
        for i in range(gru_outputs.shape[-1]):
            weighted_map = gru_outputs[..., i] * grads[..., i]
            heatmap = tf.maximum(weighted_map, 0)
            heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
            heatmaps.append(heatmap.numpy())
    else:  # Only last output
        for i in range(gru_outputs.shape[-1]):
            weighted_val = gru_outputs[i] * grads[i]
            heatmap = tf.maximum(weighted_val, 0)
            heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
            heatmaps.append(heatmap.numpy())

    return heatmaps


def display_all_heatmaps_gru(heatmaps, cols=8):
    """
    Displays all GRU heatmaps in a grid layout.
    If heatmaps are scalars, shows them as a bar chart.
    If heatmaps are arrays, shows them as time series plots.

    Args:
        heatmaps: List of heatmaps for each GRU unit.
        cols: Number of columns in the grid layout. Default is 8.
    """

    # Check if heatmaps are scalars or arrays
    if isinstance(heatmaps[0], (int, float, np.float32, np.float64)):
        # Heatmaps are scalars - display as a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(heatmaps)), heatmaps)
        plt.title('GRU Unit Importance Scores')
        plt.xlabel('GRU Unit')
        plt.ylabel('Importance Score')
        plt.grid(True, alpha=0.3)
        plt.show()
        return

    # Heatmaps are arrays - display as time series plots
    rows = (len(heatmaps) + cols - 1) // cols
    plt.figure(figsize=(3*cols, 2*rows))

    for i, heatmap in enumerate(heatmaps):
        plt.subplot(rows, cols, i + 1)
        plt.plot(heatmap)
        plt.title(f'GRU Unit {i}')
        plt.xlabel('Timestep')
        plt.ylabel('Activation')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def display_heatmaps_on_ecg(ecg_sample, heatmaps, selected_units=None, alpha=0.8, threshold=0.3):
    """
    Displays the GRU heatmaps overlaid on the ECG signal.

    Args:
        ecg_sample: A single ECG sample of shape (2800, 12).
        heatmaps: List of heatmaps for each GRU unit.
        selected_units: List of indices of GRU units to display. If None, selects the first 4 units with highest activation.
        alpha: Transparency level for the heatmap overlay. Default is 0.8.
        threshold: Minimum activation value to display in the heatmap. Default is 0.3.
    """
    # Check if heatmaps are scalars or arrays
    if isinstance(heatmaps[0], (int, float, np.float32, np.float64)):
        # Display GRU unit importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(heatmaps)), heatmaps)
        plt.title('GRU Unit Importance')
        plt.xlabel('GRU Unit')
        plt.ylabel('Importance Score')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Also display the original ECG signal
        plt.figure(figsize=(15, 8))
        for i in range(min(12, ecg_sample.shape[1])):
            plt.subplot(3, 4, i+1)
            plt.plot(ecg_sample[:, i])
            plt.title(f'Lead {i+1}')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        raise Warning("Heatmaps are scalars (GRU returns only last output). \
                      Cannot display temporal evolution. Showing unit importance instead.")

    if selected_units is None:
        # Select the top 4 most active units
        activations = [np.sum(hm) for hm in heatmaps]
        selected_units = np.argsort(activations)[-4:]

    _, axes = plt.subplots(len(selected_units), 1, figsize=(15, 3*len(selected_units)))
    if len(selected_units) == 1:
        axes = [axes]

    for idx, unit in enumerate(selected_units):
        # Resize the heatmap to match the length of the ECG signal
        heatmap_resized = np.interp(
            np.linspace(0, len(heatmaps[unit]), len(ecg_sample)),
            np.arange(len(heatmaps[unit])),
            heatmaps[unit]
        )

        # Apply a threshold to highlight only significant activations
        heatmap_thresholded = np.where(heatmap_resized > threshold, heatmap_resized, 0)

        # Plot the mean ECG signal
        ecg_mean = np.mean(ecg_sample, axis=1)
        axes[idx].plot(ecg_mean, 'b-', alpha=0.7, label='ECG Signal', linewidth=1.5)

        # Create a colormap for activations
        norm_heatmap = heatmap_thresholded / (np.max(heatmap_thresholded) + 1e-8)

        # Overlay the heatmap with more intense colors
        for i in range(len(ecg_mean)):
            if norm_heatmap[i] > 0:
                # More intense color for strong activations
                color_intensity = norm_heatmap[i]
                axes[idx].axvspan(i-0.5, i+0.5,
                                  alpha=alpha * color_intensity,
                                  color='red',
                                  label='High Activation' if i == np.argmax(norm_heatmap) else "")

        # Add vertical lines for activation peaks
        peaks = np.where(norm_heatmap > 0.7)[0]  # Find peaks above 0.7
        for peak in peaks:
            axes[idx].axvline(x=peak, color='orange', linestyle='--', alpha=0.8, linewidth=2)

        axes[idx].set_title(f'ECG with GRU Unit {unit} Activation (Activation Sum: {np.sum(heatmaps[unit]):.3f})')
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_temporal_attention(heatmaps, window_size=100):
    """
    Analyzes the temporal attention patterns from GRU heatmaps and displays them as a heatmap.

    Args:
        heatmaps: List of heatmaps for each GRU unit, where each heatmap is a list of attention scores over time.
        window_size: Size of the temporal window to average over. Default is 100.
    
    Returns:
        attention_matrix: 2D numpy array of shape (num_units, num_windows) representing the average attention scores.
    """
    # Check if heatmaps are scalars
    if isinstance(heatmaps[0], (int, float, np.float32, np.float64)):
        print("WARNING: Cannot analyze temporal attention with scalar heatmaps")
        print("GRU layer returns only last output, not full sequence")
        return np.array(heatmaps).reshape(1, -1)

    # Verify that there are enough timesteps
    if len(heatmaps[0]) < window_size:
        print(f"WARNING: Sequence length ({len(heatmaps[0])}) is shorter than window_size ({window_size})")
        window_size = max(1, len(heatmaps[0]) // 10)  # Use at least 10 windows
        print(f"Using adjusted window_size: {window_size}")

    # Compute average attention over temporal windows
    n_windows = max(1, len(heatmaps[0]) // window_size)
    attention_matrix = np.zeros((len(heatmaps), n_windows))

    for i, heatmap in enumerate(heatmaps):
        for j in range(n_windows):
            start_idx = j * window_size
            end_idx = min((j + 1) * window_size, len(heatmap))
            if end_idx > start_idx:  # Ensure the window is not empty
                attention_matrix[i, j] = np.mean(heatmap[start_idx:end_idx])

    # Check if the matrix has sufficient variance
    if np.var(attention_matrix) < 1e-10:
        print("WARNING: Attention matrix has very low variance")
        # Add a small amount of noise to avoid visualization issues
        attention_matrix += np.random.normal(0, 1e-6, attention_matrix.shape)

    plt.figure(figsize=(12, 8))

    # Use vmin and vmax to handle cases with identical values
    vmin = np.min(attention_matrix)
    vmax = np.max(attention_matrix)
    if vmax == vmin:
        vmax = vmin + 1e-6

    im = plt.imshow(attention_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Attention Score')
    plt.xlabel('Time Windows')
    plt.ylabel('GRU Units')
    plt.title(f'Temporal Attention Pattern (Window size: {window_size})')

    # Add matrix information
    plt.figtext(0.02, 0.02, f'Matrix shape: {attention_matrix.shape}, Variance: {np.var(attention_matrix):.6f}',
                fontsize=8, ha='left')

    plt.show()

    return attention_matrix


def make_all_gradcam_heatmaps(ecg_sample, model, last_conv_layer_name="conv2d_1", pred_index=None):
    """
    Generates Grad-CAM heatmaps for the last convolutional layer in a Keras model.
    Args:
        ecg_sample: A single ECG sample of shape (2800, 12).
        model: The trained Keras model containing the convolutional layer.
        last_conv_layer_name: Name of the last convolutional layer in the model. Default is "conv2d_1".
        pred_index: Index of the class for which to compute the heatmap. \
            If None, it uses the class with the highest predicted probability.
    
    Returns:
        heatmaps: List of heatmaps for each channel in the last convolutional layer.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    ecg_input = tf.expand_dims(ecg_sample, axis=0)  # (1, 2800, 12)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(ecg_input)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    conv_outputs = conv_outputs[0]  # (H, W, C)
    grads = grads[0]                # (H, W, C)

    # Compute all heatmaps, one for each channel
    heatmaps = []
    for i in range(conv_outputs.shape[-1]):
        weighted_map = conv_outputs[..., i] * grads[..., i]
        heatmap = tf.maximum(weighted_map, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmaps.append(heatmap.numpy())

    return heatmaps  # List of heatmaps (H, W) for each channel


def get_spectrogram(model, sample, lead=0):
    """Extracts the spectrogram output from the LogSpectrogram layer of the model."""
    spectrogram_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=1).output)
    sample = tf.expand_dims(sample, axis=0)  # (1, 2800, 12)
    spectrogram = spectrogram_model(sample)  # (1, 88, 64, 12)
    return spectrogram[lead]  # (88, 64, 12)

def display_all_heatmaps(heatmaps, cols=8):
    """Displays all heatmaps in a grid layout."""
    rows = (len(heatmaps) + cols - 1) // cols
    plt.figure(figsize=(2*cols, 2*rows))
    for i, heatmap in enumerate(heatmaps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(heatmap, cmap='jet')
        plt.title(f'Channel {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_all_heatmaps_on_spectrogram(spectrogram, heatmaps, cols=8, alpha=0.4):
    """
    Displays multiple heatmaps overlaid on a spectrogram in a grid layout.

    Args:
        spectrogram (tf.Tensor): The input spectrogram tensor of shape (H, W, C), 
                                 where H is the height, W is the width, and C is the number of channels.
        heatmaps (list of tf.Tensor): A list of heatmap tensors to overlay on the spectrogram. 
                                      Each heatmap should have a shape compatible with resizing to (H, W).
        cols (int, optional): The number of columns in the grid layout. Defaults to 8.
        alpha (float, optional): The transparency level for the heatmap overlay. 
                                 A value between 0 (completely transparent) and 1 (completely opaque). Defaults to 0.4.
    """
    rows = (len(heatmaps) + cols - 1) // cols
    plt.figure(figsize=(2.5*cols, 2.5*rows))

    for i, heatmap in enumerate(heatmaps):
        spectrogram_resized = tf.image.resize(spectrogram, heatmap.shape[:2]).numpy()
        spectrogram_gray = tf.reduce_mean(spectrogram_resized, axis=-1)  # Convert to grayscale

        plt.subplot(rows, cols, i + 1)
        plt.imshow(spectrogram_gray, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=alpha)
        plt.title(f'Channel {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
