import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
from io import StringIO
import contextlib
from .utils import compute_predictions


def plot_full_report_and_metrics(model, X, y, history, threshold, class_names=["Neg", "Pos"], return_pillow=False):
    """
    Display a comprehensive report of the model's performance including:
    - Confusion Matrix
    - Classification Report
    - Model Summary
    - Training History (Accuracy, AUC, Loss)

    Args:
        model: Keras model to evaluate.
        X: Input features for prediction.
        y: True labels for the input features.
        history: Keras History object containing training metrics.
        threshold: Threshold for binary classification.
        class_names: List of class names for the confusion matrix.
        return_pillow: If True, returns a PIL Image instead of displaying the plot.

    Returns:
        If return_pillow is True, returns a PIL Image of the plot.
        Otherwise, displays the plot.
    """
    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs > threshold).astype("int32")

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=False)
    acc = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

    acc_hist = history.history.get('accuracy')
    val_acc_hist = history.history.get('val_accuracy')
    auc_hist = history.history.get('auc')
    val_auc_hist = history.history.get('val_auc')
    loss_hist = history.history.get('loss')
    val_loss_hist = history.history.get('val_loss')
    epochs = range(1, len(acc_hist or loss_hist) + 1)

    fig = plt.figure(figsize=(12, 10))
    outer_gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

    # --- [0, 0]: Confusion Matrix + Classification Report ---
    cmrep_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, 0], height_ratios=[.75, .75], hspace=0)

    # Classification Report
    ax0 = fig.add_subplot(cmrep_gs[0, 0])
    ax0.axis("off")
    ax0.text(0, 1.0, "Classification Report", fontsize=12, fontweight='bold')
    ax0.text(0, 0.9, report, family='monospace', fontsize=9, va='top')
    ax0.text(0, 0.3, f"Accuracy: {acc:.3f}   Threshold: {threshold:.2f}", fontsize=10)

    # Confusion Matrix
    ax2 = fig.add_subplot(cmrep_gs[1, 0])
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    # --- [0, 1]: Model Summary ---
    ax1 = fig.add_subplot(outer_gs[0, 1])
    ax1.axis("off")
    with StringIO() as buf, contextlib.redirect_stdout(buf):
        model.summary(print_fn=lambda x: print(x, file=buf))
        summary_str = buf.getvalue()
    ax1.text(0, 1.0, "Model Summary", fontsize=12, fontweight='bold')
    ax1.text(0, 0.95, summary_str, family='monospace', fontsize=8, va='top')

    # --- [1, 0]: History Plots (occupano tutta la riga) ---
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1, :], wspace=0.3)

    # Plot Accuracy
    ax_acc = fig.add_subplot(inner_gs[0, 0])
    if acc_hist and val_acc_hist:
        ax_acc.plot(epochs, acc_hist, 'bo-', label='Train Acc')
        ax_acc.plot(epochs, val_acc_hist, 'ro-', label='Val Acc')
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Acc")
    ax_acc.legend()
    ax_acc.grid(True)

    # Plot AUC
    ax_auc = fig.add_subplot(inner_gs[0, 1])
    if auc_hist and val_auc_hist:
        ax_auc.plot(epochs, auc_hist, 'bo-', label='Train AUC')
        ax_auc.plot(epochs, val_auc_hist, 'ro-', label='Val AUC')
    ax_auc.set_title("AUC")
    ax_auc.set_xlabel("Epochs")
    ax_auc.set_ylabel("AUC")
    ax_auc.legend()
    ax_auc.grid(True)

    # Plot Loss
    ax_loss = fig.add_subplot(inner_gs[0, 2])
    if loss_hist and val_loss_hist:
        ax_loss.plot(epochs, loss_hist, 'bo-', label='Train Loss')
        ax_loss.plot(epochs, val_loss_hist, 'ro-', label='Val Loss')
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    plt.tight_layout()
    
    if return_pillow:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    else:
        plt.show()
        return None


def plot_training_metrics(history):
    """
    Display training metrics from the Keras History object. Incudes plots for:
    - Accuracy
    - AUC
    - Loss

    Args:
        history: Keras History object containing training metrics.

    Returns:
        None. Displays the plots directly.
    """
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    auc = history.history.get('auc')
    val_auc = history.history.get('val_auc')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1) if acc else range(1, len(loss) + 1)

    plt.figure(figsize=(18, 5))

    # Plot Accuracy
    if acc and val_acc:
        plt.subplot(1, 3, 1)
        plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    # Plot AUC
    if auc and val_auc:
        plt.subplot(1, 3, 2)
        plt.plot(epochs, auc, 'bo-', label='Training AUC')
        plt.plot(epochs, val_auc, 'ro-', label='Validation AUC')
        plt.title('AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

    # Plot Loss
    if loss and val_loss:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_correct_incorrect_histogram(ax, y_pred, correct, bins, threshold):
    """
    Plots a histogram showing the distribution of correct and incorrect predictions.

    Args:
        ax: Matplotlib axis to plot on.
        y_pred: Array of model predictions.
        correct: Boolean array indicating whether predictions are correct.
        bins: Array of bin edges for the histogram.
        threshold: Threshold value for classification.
    """
    ax.hist(y_pred[correct], bins=bins, color='blue', alpha=0.6, label='Correct')
    ax.hist(y_pred[~correct], bins=bins, color='red', alpha=0.6, label='Incorrect')
    ax.axvline(threshold, color='gray', linestyle='--', label=f'Threshold {threshold:.2f}')
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Predictions - Correct vs Incorrect")
    ax.legend()
    ax.grid(True)
    

def plot_grouped_histogram(ax, bins, counts_by_group, color_map, label_format=str):
    """
    Plots a grouped histogram with stacked bars for each group.

    Args:
        ax: Matplotlib axis to plot on.
        bins: Array of bin edges for the histogram.
        counts_by_group: Dictionary where keys are group labels and values are counts in each bin.
        color_map: Dictionary mapping group labels to colors.
        label_format: Function to format the group labels for the legend.
    """
    bottom = np.zeros(len(bins))
    for label in counts_by_group:
        counts = counts_by_group[label]
        color = color_map[label]
        ax.bar(bins, counts, width=(bins[1]-bins[0]), bottom=bottom,
               label=label_format(label), color=color, align='center',
               edgecolor='white', linewidth=0.5)
        bottom += counts
    ax.set_ylabel("Frequenza per gruppo")
    ax.set_title("Distribuzione dei gruppi per valore di output")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)


def bin_age_groups(ages, step=5, max_age=100):
    """Bins ages into groups of a specified step size."""
    bins = np.arange(0, max_age + step, step)
    labels = [f"{i}-{i+step-1}" for i in bins[:-1]]
    bin_indices = np.digitize(ages, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(labels)-1)
    return bin_indices, labels

def plot_age_distribution(ax_correct, ax_wrong, y_pred, correct, ages, bins_output, 
                         age_step=5, normalize=True):
    """
    Creates a stacked bar chart of age distribution for correct/incorrect predictions.

    Args:
        ax_correct: Matplotlib axis for correct predictions.
        ax_wrong: Matplotlib axis for incorrect predictions.
        y_pred: Array of model predictions.
        correct: Boolean array indicating whether predictions are correct.
        ages: Array of ages corresponding to the predictions.
        bins_output: Array of bin edges for the output values.
        age_step: Step size for age groups (default is 5).
        normalize: If True, normalizes the counts to percentages.
    """
    max_age = max(ages) if ages else 100
    age_bin_indices, age_labels = bin_age_groups(ages, step=age_step, max_age=max_age)
    n_age_bins = len(age_labels)
    
    # Prepare bins for output values [n_output_bins x n_age_bins]
    bin_indices = np.digitize(y_pred, bins_output) - 1
    age_distribution_correct = np.zeros((len(bins_output), n_age_bins))
    age_distribution_wrong = np.zeros((len(bins_output), n_age_bins))

    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(bins_output):
            age_bin = age_bin_indices[i]
            if correct[i]:
                age_distribution_correct[bin_idx, age_bin] += 1
            else:
                age_distribution_wrong[bin_idx, age_bin] += 1

    # If required, display the plot as percentage
    if normalize:
        def normalize_rows(mat):
            row_sums = mat.sum(axis=1, keepdims=True)
            return np.divide(mat, row_sums, where=row_sums != 0) * 100
        
        age_distribution_correct = normalize_rows(age_distribution_correct)
        age_distribution_wrong = normalize_rows(age_distribution_wrong)

    # Color gradient for age bins
    cmap = plt.colormaps.get_cmap('viridis').resampled(n_age_bins)
    colors = [cmap(i) for i in range(n_age_bins)]

    def plot_stack(ax, matrix, title):
        bottom = np.zeros(len(bins_output))
        bars = []
        
        # Compute the width of the bars
        if len(bins_output) > 1:
            bar_width = bins_output[1] - bins_output[0]  # Full width
        else:
            bar_width = 1.0
        
        for age_idx in range(n_age_bins):
            bar = ax.bar(bins_output, matrix[:, age_idx], bottom=bottom,
                        width=bar_width, color=colors[age_idx],
                        edgecolor='none', align='center', 
                        label=age_labels[age_idx])
            bars.append(bar)
            bottom += matrix[:, age_idx]
        
        # Set the y-axis limits and labels
        if normalize:
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percentage (%)")
        else:
            ax.set_ylim(0, np.max(bottom) * 1.1)
            ax.set_ylabel("Count")
        
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        
        # Set the x-axis limits and ticks
        ax.set_xlim(bins_output[0] - bar_width/2, bins_output[-1] + bar_width/2)
        
        if len(bins_output) <= 10:
            ax.set_xticks(bins_output)
        else:
            # If there are many bins, reduce the number of ticks
            step = max(1, len(bins_output) // 10)
            ax.set_xticks(bins_output[::step])
        
        return bars

    # Plot the stacked bar charts for correct and wrong predictions
    bars_correct = plot_stack(ax_correct, age_distribution_correct, 
                             "Age Distribution - Correct Predictions")
    bars_wrong = plot_stack(ax_wrong, age_distribution_wrong, 
                           "Age Distribution - Incorrect Predictions")
    
    # Add legends
    ax_correct.legend(bbox_to_anchor=(1.00, 1), loc='upper left', 
                     title="Age Groups", fontsize=9, title_fontsize=10)
    
    return bars_correct, bars_wrong


def plot_age_histogram_stacked(ax, correct, ages, age_bins=None, age_step=5):
    """
    Create a stacked histogram of age distribution for correct and incorrect predictions.

    Args:
        ax: Matplotlib axis to plot on.
        correct: Boolean array indicating whether predictions are correct.
        ages: Array of ages corresponding to the predictions.
        age_bins: Array of bin edges for the age groups. If None, uses age_step.
        age_step: Step size for age groups if age_bins is None (default is 5).
    """
    
    # Create bins for age if not provided
    if age_bins is None:
        min_age = min(ages) if ages else 0
        max_age = max(ages) if ages else 100
        age_bins = np.arange(min_age, max_age + age_step, age_step)
    
    # Separate ages for correct and incorrect predictions
    ages_correct = [ages[i] for i in range(len(ages)) if correct[i]]
    ages_wrong = [ages[i] for i in range(len(ages)) if not correct[i]]
    
    # Create the stacked histogram
    ax.hist([ages_wrong, ages_correct], bins=age_bins, 
            color=['red', 'green'], alpha=0.7,
            label=['Incorrect Predictions', 'Correct Predictions'],
            edgecolor='white', linewidth=0.5, stacked=True)
    
    # Formatting
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Number of Predictions', fontsize=12)
    ax.set_title('Age Distribution of Predictions (Stacked)', fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Improve x-axis ticks
    if len(age_bins) <= 15:
        ax.set_xticks(age_bins)
    else:
        step = max(1, len(age_bins) // 10)
        ax.set_xticks(age_bins[::step])
    
    return ax


def plot_age_histogram_percentage(ax, correct, ages, age_bins=None, age_step=5):
    """
    Create a histogram showing the percentage of correct predictions by age group.

    Args:
        ax: Matplotlib axis to plot on.
        correct: Boolean array indicating whether predictions are correct.
        ages: Array of ages corresponding to the predictions.
        age_bins: Array of bin edges for the age groups. If None, uses age_step.
        age_step: Step size for age groups if age_bins is None (default is 5).
    """
    
    # Create bins for age if not provided
    if age_bins is None:
        min_age = min(ages) if ages else 0
        max_age = max(ages) if ages else 100
        age_bins = np.arange(min_age, max_age + age_step, age_step)
    
    # Calculate the percentages for each age bin
    bin_centers = []
    success_rates = []
    total_counts = []
    
    for i in range(len(age_bins) - 1):
        # Find the indices of people in this age bin
        in_bin = [(ages[j] >= age_bins[i] and ages[j] < age_bins[i+1]) 
                  for j in range(len(ages))]
        
        if any(in_bin):
            correct_in_bin = [correct[j] for j in range(len(correct)) if in_bin[j]]
            success_rate = np.mean(correct_in_bin) * 100
            total_count = len(correct_in_bin)
            
            bin_centers.append((age_bins[i] + age_bins[i+1]) / 2)
            success_rates.append(success_rate)
            total_counts.append(total_count)
    
    # If there is no data, exit
    if not bin_centers:
        ax.text(0.5, 0.5, 'Nessun dato disponibile', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Create the bar chart
    if len(bin_centers) > 1:
        bar_width = (bin_centers[1] - bin_centers[0]) * 0.8
    else:
        bar_width = age_step * 0.8
        
    bars = ax.bar(bin_centers, success_rates, width=bar_width, 
                  color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Add labels with the total number of samples
    for bar, count in zip(bars, total_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Età', fontsize=12)
    ax.set_ylabel('Percentuale di predizioni corrette (%)', fontsize=12)
    ax.set_title('Tasso di successo delle predizioni per età', fontsize=14, pad=15)
    ax.set_ylim(0, 105)  # Piccolo margine sopra il 100%
    ax.grid(True, alpha=0.3)
    
    # Improve the x-axis ticks - use only the bin centers that have data
    if len(bin_centers) > 0:
        ax.set_xticks(bin_centers)
        # Create labels only for bins with data
        labels = []
        for center in bin_centers:
            # Trova l'indice del bin corrispondente
            bin_idx = None
            for i in range(len(age_bins) - 1):
                if abs(center - (age_bins[i] + age_bins[i+1])/2) < 0.1:
                    bin_idx = i
                    break
            if bin_idx is not None:
                labels.append(f'{int(age_bins[bin_idx])}-{int(age_bins[bin_idx+1])}')
            else:
                labels.append(f'{int(center)}')
        
        ax.set_xticklabels(labels, rotation=45)
    
    return ax


def extract_labels(y_test_info, *label_keys):
    """Extracts tuples of labels from the y_test_info dictionaries."""
    return [tuple(info[key] for key in label_keys) for info in y_test_info]

def build_grouped_counts(y_pred, bins, label_tuples):
    """ Creates a dictionary of counts for each label group in the specified bins. """
    bin_indices = np.digitize(y_pred, bins) - 1
    unique_labels = sorted(set(label_tuples))
    counts_by_group = {label: np.zeros(len(bins)) for label in unique_labels}
    
    for idx, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(bins):
            counts_by_group[label_tuples[idx]][bin_idx] += 1
            
    return counts_by_group, unique_labels

def get_color_map(unique_labels, base_key_index=1):
    """Returns a color map for the labels, using a base color for a primary key."""
    # Extract base keys (e.g., source)
    base_keys = sorted(set(label[base_key_index] for label in unique_labels))
    base_colors = plt.cm.Set3(np.linspace(0, 1, len(base_keys)))
    base_color_map = {key: base_colors[i] for i, key in enumerate(base_keys)}

    # Final colors for the complete labels
    color_map = {}
    for label in unique_labels:
        base_color = base_color_map[label[base_key_index]]
        alpha = 1.0 if label[0] == 1 else 0.6  # e.g., chagas = 1 -> opaque
        color_map[label] = mcolors.to_rgba(base_color, alpha=alpha)
    return color_map

def plot_model_analysis(
    model,
    X_test,
    y_test,
    y_test_info,
    threshold,
    bins=np.linspace(0, 1, 21),
    label_keys=["Chagas", "Source"],
    label_format=lambda l: f"Chagas={l[0]}, {l[1]}",
    return_pillow=False
):
    """
    Analyzes the model's predictions and visualizes various metrics and distributions.

    Args:
        model: Trained model to evaluate.
        X_test: Test input data.
        y_test: True labels for the test data.
        y_test_info: Additional information about the test samples (e.g., age, source).
        threshold: Threshold for binary classification.
        bins: Array of bin edges for histograms.
        label_keys: Keys to extract labels from y_test_info for grouping.
        label_format: Function to format labels for the legend.
        return_pillow: If True, returns a PIL Image of the plot; otherwise, displays the plot.

    Returns:
        If return_pillow is True, returns a PIL Image of the plot. Otherwise, displays the plot.
    """
    y_pred, _, correct = compute_predictions(model, X_test, y_test, threshold)

    for i, info in enumerate(y_test_info):
        info['Chagas'] = int(y_test[i])

    label_tuples = extract_labels(y_test_info, *label_keys)
    counts_by_group, unique_labels = build_grouped_counts(y_pred, bins, label_tuples)
    color_map = get_color_map(unique_labels, base_key_index=1 if len(label_keys) > 1 else 0)
    ages = [info['Age'] for info in y_test_info]

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.5, 2, 1.7])

    ax1 = fig.add_subplot(gs[0])
    plot_correct_incorrect_histogram(ax1, y_pred, correct, bins, threshold)

    ax2 = fig.add_subplot(gs[1])
    plot_grouped_histogram(ax2, bins, counts_by_group, color_map, label_format)

    gs_age = gs[2].subgridspec(2, 1, hspace=0.5)
    ax3 = fig.add_subplot(gs_age[0])
    ax4 = fig.add_subplot(gs_age[1], sharex=ax3)
    plot_age_distribution(ax3, ax4, y_pred, correct, ages, bins, normalize=False)
    ax4.set_xlabel("Model Output Value")

    ax5 = fig.add_subplot(gs[3])
    plot_age_histogram_percentage(ax5, y_pred, correct, ages, age_bins=bins, age_step=2)

    plt.tight_layout()

    if return_pillow:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    else:
        plt.show()
        return None
    

def plot_bento_analysis(model, X_test, y_test, y_test_info, history, threshold):
    """
    Combines the full report and model analysis into a single image for easy comparison.
    
    Args:
        model: Trained model to evaluate.
        X_test: Test input data.
        y_test: True labels for the test data.
        y_test_info: Additional information about the test samples (e.g., age, source).
        history: Keras History object containing training metrics.
        threshold: Threshold for binary classification.

    Returns:
        A PIL Image containing the combined analysis.

    Example:
        >>> img = plot_bento_analysis(model, X_test, y_test, y_test_info, history, threshold)
        >>> display(img)
    """
    img1 = plot_full_report_and_metrics(model, X_test, y_test, history, threshold, return_pillow=True)
    img2 = plot_model_analysis(model, X_test, y_test, y_test_info, threshold, return_pillow=True)

    # 1. Resize img2 to match the height of img1, maintaining proportions
    new_height = int(img1.height * 1.2)
    new_width = int(img2.width * (new_height / img2.height))
    img2_resized = img2.resize((new_width, new_height), Image.LANCZOS)

    # 2. Create the new combined image
    total_width = img1.width + img2_resized.width
    max_height = max(img1.height, img2_resized.height)
    new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    # 3. Center img1 vertically (if img1 is shorter than max_height)
    y_offset_img1 = (max_height - img1.height) // 2
    y_offset_img2 = (max_height - img2_resized.height) // 2

    new_img.paste(img1, (0, y_offset_img1))
    new_img.paste(img2_resized, (img1.width, y_offset_img2))

    # 4. Return the combined image
    return new_img