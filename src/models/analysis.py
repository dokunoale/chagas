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
from .utils import find_optimal_threshold

def plot_full_report_and_metrics(model, X, y, history, threshold, class_names=["Neg", "Pos"], return_pillow=False):
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


def compute_predictions(model, X_test, y_test, threshold):
    y_pred = model.predict(X_test).flatten()
    y_pred_class = (y_pred >= threshold).astype(int)
    correct = (y_pred_class == y_test)
    return y_pred, y_pred_class, correct

def extract_labels(y_test_info, *label_keys):
    """Estrae tuple di etichette dai dizionari y_test_info."""
    return [tuple(info[key] for key in label_keys) for info in y_test_info]

def plot_correct_incorrect_histogram(ax, y_pred, correct, bins, threshold):
    ax.hist(y_pred[correct], bins=bins, color='blue', alpha=0.6, label='Corrette')
    ax.hist(y_pred[~correct], bins=bins, color='red', alpha=0.6, label='Sbagliate')
    ax.axvline(threshold, color='gray', linestyle='--', label=f'Soglia {threshold:.2f}')
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione delle predizioni - corrette vs sbagliate")
    ax.legend()
    ax.grid(True)

def build_grouped_counts(y_pred, bins, label_tuples):
    bin_indices = np.digitize(y_pred, bins) - 1
    unique_labels = sorted(set(label_tuples))
    counts_by_group = {label: np.zeros(len(bins)) for label in unique_labels}
    
    for idx, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(bins):
            counts_by_group[label_tuples[idx]][bin_idx] += 1
            
    return counts_by_group, unique_labels

def get_color_map(unique_labels, base_key_index=1):
    """Restituisce una mappa di colori per le label, usando un colore base per una chiave principale."""
    # Estrai chiavi base (es. source)
    base_keys = sorted(set(label[base_key_index] for label in unique_labels))
    base_colors = plt.cm.Set3(np.linspace(0, 1, len(base_keys)))
    base_color_map = {key: base_colors[i] for i, key in enumerate(base_keys)}

    # Colori finali per le label complete
    color_map = {}
    for label in unique_labels:
        base_color = base_color_map[label[base_key_index]]
        alpha = 1.0 if label[0] == 1 else 0.6  # es. chagas = 1 -> opaco
        color_map[label] = mcolors.to_rgba(base_color, alpha=alpha)
    return color_map

def plot_grouped_histogram(ax, bins, counts_by_group, color_map, label_format=str):
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

# Funzione helper: assegna età ai bin
def bin_age_groups(ages, step=5, max_age=100):
    bins = np.arange(0, max_age + step, step)
    labels = [f"{i}-{i+step-1}" for i in bins[:-1]]
    bin_indices = np.digitize(ages, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(labels)-1)
    return bin_indices, labels

# Plot dell'età
def plot_age_distribution(ax_correct, ax_wrong, y_pred, correct, ages, bins_output, 
                         age_step=5, normalize=True):
    """
    Crea un grafico a barre impilate della distribuzione dell'età per predizioni corrette/sbagliate
    
    Parameters:
    -----------
    ax_correct, ax_wrong : matplotlib axes
        Gli assi su cui plottare
    y_pred : array-like
        Predizioni del modello
    correct : array-like
        Array booleano che indica se le predizioni sono corrette
    ages : array-like
        Età dei partecipanti
    bins_output : array-like
        Bin per l'output del modello
    age_step : int, default=5
        Passo per i gruppi di età
    normalize : bool, default=True
        Se True, normalizza le distribuzioni per riga (percentuali)
        Se False, mostra i conteggi assoluti
    """
    max_age = max(ages) if ages else 100
    age_bin_indices, age_labels = bin_age_groups(ages, step=age_step, max_age=max_age)
    n_age_bins = len(age_labels)
    
    # Prepara matrici [n_output_bins x n_age_bins]
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

    # Normalizza per riga (somma a 100%) se richiesto
    if normalize:
        def normalize_rows(mat):
            row_sums = mat.sum(axis=1, keepdims=True)
            return np.divide(mat, row_sums, where=row_sums != 0) * 100
        
        age_distribution_correct = normalize_rows(age_distribution_correct)
        age_distribution_wrong = normalize_rows(age_distribution_wrong)

    # Gradiente di colori per età
    cmap = plt.colormaps.get_cmap('viridis').resampled(n_age_bins)
    colors = [cmap(i) for i in range(n_age_bins)]

    def plot_stack(ax, matrix, title):
        bottom = np.zeros(len(bins_output))
        bars = []
        
        # Calcola la larghezza delle barre - riempi completamente lo spazio
        if len(bins_output) > 1:
            bar_width = bins_output[1] - bins_output[0]  # Larghezza completa
        else:
            bar_width = 1.0
        
        for age_idx in range(n_age_bins):
            bar = ax.bar(bins_output, matrix[:, age_idx], bottom=bottom,
                        width=bar_width, color=colors[age_idx],
                        edgecolor='none', align='center', 
                        label=age_labels[age_idx])
            bars.append(bar)
            bottom += matrix[:, age_idx]
        
        # Imposta i limiti e le etichette
        if normalize:
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percentuale (%)")
        else:
            ax.set_ylim(0, np.max(bottom) * 1.1)  # Aggiungi un po' di spazio sopra
            ax.set_ylabel("Conteggio")
        
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        
        # Imposta i limiti dell'asse x per valori continui
        ax.set_xlim(bins_output[0] - bar_width/2, bins_output[-1] + bar_width/2)
        
        # Imposta i tick dell'asse x
        if len(bins_output) <= 10:
            ax.set_xticks(bins_output)
        else:
            # Se ci sono troppi bin, mostra solo alcuni tick
            step = max(1, len(bins_output) // 10)
            ax.set_xticks(bins_output[::step])
        
        return bars

    # Crea i grafici
    bars_correct = plot_stack(ax_correct, age_distribution_correct, 
                             "Distribuzione età - predizioni corrette")
    bars_wrong = plot_stack(ax_wrong, age_distribution_wrong, 
                           "Distribuzione età - predizioni sbagliate")
    
    # Aggiungi la legenda al grafico superiore, posizionata meglio
    ax_correct.legend(bbox_to_anchor=(1.00, 1), loc='upper left', 
                     title="Fasce d'età", fontsize=9, title_fontsize=10)
    
    return bars_correct, bars_wrong

def plot_age_histogram_stacked(ax, y_pred, correct, ages, age_bins=None, age_step=5):
    """
    Crea un istogramma impilato con l'età sull'asse X
    
    Parameters:
    -----------
    ax : matplotlib axis
        L'asse su cui plottare
    y_pred : array-like
        Predizioni del modello
    correct : array-like
        Array booleano che indica se le predizioni sono corrette
    ages : array-like
        Età dei partecipanti
    age_bins : array-like, optional
        Bin personalizzati per l'età. Se None, usa age_step
    age_step : int, default=5
        Passo per i gruppi di età se age_bins è None
    """
    
    # Crea i bin per l'età se non forniti
    if age_bins is None:
        min_age = min(ages) if ages else 0
        max_age = max(ages) if ages else 100
        age_bins = np.arange(min_age, max_age + age_step, age_step)
    
    # Separa le età per predizioni corrette e sbagliate
    ages_correct = [ages[i] for i in range(len(ages)) if correct[i]]
    ages_wrong = [ages[i] for i in range(len(ages)) if not correct[i]]
    
    # Crea l'istogramma impilato
    ax.hist([ages_wrong, ages_correct], bins=age_bins, 
            color=['red', 'green'], alpha=0.7,
            label=['Predizioni sbagliate', 'Predizioni corrette'],
            edgecolor='white', linewidth=0.5, stacked=True)
    
    # Formattazione
    ax.set_xlabel('Età', fontsize=12)
    ax.set_ylabel('Numero di predizioni', fontsize=12)
    ax.set_title('Distribuzione delle predizioni per età (impilato)', fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Migliora i tick dell'asse x
    if len(age_bins) <= 15:
        ax.set_xticks(age_bins)
    else:
        step = max(1, len(age_bins) // 10)
        ax.set_xticks(age_bins[::step])
    
    return ax

# Versione con percentuali
def plot_age_histogram_percentage(ax, y_pred, correct, ages, age_bins=None, age_step=5):
    """
    Crea un istogramma con percentuali di successo per età
    
    Parameters:
    -----------
    ax : matplotlib axis
        L'asse su cui plottare
    y_pred : array-like
        Predizioni del modello
    correct : array-like
        Array booleano che indica se le predizioni sono corrette
    ages : array-like
        Età dei partecipanti
    age_bins : array-like, optional
        Bin personalizzati per l'età. Se None, usa age_step
    age_step : int, default=5
        Passo per i gruppi di età se age_bins è None
    """
    
    # Crea i bin per l'età se non forniti
    if age_bins is None:
        min_age = min(ages) if ages else 0
        max_age = max(ages) if ages else 100
        age_bins = np.arange(min_age, max_age + age_step, age_step)
    
    # Calcola le percentuali per ogni bin di età
    bin_centers = []
    success_rates = []
    total_counts = []
    
    for i in range(len(age_bins) - 1):
        # Trova gli indici delle persone in questo bin di età
        in_bin = [(ages[j] >= age_bins[i] and ages[j] < age_bins[i+1]) 
                  for j in range(len(ages))]
        
        if any(in_bin):
            correct_in_bin = [correct[j] for j in range(len(correct)) if in_bin[j]]
            success_rate = np.mean(correct_in_bin) * 100
            total_count = len(correct_in_bin)
            
            bin_centers.append((age_bins[i] + age_bins[i+1]) / 2)
            success_rates.append(success_rate)
            total_counts.append(total_count)
    
    # Se non ci sono dati, esci
    if not bin_centers:
        ax.text(0.5, 0.5, 'Nessun dato disponibile', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Crea il grafico a barre
    if len(bin_centers) > 1:
        bar_width = (bin_centers[1] - bin_centers[0]) * 0.8
    else:
        bar_width = age_step * 0.8
        
    bars = ax.bar(bin_centers, success_rates, width=bar_width, 
                  color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Aggiungi etichette con il numero totale di campioni
    for bar, count in zip(bars, total_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # Formattazione
    ax.set_xlabel('Età', fontsize=12)
    ax.set_ylabel('Percentuale di predizioni corrette (%)', fontsize=12)
    ax.set_title('Tasso di successo delle predizioni per età', fontsize=14, pad=15)
    ax.set_ylim(0, 105)  # Piccolo margine sopra il 100%
    ax.grid(True, alpha=0.3)
    
    # Migliora i tick dell'asse x - usa solo i bin centers che hanno dati
    if len(bin_centers) > 0:
        ax.set_xticks(bin_centers)
        # Crea le etichette solo per i bin con dati
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
    y_pred, y_pred_class, correct = compute_predictions(model, X_test, y_test, threshold)

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
    ax4.set_xlabel("Output del modello")

    ax5 = fig.add_subplot(gs[3])
    plot_age_histogram_percentage(ax5, y_pred, correct, ages, age_step=2)

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