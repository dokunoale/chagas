import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
import tensorflow as tf
from tabulate import tabulate
import numpy as np

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Ritorna una funzione di perdita di tipo Focal Loss per problemi di classificazione binaria.

    Args:
        gamma (float): Fattore di focalizzazione, che riduce il peso delle istanze facili da classificare.
        alpha (float): Fattore di bilanciamento tra classi positive e negative.

    Returns:
        funzione di perdita personalizzata da usare con model.compile()
    """
    def loss(y_true, y_pred):
        # Calcola la Binary Crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Probabilità del target
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # Fattore di bilanciamento alpha
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Fattore di modulazione focal loss
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        # Perdita focalizzata
        focal_loss_value = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(focal_loss_value)

    return loss


def make_callback(name):
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    early_stop = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max')
    checkpoint = ModelCheckpoint(f"{name}_best_model.h5",
                             monitor='val_auc',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
    
    callback = [early_stop, reduce_lr, checkpoint]
    return callback


def plot_training_metrics(history):
    """
    Plotta Accuracy, AUC e Loss per training e validation da un oggetto history di Keras.

    Args:
        history: History object restituito da model.fit()
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
    

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Calcola la curva ROC e restituisce la soglia ottimale che massimizza TPR - FPR.

    Args:
        y_true: array-like, etichette vere (0/1)
        y_pred_proba: array-like, probabilità predette per la classe positiva

    Returns:
        optimal_threshold: float, soglia ottimale
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    return optimal_threshold

def show_confusion_matrix(cm, labels=["Negativo", "Positivo"]):
    """
    Visualizza una matrice di confusione già calcolata.

    Args:
        cm (np.ndarray): matrice di confusione
        labels (list): etichette delle classi
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione")
    plt.show()
    

def compare_classification_reports(report1, report2, labels=None, model_names=("Model 1", "Model 2")):
    """
    Confronta due classification_report (output_dict=True) e mostra una tabella con le metriche principali.

    Args:
        report1 (dict): classification_report del primo modello (output_dict=True)
        report2 (dict): classification_report del secondo modello (output_dict=True)
        labels (list, optional): lista di classi da mostrare. Se None, usa tutte le classi trovate.
        model_names (tuple): tuple con i nomi dei due modelli da mostrare nella tabella.

    Stampa una tabella con precision, recall, f1-score per ogni classe e macro/weighted avg.
    """

    # Se labels non specificato, prendi tutte le chiavi eccetto 'accuracy'
    if labels is None:
        labels = [k for k in report1.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]

    rows = []
    header = ["Classe", 
              f"{model_names[0]} Precision", f"{model_names[1]} Precision",
              f"{model_names[0]} Recall",    f"{model_names[1]} Recall",
              f"{model_names[0]} F1-Score",  f"{model_names[1]} F1-Score"]

    for label in labels + ['macro avg', 'weighted avg']:
        r1 = report1.get(label, {})
        r2 = report2.get(label, {})
        row = [
            label,
            f"{r1.get('precision', np.nan):.3f}" if r1 else "N/A",
            f"{r2.get('precision', np.nan):.3f}" if r2 else "N/A",
            f"{r1.get('recall', np.nan):.3f}" if r1 else "N/A",
            f"{r2.get('recall', np.nan):.3f}" if r2 else "N/A",
            f"{r1.get('f1-score', np.nan):.3f}" if r1 else "N/A",
            f"{r2.get('f1-score', np.nan):.3f}" if r2 else "N/A",
        ]
        rows.append(row)

    print(tabulate(rows, headers=header, tablefmt="grid"))
