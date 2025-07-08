from sklearn.metrics import roc_curve

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