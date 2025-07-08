from sklearn.metrics import roc_curve

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
