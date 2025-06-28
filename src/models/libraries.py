# src/models/libraries.py

import numpy as np
import matplotlib.pyplot as plt
import gdown
import wfdb
import importlib
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.utils import class_weight
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    classification_report
)

# Definisci cosa verrà importato con `from libraries import *`
__all__ = [
    'np', 'plt', 'gdown', 'wfdb', 'importlib', 'tf', 'layers', 'models',
    'EarlyStopping', 'ReduceLROnPlateau', 'ModelCheckpoint', 'AUC',
    'class_weight', 'roc_curve', 'confusion_matrix', 'ConfusionMatrixDisplay',
    'accuracy_score', 'roc_auc_score', 'classification_report', 'BinaryCrossentropy'
]