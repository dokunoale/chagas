import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def build_conv_blocks(inputs, dropout_rate=0.3):
    # Conv Block 1
    x = layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv Block 2
    x = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 3
    x = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 4
    x = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)

    return x



def build_cnn_ecg_model(input_shape=(2800, 12), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)
    x = build_conv_blocks(inputs, dropout_rate=dropout_rate)

    # Fully Connected
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
