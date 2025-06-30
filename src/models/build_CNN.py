from layers import LogSpectrogram
import tensorflow as tf
from tensorflow.keras import layers, models


def build_ecg_model_with_spectrogram(
    input_shape=(2800, 12),
    dropout_rate=0.3
):
    inputs = layers.Input(shape=input_shape)  # (2800, 12)

    # 1) Compute log spectrograms
    x = LogSpectrogram()(inputs)  # (batch, time_frames, freq_bins, leads=12)

    # 2) Conv blocks with padding='same' to avoid dimension collapse
    x = layers.Conv2D(96, kernel_size=7, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = layers.Conv2D(384, kernel_size=3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(pool_size=(5, 3), strides=(3, 2), padding='same')(x)

    # Flatten e fully connected
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # output binario

    model = models.Model(inputs, outputs)
    return model