from tensorflow.keras import layers, models
from layers import LogSpectrogram
import tensorflow as tf

def build_ecg_model_with_spectrogram(input_shape=(2800, 12), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)

    # Log-spectrogram: output (88, 64, 12)
    x = LogSpectrogram()(inputs)  # (None, 88, 64, 12)

    # conv1: kernel 5x5, stride 2x2, 96 filters
    x = layers.Conv2D(96, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)  # → (44, 32, 96)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)                                     # → (22, 16, 96)

    # conv2: kernel 3x3, stride 2x2, 256 filters
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x) # → (11, 8, 256)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)                                     # → (5, 4, 256)

    # conv3: 3x3, stride 1x1, 384 filters
    x = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x) # → (5, 4, 384)

    # conv4: 3x3, stride 1x1, 256 filters
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x) # → (5, 4, 256)

    # conv5: 3x3, stride 1x1, 256 filters
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x) # → (5, 4, 256)

    # mpool5: kernel 3x2, stride 2x2 (adattato da 5x3, 3x2)
    x = layers.MaxPooling2D(pool_size=(3, 2), strides=(2, 2), padding='valid')(x)                    # → (2, 2, 256)

    # fc6: 2x1 conv (adattamento di 9x1), 4096 filters
    x = layers.Conv2D(4096, kernel_size=(2, 1), activation='relu')(x)                               # → (1, 2, 4096)

    # apool6: global average pooling (asse width)
    x = layers.GlobalAveragePooling2D()(x)                                                          # → (4096,)

    # fc7: Dense 1024
    x = layers.Dense(1024, activation='relu')(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    # fc8: final output
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
