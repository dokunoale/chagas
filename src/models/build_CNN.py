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

    # 2) Conv blocks
    # Permute to (batch, freq_bins, time_frames, channels) to better match PyTorch conv2d (C,H,W)
    # Actually TensorFlow conv2d expects (batch, H, W, C)
    # Attenzione: la dimensione H=tempo_frames, W=freq_bins, C=leads
    # Nel tuo codice PyTorch, input era (1, 512, 300), quindi canale=1, altezza=512, larghezza=300
    # Qui abbiamo (batch, time_frames, freq_bins, 12), possiamo tenere così (time_frames=freq_frames)
    # Oppure permutiamo per mettere canale come canale e lasciare tempo e freq come H,W
    # Conv2D in TF usa canali-last: (batch, H, W, C)
    # Quindi va bene così (time_frames, freq_bins, 12)
    
    # Primo blocco conv
    x = layers.Conv2D(96, kernel_size=7, strides=2, padding='valid')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Secondo blocco conv
    x = layers.Conv2D(256, kernel_size=5, strides=2, padding='valid')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Terzo, quarto e quinto blocco conv
    x = layers.Conv2D(384, kernel_size=3, strides=1, padding='valid')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(x)
    x = layers.ReLU()(x)

    # Pooling finale
    x = layers.MaxPooling2D(pool_size=(5,3), strides=(3,2))(x)

    # Flatten e fully connected
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # output binario

    model = models.Model(inputs, outputs)
    return model
