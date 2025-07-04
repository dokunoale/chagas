import tensorflow as tf
from tensorflow.keras import layers, models

def build_ecg_lstm_model(
    input_shape=(2800, 12),
    lstm_units=128,
    dropout_rate=0.3
):
    inputs = layers.Input(shape=input_shape)  # (2800, 12)

    # Primo livello LSTM bidirezionale
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)

    # Secondo livello LSTM bidirezionale (senza return_sequences)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output layer (per classificazione binaria)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
