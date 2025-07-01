import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_ecg_lstm_model(
    input_shape=(2800, 12),
    lstm_units=64,
    dropout_rate=0.4
):
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)


    #Primo blocco LSTM
    x = layers.Bidirectional(layers.LSTM(
        lstm_units, return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4)
    ))(x)
    
    x = layers.Dropout(dropout_rate)(x)

    #Secondo blocco LSTM
    x = layers.Bidirectional(layers.LSTM(
        lstm_units,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4)
    ))(x)
    
    x = layers.Dropout(dropout_rate)(x)

    #Dense Layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)

    #Output Layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model
