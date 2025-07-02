# classifier.py

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from lstm_block import build_lstm_block

def build_classifier(input_shape=(1800,12),
                    num_classes=2,
                    lstm_units=128,
                    lstm_layers=2,
                    bidirectional=True,
                    dropout_rate=0.3):
    """
    Costruisce un modello LSTM per la classificazione sequenziale.

    Parametri:
    ----------
    input_shape : tuple, opzionale
        La forma dell'input, tipicamente (timesteps, features). Default è (1800, 12).
    num_classes : int, opzionale
        Numero di classi per la classificazione. Se è 1, si assume classificazione binaria con output singolo.
        Default è 2.
    lstm_units : int, opzionale
        Numero di unità in ogni layer LSTM. Default è 128.
    lstm_layers : int, opzionale
        Numero di layer LSTM impilati. Default è 2.
    bidirectional : bool, opzionale
        Se True, utilizza LSTM bidirezionali. Default è True.
    dropout_rate : float, opzionale
        Tasso di dropout applicato tra i layer LSTM. Default è 0.3.

    Ritorna:
    --------
    model : tensorflow.keras.models.Model
        Il modello Keras LSTM compilato pronto per l'addestramento.
    """
    inputs = Input(shape=input_shape)
    x = build_lstm_block(inputs,
                        num_units=lstm_units,
                        num_layers=lstm_layers,
                        bidirectional=bidirectional,
                        dropout_rate=dropout_rate)
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = Dense(num_classes, activation=activation)(x)
    
    model = Model(inputs, outputs)
    return model
