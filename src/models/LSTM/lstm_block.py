# lstm_block.py

from tensorflow.keras.layers import LSTM, Bidirectional, Dropout

def build_lstm_block(inputs,
                    num_units=128,
                    num_layers=2,
                    bidirectional=True,
                    dropout_rate=0.3,
                    recurrent_dropout=0.0):
    """
    Costruisce un blocco LSTM con più layer, opzionalmente bidirezionale,
    con dropout applicato dopo ogni layer.
    """
    x = inputs
    
    for i in range(num_layers):
        # Determina se il layer deve restituire sequenze (True per tutti tranne l'ultimo)
        return_sequences = (i < num_layers - 1)
        lstm = LSTM(num_units, return_sequences=return_sequences, recurrent_dropout=recurrent_dropout)
        if bidirectional:
            lstm = Bidirectional(lstm)
        x = lstm(x)
        x = Dropout(dropout_rate)(x)
    
    return x
