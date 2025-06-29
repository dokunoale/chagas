import tensorflow as tf
from tensorflow.keras import layers, models

# SE Block
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.multiply([input_tensor, se])

# Conv Blocks with optional SE Block
def build_conv_blocks(inputs, dropout_rate=0.3, use_se=True):
    # Conv Block 1
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv Block 2
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 3
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_se:
        x = se_block(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 4
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_se:
        x = se_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    return x

# Model
def build_cnn_ecg_model(input_shape=(2800, 12), dropout_rate=0.3, use_se=True):
    inputs = layers.Input(shape=input_shape)
    x = build_conv_blocks(inputs, dropout_rate=dropout_rate, use_se=use_se)

    # Fully Connected
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
