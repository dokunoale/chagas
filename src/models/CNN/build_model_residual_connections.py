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

# Residual Block
def residual_block(inputs, filters, kernel_size, use_se=True):
    x = layers.Conv1D(filters, kernel_size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if use_se:
        x = se_block(x)

    # Match dimensions if needed
    if inputs.shape[-1] != filters:
        inputs = layers.Conv1D(filters, 1, padding='same')(inputs)

    x = layers.add([x, inputs])
    x = layers.ReLU()(x)
    return x

# Conv Blocks con Residual
def build_residual_conv_blocks(inputs, dropout_rate=0.3, use_se=True):
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = residual_block(x, filters=64, kernel_size=5, use_se=use_se)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = residual_block(x, filters=128, kernel_size=3, use_se=use_se)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = residual_block(x, filters=256, kernel_size=3, use_se=use_se)
    x = layers.GlobalAveragePooling1D()(x)

    return x

# Modello CNN con Residual Blocks
def build_resnet_ecg_model(input_shape=(2800, 12), dropout_rate=0.3, use_se=True):
    inputs = layers.Input(shape=input_shape)
    x = build_residual_conv_blocks(inputs, dropout_rate=dropout_rate, use_se=use_se)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
