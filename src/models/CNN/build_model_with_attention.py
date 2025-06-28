import tensorflow as tf
from tensorflow.keras import layers, models

def build_conv_blocks_with_multihead_attention(inputs, dropout_rate=0.3):
    # Conv Block 1
    x = layers.Conv1D(32, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv Block 2
    x = layers.Conv1D(64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 3
    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Multi-Head Attention (self-attention)
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_output])  # Residual connection
    x = layers.LayerNormalization()(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    return x

def build_cnn_multihead_attention_model(input_shape=(2800, 12), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)
    x = build_conv_blocks_with_multihead_attention(inputs, dropout_rate)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
