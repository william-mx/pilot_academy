"""Vision Transformer model for behavioral cloning with command input."""

import tensorflow as tf
from tensorflow.keras import layers


def build_vit_nav(
    input_shape,
    channels=(64, 128),
    heads=4,
    mlp_ratio=2.0,
    dropout=0.0,
    patch_size=16,
    num_commands=6,
):
    """
    Build a ViT-style model for behavioral cloning with command conditioning.

    Args:
        input_shape: Tuple of (height, width, channels) for input images
        channels: Tuple of channel sizes for tokenizer convolutions
        heads: Number of attention heads in transformer blocks
        mlp_ratio: Expansion ratio for MLP layers
        dropout: Dropout rate for regularization
        patch_size: Size of image patches
        num_commands: Number of possible navigation commands

    Returns:
        Keras model that takes [image, command] and predicts steering angle
    """
    H, W, C = input_shape
    d_model = channels[-1]

    # Inputs
    image_input = layers.Input(shape=input_shape, name="image")
    command_input = layers.Input(shape=(), dtype="int32", name="command")

    # Normalize pixel values
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(image_input)

    # Tokenizer: Convert image to patches
    x = layers.Conv2D(
        d_model,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="patch_embed",
    )(x)
    tokens = layers.Reshape((-1, d_model), name="tokens")(x)

    # Command embedding
    cmd_emb = layers.Embedding(
        input_dim=num_commands,
        output_dim=d_model,
        name="cmd_embedding",
    )(command_input)
    cmd_token = layers.Reshape((1, d_model), name="cmd_token")(cmd_emb)

    # Prepend command token to image tokens
    x = layers.Concatenate(axis=1, name="concat_tokens")([cmd_token, tokens])

    # Transformer encoder blocks
    x = _encoder_block(x, d_model, heads, mlp_ratio, dropout, prefix="enc1")
    x = _encoder_block(x, d_model, heads, mlp_ratio, dropout, prefix="enc2")

    # Pool and predict
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    outputs = layers.Dense(1, name="steering")(x)

    return tf.keras.Model(
        inputs=[image_input, command_input],
        outputs=outputs,
        name="command_vit",
    )


def _encoder_block(x, model_dim, heads, mlp_ratio, dropout, prefix):
    """Single transformer encoder block with self-attention and FFN."""
    mlp_dim = int(model_dim * mlp_ratio)
    key_dim = model_dim // heads

    # Self-attention with residual
    skip = x
    x_norm = layers.LayerNormalization(name=f"{prefix}_ln1")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=key_dim,
        name=f"{prefix}_mha",
    )(x_norm, x_norm)
    x = layers.Dropout(dropout, name=f"{prefix}_drop1")(attn_out)
    x = layers.Add(name=f"{prefix}_add1")([x, skip])

    # Feed-forward network with residual
    skip2 = x
    y = layers.LayerNormalization(name=f"{prefix}_ln2")(x)
    y = layers.Dense(mlp_dim, activation="gelu", name=f"{prefix}_ff1")(y)
    y = layers.Dense(model_dim, name=f"{prefix}_ff2")(y)
    y = layers.Dropout(dropout, name=f"{prefix}_drop2")(y)
    x = layers.Add(name=f"{prefix}_add2")([y, skip2])

    return x