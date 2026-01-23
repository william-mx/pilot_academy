import tensorflow as tf

def build_pilotnet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(10, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # keep your two-layer head, but simplified naming
    x = tf.keras.layers.Dense(1, name="kp", use_bias=False)(x)
    outputs = tf.keras.layers.Dense(1, name="steering")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="pilotnet")
