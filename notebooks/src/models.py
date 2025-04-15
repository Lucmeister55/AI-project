import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input


def custom_CNN(img_shape, dropout_rate, filters, dense_unit, lr, metrics):
    model = Sequential(
        [
            Input(shape=img_shape),
            Conv2D(filters, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Dropout(dropout_rate),
            Conv2D(filters * 2, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(filters * 4, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Dropout(dropout_rate),
            Flatten(),
            Dense(dense_unit, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Create optimizer with the given learning rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics
    )

    return model
