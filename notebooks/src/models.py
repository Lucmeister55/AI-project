import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, RandomBrightness, RandomContrast, RandomRotation, GlobalAveragePooling2D, Input, BatchNormalization, RandomZoom
from tensorflow.keras.models import Sequential


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



def transfer_model(base_model, dropout_rate, dense_unit, lr, metrics):

    # Define augmentation pipeline
    augmentation = Sequential([
        RandomBrightness(factor=0.1),
        RandomContrast(factor=0.1),
        RandomRotation((-0.040, 0.040)),
    ], name="augmentation")


    model = Sequential(
        [
            Input(shape=(128, 128, 3)),
            augmentation,
            base_model, 
            Dropout(dropout_rate),
            GlobalAveragePooling2D(),
            Dense(dense_unit, activation="relu"),
            Dropout(dropout_rate),
            Dense(dense_unit, activation="relu"),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Create optimizer with the given learning rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics
    )

    return model

