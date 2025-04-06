from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

def custom_CNN(img_shape, dropout_rate, filters, lr, dense_unit):
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(filters, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(dropout_rate),
        Conv2D(filters * 2, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(filters * 4, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(dropout_rate),
        Flatten(),
        Dense(dense_unit, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model