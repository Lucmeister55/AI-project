import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def augment_brightness(images):
    images = tf.map_fn(
        lambda img: tf.image.random_brightness(img, max_delta=0.4), images
    )  # ±40% brightness
    return tf.clip_by_value(images, 0, 255)


def augment_contrast(images):
    def adjust_contrast(img):
        img = tf.expand_dims(
            img, axis=-1
        )  # Restore channel dimension (H, W) → (H, W, 1)
        img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
        img = tf.squeeze(img, axis=-1)  # Remove channel dimension (H, W, 1) → (H, W)
        return tf.clip_by_value(img, 0, 255)

    return tf.map_fn(adjust_contrast, images)


def augment_gaussian_noise(images, stddev=0.1, noise_prob=0.01):
    noise = tf.random.normal(shape=tf.shape(images), mean=0, stddev=stddev)
    mask = tf.cast(tf.random.uniform(tf.shape(images), 0, 1) < noise_prob, tf.float32)
    return tf.clip_by_value(images + noise * mask, 0, 255)


def downsample_images(images, size):
    images = tf.convert_to_tensor(images, dtype=tf.float32)  # Convert to Tensor
    return tf.image.resize(images[..., None], (size, size), method="bilinear")[
        ..., 0
    ].numpy()


def extract_data(path):
    X_array = []
    y_array = []

    # Loop over all folders (classes)
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if os.path.isdir(folder_path):  # Check if it's a folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)

                # Load image and append to dataset
                img = plt.imread(file_path)
                X_array.append(np.asarray(img))
                y_array.append(folder)  # Label is the folder name

    return np.asarray(X_array), np.asarray(y_array)
