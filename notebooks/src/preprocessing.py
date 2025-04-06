import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def downsample_images(images, size):
    images = tf.convert_to_tensor(images, dtype=tf.float32)  # Convert to Tensor
    resized = tf.image.resize(images, (size, size), method="bilinear").numpy()
    return resized

def normalize_images(images, strategy="fixed", training_mean=None, training_std=None):
    # Convert images to float32 for safe division and arithmetic.
    images = images.astype(np.float32)
    
    if strategy == "fixed":
        # If the image pixel values are in [0, 255], then dividing by 255.0 maps them to [0, 1].
        normalized = images / 255.0
        
    elif strategy == "sample":
        # Compute mean and std over the entire batch.
        mean = images.mean()
        std = images.std()
        normalized = (images - mean) / std
        
    elif strategy == "training":
        # Use the provided training mean and std to normalize.
        normalized = (images - training_mean) / training_std
        
    return normalized

def augment_brightness(images, max_delta=0.4):
    # Applies random brightness adjustments (±40% brightness) to each 3-channel image.
    augmented = tf.vectorized_map(
        lambda img: tf.clip_by_value(tf.image.random_brightness(img, max_delta=max_delta), 0, 255),
        images
    )
    return augmented.numpy()

def augment_contrast(images, lower=0.6, upper=1.4):
    # Applies random contrast adjustment to each 3-channel image.
    augmented = tf.vectorized_map(
        lambda img: tf.clip_by_value(tf.image.random_contrast(img, lower=lower, upper=upper), 0, 255),
        images
    )
    return augmented.numpy()

def augment_gaussian_noise(images, stddev=0.1, noise_prob=0.01):
    # Adds Gaussian noise with a given stddev and probability per pixel to 3-channel images.
    noise = tf.random.normal(tf.shape(images), mean=0, stddev=stddev)
    mask = tf.cast(tf.random.uniform(tf.shape(images)) < noise_prob, tf.float32)
    augmented = tf.clip_by_value(images + noise * mask, 0, 255)
    return augmented.numpy()

def get_image_generator(path, img_height, img_width, batch_size, shuffle=True, preprocess=False):
    if preprocess == True:
        image_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.5
        )
    else:
        # For validation and testing, only rescale (or add any minimal preprocessing you need)
        image_gen = ImageDataGenerator(rescale=1./255)

    data_gen = image_gen.flow_from_directory(
        directory=path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Change this if you have a different type of label structure
        shuffle=shuffle
    )
    
    return data_gen

