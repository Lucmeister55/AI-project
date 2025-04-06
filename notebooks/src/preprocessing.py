import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

def downsample_images(images, size):
    images = tf.convert_to_tensor(images, dtype=tf.float32)  # Convert to Tensor
    resized = tf.image.resize(images, (size, size), method="bilinear").numpy()
    return resized

def normalize_images(images, strategy="fixed", training_mean=None, training_std=None):
    images = images.astype(np.float32)
    
    if strategy == "fixed":
        # Assume pixel values are in [0, 255].
        normalized = images / 255.0
    elif strategy == "sample":
        # Compute mean and std from the provided sample.
        mean = images.mean()
        std = images.std()
        normalized = (images - mean) / std
    elif strategy == "training":
        normalized = (images - training_mean) / training_std
    
    return normalized

@tf.autograph.experimental.do_not_convert
def augment_brightness(images, max_delta=0.4):
    # Applies random brightness adjustments (±40% brightness) to each image.
    images = tf.map_fn(
        lambda img: tf.image.random_brightness(img, max_delta=max_delta),
        images
    )
    return tf.clip_by_value(images, 0, 255)

@tf.autograph.experimental.do_not_convert
def augment_contrast(images, lower=0.6, upper=1.4):
    # Applies random contrast adjustment to each image.
    def adjust_contrast(img):
        # Expand dims if needed: (H, W) → (H, W, 1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.image.random_contrast(img, lower=lower, upper=upper)
        # Squeeze back: (H, W, 1) → (H, W)
        img = tf.squeeze(img, axis=-1)
        return tf.clip_by_value(img, 0, 255)
    
    return tf.map_fn(adjust_contrast, images)

def augment_gaussian_noise(images, stddev=0.1, noise_prob=0.01):
    # Adds Gaussian noise with given stddev and a probability per pixel.
    noise = tf.random.normal(shape=tf.shape(images), mean=0, stddev=stddev)
    mask = tf.cast(tf.random.uniform(tf.shape(images), 0, 1) < noise_prob, tf.float32)
    return tf.clip_by_value(images + noise * mask, 0, 255)

def preprocess(images, size, max_delta, lower, upper, stddev, noise_prob):
    images = downsample_images(images, size)
    images = augment_brightness(images, max_delta)
    images = augment_contrast(images, lower, upper)
    images = augment_gaussian_noise(images, stddev, noise_prob)
    return images