import tensorflow as tf


def normalize_images(images, strategy="fixed", training_mean=None, training_std=None):
    # Convert images to float32 for safe division and arithmetic.
    images = tf.cast(images, tf.float32)

    if strategy == "fixed":
        # If the image pixel values are in [0, 255], then dividing by 255.0 maps them to [0, 1].
        normalized = images / 255.0

    elif strategy == "sample":
        # Compute mean and std over the entire batch.
        mean = tf.reduce_mean(images)
        std = tf.math.reduce_std(images)
        normalized = (images - mean) / std

    elif strategy == "training":
        # Use the provided training mean and std to normalize.
        normalized = (images - training_mean) / training_std

    return normalized
