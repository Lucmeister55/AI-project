import tensorflow as tf


def downsample_images(images, size):
    images = tf.cast(images, tf.float32)
    images = tf.convert_to_tensor(images, dtype=tf.float32)  # Convert to Tensor
    resized = tf.image.resize(images, (size, size), method="bilinear")
    return resized


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


def get_image_generator(
    path, img_height, img_width, batch_size, norm=False, shuffle=True
):
    data_gen = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        label_mode="binary",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=shuffle,
        color_mode="grayscale",
    )

    if norm:
        class_names = data_gen.class_names
        data_gen = data_gen.map(lambda x, y: (normalize_images(x), y))
        data_gen.class_names = class_names

    return data_gen
