import numpy as np
import pathlib
import tensorflow as tf

import src.preprocessing as preprocessing


# Function to convert data from a generator into NumPy arrays.
def generator_to_array(generator):
    X_list, y_list = [], []

    # Loop over all batches in the generator.
    for batch in generator:
        batch_x, batch_y = batch
        X_list.append(batch_x.numpy())
        y_list.append(batch_y.numpy())

    # Concatenate all batches.
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Map numeric labels to string class names.
    labels = np.array([generator.class_names[int(label)] for label in y])

    return X, y, labels


def _count_files(directory) -> int:
    base_path = pathlib.Path(directory)
    total = 0
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            total += sum(1 for _ in subdir.iterdir() if _.is_file())
    return total


def _extract_classes(data_gen):
    all_labels = []
    for _, labels in data_gen:
        all_labels.extend(labels.numpy())
    return np.array(all_labels)


def get_images(
    path, img_height, img_width, batch_size, norm=None, color_mode="grayscale"
):
    train_data_gen = tf.keras.utils.image_dataset_from_directory(
        directory=path + "/train",
        label_mode="binary",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        color_mode=color_mode,
    )
    val_data_gen = tf.keras.utils.image_dataset_from_directory(
        directory=path + "/val",
        label_mode="binary",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,
        color_mode=color_mode,
    )
    train_and_val_data_gen = tf.keras.utils.image_dataset_from_directory(
        directory=path + "/train_and_val",
        label_mode="binary",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        color_mode=color_mode,
    )
    test_data_gen = tf.keras.utils.image_dataset_from_directory(
        directory=path + "/test",
        label_mode="binary",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,
        color_mode=color_mode,
    )

    if norm is not None:
        # Compute training mean and std
        training_images = []
        for x_batch, _ in train_data_gen:
            training_images.append(tf.cast(x_batch, tf.float32))
        training_images = tf.concat(training_images, axis=0)
        training_mean = tf.reduce_mean(training_images)
        training_std = tf.math.reduce_std(training_images)

        # Define the normalization function
        def normalize_fn(x, y):
            return (
                preprocessing.normalize_images(
                    x,
                    strategy=norm,
                    training_mean=training_mean,
                    training_std=training_std,
                ),
                y,
            )

        # Apply normalization to all datasets
        class_names = train_data_gen.class_names

        train_data_gen = train_data_gen.map(normalize_fn)
        val_data_gen = val_data_gen.map(normalize_fn)
        train_and_val_data_gen = train_and_val_data_gen.map(normalize_fn)
        test_data_gen = test_data_gen.map(normalize_fn)

        train_data_gen.class_names = class_names
        val_data_gen.class_names = class_names
        train_and_val_data_gen.class_names = class_names
        test_data_gen.class_names = class_names

    train_data_gen.samples = _count_files(path + "/train")
    val_data_gen.samples = _count_files(path + "/val")
    train_and_val_data_gen.samples = _count_files(path + "/train_and_val")
    test_data_gen.samples = _count_files(path + "/test")
    train_data_gen.classes = _extract_classes(train_data_gen)
    val_data_gen.classes = _extract_classes(val_data_gen)
    train_and_val_data_gen.classes = _extract_classes(train_and_val_data_gen)
    test_data_gen.classes = _extract_classes(test_data_gen)

    return train_data_gen, val_data_gen, train_and_val_data_gen, test_data_gen
