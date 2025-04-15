import numpy as np


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
