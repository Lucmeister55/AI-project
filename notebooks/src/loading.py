import matplotlib.pyplot as plt
import numpy as np
import os

# Function to convert data from a generator into NumPy arrays.
def generator_to_array(generator):
    X_list, y_list = [], []
    # Loop over all batches in the generator.
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        X_list.append(batch_x)
        y_list.append(batch_y)
    # Concatenate all batches into a single array.
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Invert the class_indices dictionary to map {0: 'COVID', 1: 'NORMAL'} etc.
    index_to_class = {v: k for k, v in generator.class_indices.items()}

    # Convert float labels to int, then map them to class names.
    labels = np.array([index_to_class[int(label)] for label in y])
    
    return X, y, labels