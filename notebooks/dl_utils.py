import numpy as np
import os
from PIL import Image


def extract_data(path):
    X_array = []
    y_array = []

    # Loop over all folders (classes)
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if os.path.isdir(folder_path):  # Check if it's a folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)

                # Load image, normalize (0 to 1 scale), and append to dataset
                img = Image.open(file_path)
                X_array.append(np.asarray(img) / 255)
                y_array.append(folder)  # Label is the folder name

    return np.asarray(X_array), np.asarray(y_array)
