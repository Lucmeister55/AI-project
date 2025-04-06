import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_dataset_size(X, y):
    print(f"Train shape: {X.shape}, Labels: {y.shape}")

def check_image_shape_consistency(X):
    consistent = all(img.shape == X[0].shape for img in X)
    print("All images have the same shape:", consistent)
    return consistent

def plot_label_distribution(y):
    plt.figure(figsize=(8,6))
    sns.countplot(x=y, hue=y)
    plt.title("Label Distribution")
    plt.show()

def get_indices_by_label(y):
    unique_labels = np.unique(y)
    indices_by_label = {label: np.where(y == label)[0] for label in unique_labels}
    return indices_by_label

def plot_first_images_by_label(X, y, num_images=5):
    """Plot the first 'num_images' for each label."""
    indices_by_label = get_indices_by_label(y)
    unique_labels = np.unique(y)
    
    n_labels = len(unique_labels)
    fig, axs = plt.subplots(n_labels, num_images, figsize=(15, 3 * n_labels))
    
    for i, label in enumerate(unique_labels):
        indices = indices_by_label[label]
        for j in range(num_images):
            # If there's only one label, axs might be 1D
            if n_labels > 1:
                ax = axs[i, j]
            else:
                ax = axs[j]
            # Use modulo if there are fewer than num_images images
            index = indices[j % len(indices)]
            ax.imshow(X[index], cmap='gray')
            ax.set_title(label)
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def display_pixel_statistics(X):
    # Compute per-pixel statistics (across the dataset)
    pixel_average = np.mean(X, axis=0)
    global_average = np.mean(X)
    pixel_std_dev = np.std(X, axis=0)
    global_std_dev = np.std(X)
    
    # Plot the computed statistics
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pixel_average, cmap="gray")
    plt.xlabel(f"Global average: {global_average:.2f}")
    
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pixel_std_dev, cmap="gray")
    plt.xlabel(f"Global std dev: {global_std_dev:.2f}")
    
    plt.tight_layout()
    plt.show()

def run_all_exploration(X, y):
    print_dataset_size(X, y)
    check_image_shape_consistency(X)
    plot_label_distribution(y)
    plot_first_images_by_label(X, y, num_images=5)
    display_pixel_statistics(X)