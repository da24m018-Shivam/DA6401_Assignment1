import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(X_train, y_train, X_test, y_test):
    """Reshapes, normalizes images, and one-hot encodes labels."""

    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0  
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

    n_classes = 10
    Y_train = np.zeros((n_classes, X_train.shape[1]))
    Y_test = np.zeros((n_classes, X_test.shape[1]))

    for i in range(X_train.shape[1]):
        Y_train[y_train[i], i] = 1
    for i in range(X_test.shape[1]):
        Y_test[y_test[i], i] = 1

    return X_train, Y_train, X_test, Y_test, y_train, y_test

def create_mini_batches(X, Y, batch_size):
    """Shuffles data and creates mini-batches."""
    m = X.shape[1]
    mini_batches = []
    indices = np.random.permutation(m)

    for k in range(0, m, batch_size):
        batch_indices = indices[k:k + batch_size]
        mini_batches.append((X[:, batch_indices], Y[:, batch_indices]))

    return mini_batches

def compute_accuracy(predictions, labels):
    """Calculates classification accuracy as a percentage."""
    return np.mean(predictions == labels) * 100

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Computes confusion matrix for classification results."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Generates and visualizes a confusion matrix."""
    cm = compute_confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return plt.gcf()  # Return figure for logging

def plot_sample_images(X, y, class_names=None):
    """Displays sample images from the dataset."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(10):
        idx = np.where(y == i)[0][0]
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(class_names[i] if class_names else f'Class {i}')
        axes[i].axis('off')

    plt.tight_layout()
    return fig
