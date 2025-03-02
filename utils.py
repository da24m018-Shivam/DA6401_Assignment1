import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(X_train, y_train, X_test, y_test):
    
    # Reshape and normalize images
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0  # Reshape to (784, n_samples)
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
    
    # One-hot encode the labels
    n_classes = 10
    Y_train = np.zeros((n_classes, X_train.shape[1]))
    Y_test = np.zeros((n_classes, X_test.shape[1]))
    
    for i in range(X_train.shape[1]):
        Y_train[y_train[i], i] = 1
    
    for i in range(X_test.shape[1]):
        Y_test[y_test[i], i] = 1
    
    return X_train, Y_train, X_test, Y_test, y_train, y_test

def create_mini_batches(X, Y, batch_size):
    
    m = X.shape[1]
    mini_batches = []
    
    # Shuffle the data
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    # Create mini-batches
    num_complete_minibatches = m // batch_size
    
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * batch_size:(k + 1) * batch_size]
        mini_batch_Y = shuffled_Y[:, k * batch_size:(k + 1) * batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Handle the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches

def compute_accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    
    # Compute confusion matrix using our custom function
    cm = compute_confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return plt.gcf()  # Return figure for wandb logging

def plot_sample_images(X, y, class_names=None):
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    # Get one sample from each class
    for i in range(10):
        idx = np.where(y == i)[0][0]
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        if class_names:
            axes[i].set_title(class_names[i])
        else:
            axes[i].set_title(f'Class {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig