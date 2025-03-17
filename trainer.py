import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Import from project modules
from model import NeuralNetwork
from optimizers import get_optimizer
from utils import preprocess_data, create_mini_batches, plot_sample_images
from config import FASHION_MNIST_CLASSES, build_train_config_from_args

def train_model(config, dataset_type='fashion_mnist', return_model=False):
    
    # Convert args to config object if passed directly from command line
    if not hasattr(config, 'hidden_layers'):
        config = build_train_config_from_args(config)
        
    if not wandb.run:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config)

    # Load dataset
    dataset = tf.keras.datasets.fashion_mnist if dataset_type == 'fashion_mnist' else tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = dataset.load_data()
    class_names = FASHION_MNIST_CLASSES if dataset_type == 'fashion_mnist' else [str(i) for i in range(10)]

    # Log sample images
    sample_images_fig = plot_sample_images(X_train_full, y_train_full, class_names)
    wandb.log({"sample_images": wandb.Image(sample_images_fig)})
    plt.close(sample_images_fig)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=config.val_split, random_state=42)

    # Preprocessing
    X_train, Y_train, X_val, Y_val, _, _ = preprocess_data(X_train, y_train, X_val, y_val)
    X_test_processed, Y_test, _, _, _, _ = preprocess_data(X_test, y_test, X_test, y_test)

    # Parse hidden layers (if string) - already handled if we used build_train_config_from_args
    if isinstance(config.hidden_layers, str):
        hidden_layers = [int(size) for size in config.hidden_layers.split(',')]
    else:
        hidden_layers = config.hidden_layers

    # Initialize model
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers,
        output_size=10,
        activation=config.activation,
        weight_init=config.weight_init
    )

    optimizer = get_optimizer(config.optimizer, config.learning_rate)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0

    for epoch in range(config.epochs):
        mini_batches = create_mini_batches(X_train, Y_train, config.batch_size)

        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, Y_batch in mini_batches:
            Y_pred = model.forward(X_batch)
            batch_loss = model.compute_loss(Y_pred, Y_batch, config.loss, config.weight_decay)
            train_loss += batch_loss

            gradients = model.backward(Y_pred, Y_batch, config.loss, config.weight_decay)
            model.parameters = optimizer.update(model.parameters, gradients)

            predictions = np.argmax(Y_pred, axis=0)
            true_labels = np.argmax(Y_batch, axis=0)
            train_correct += np.sum(predictions == true_labels)
            train_total += len(true_labels)

        avg_train_loss = train_loss / len(mini_batches)
        train_accuracy = (train_correct / train_total) * 100

        # Validation
        Y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(Y_val_pred, Y_val, config.loss, config.weight_decay)
        val_predictions = np.argmax(Y_val_pred, axis=0)
        val_true_labels = np.argmax(Y_val, axis=0)
        val_accuracy = (np.sum(val_predictions == val_true_labels) / len(val_true_labels)) * 100

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Evaluate on test set
    Y_test_pred = model.forward(X_test_processed)
    test_loss = model.compute_loss(Y_test_pred, Y_test, config.loss, config.weight_decay)
    test_predictions = np.argmax(Y_test_pred, axis=0)
    test_true_labels = np.argmax(Y_test, axis=0)
    test_accuracy = (np.sum(test_predictions == test_true_labels) / len(test_true_labels)) * 100

    # Log final test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    # Plot loss and accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()

    plt.tight_layout()
    wandb.log({"learning_curves": wandb.Image(fig)})
    plt.close(fig)

    if return_model:
        return test_accuracy, model, X_test, y_test, test_predictions  # Return more information
    else:
        return test_accuracy