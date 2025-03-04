import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import argparse

# Import our modules
from model import NeuralNetwork
from trainer import Trainer
from optimizers import get_optimizer
from utils import preprocess_data, plot_confusion_matrix, plot_sample_images

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of units in hidden layer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use (sgd)')
    args = parser.parse_args()
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess data
    X_train, Y_train, X_test, Y_test, y_train, y_test = preprocess_data(
        x_train[:10000], y_train[:10000], x_test[:1000], y_test[:1000]
    )
    
    # Plot sample images
    class_names = [str(i) for i in range(10)]
    sample_img_fig = plot_sample_images(x_test, y_test, class_names)
    plt.show()
    
    # Define network architecture
    input_dim = X_train.shape[0]  # 784
    hidden_dim = args.hidden_dim  # Hidden layer size
    output_dim = 10  # Number of classes
    
    layer_dims = [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'sigmoid']  # ReLU for hidden, sigmoid for output
    
    # Create neural network
    model = NeuralNetwork(
        layer_dims=layer_dims,
        activations=activations,
        learning_rate=args.learning_rate
    )
    
    # Get optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    
    # Create trainer
    trainer = Trainer(model, optimizer)
    
    # Train the model
    print(f"Training with: hidden_dim={args.hidden_dim}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, learning_rate={args.learning_rate}, "
          f"optimizer={args.optimizer}")
    
    trainer.train(
        X_train, Y_train, X_test, Y_test, y_train, y_test,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        print_every=1
    )
    
    # Plot training history
    history_fig = trainer.plot_training_history()
    plt.show()
    
    # Make predictions on test set
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test) * 100
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_test, test_predictions, class_names)
    plt.show()

if __name__ == "__main__":
    main()