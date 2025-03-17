import numpy as np
from activation import get_activation, get_activation_backward, softmax

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', weight_init='xavier'):
        """
        Initialize a neural network with configurable architecture.
        
        Args:
            input_size (int): Size of the input layer
            hidden_layers (list): List of integers specifying the size of each hidden layer
            output_size (int): Size of the output layer
            activation (str): Activation function to use ('relu', 'sigmoid', 'tanh', 'identity')
            weight_init (str): Weight initialization method ('xavier', 'random')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        
        # Get activation functions
        self._activation_forward = get_activation(activation)
        self._activation_backward = get_activation_backward(activation)
        
        # Initialize parameters
        self.parameters = {}
        self.gradients = {}
        self.layer_inputs = {}
        self.layer_outputs = {}
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(1, len(layer_sizes)):
            # Weight initialization methods
            if weight_init.lower() == 'xavier':
                # Xavier/Glorot initialization for sigmoid/tanh
                scale = np.sqrt(2.0 / (layer_sizes[i-1] + layer_sizes[i]))
                self.parameters[f'W{i}'] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * scale
            else:
                # Standard random initialization
                self.parameters[f'W{i}'] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01
                
            # Bias initialization
            self.parameters[f'b{i}'] = np.zeros((layer_sizes[i], 1))
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X (np.ndarray): Input data of shape (input_size, batch_size)
        
        Returns:
            np.ndarray: Output predictions
        """
        # Store input
        self.layer_inputs['A0'] = X
        
        num_layers = len(self.hidden_layers) + 1
        
        # Forward through hidden layers
        for i in range(1, num_layers):
            Z = np.dot(self.parameters[f'W{i}'], self.layer_inputs[f'A{i-1}']) + self.parameters[f'b{i}']
            self.layer_inputs[f'Z{i}'] = Z
            self.layer_inputs[f'A{i}'] = self._activation_forward(Z)
        
        # Output layer with softmax activation
        Z = np.dot(self.parameters[f'W{num_layers}'], self.layer_inputs[f'A{num_layers-1}']) + self.parameters[f'b{num_layers}']
        self.layer_inputs[f'Z{num_layers}'] = Z
        A = softmax(Z)
        self.layer_inputs[f'A{num_layers}'] = A
        
        return A
    
    def compute_loss(self, Y_pred, Y_true, loss='cross_entropy', weight_decay=0):
        """
        Compute loss between predictions and ground truth
        
        Args:
            Y_pred (np.ndarray): Predictions from the network
            Y_true (np.ndarray): Ground truth labels
            loss (str): Type of loss function ('cross_entropy', 'mean_squared_error')
            weight_decay (float): L2 regularization parameter
            
        Returns:
            float: Computed loss value
        """
        m = Y_true.shape[1]
        
        # Compute the main loss based on specified type
        if loss == 'cross_entropy':
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            loss = -np.sum(Y_true * np.log(Y_pred + epsilon)) / m
        elif loss == 'squared_error' or loss == 'mean_squared_error':
            loss = np.sum(np.square(Y_pred - Y_true)) / (2 * m)
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        # Add L2 regularization if weight_decay > 0
        if weight_decay > 0:
            l2_reg = 0
            for i in range(1, len(self.hidden_layers) + 2):
                l2_reg += np.sum(np.square(self.parameters[f'W{i}']))
            loss += (weight_decay / (2 * m)) * l2_reg
            
        return loss
    
    def backward(self, Y_pred, Y_true, loss='cross_entropy', weight_decay=0):
        """
        Backpropagation to compute gradients
        
        Args:
            Y_pred (np.ndarray): Predictions from the network
            Y_true (np.ndarray): Ground truth labels
            loss (str): Type of loss function ('cross_entropy', 'mean_squared_error')
            weight_decay (float): L2 regularization parameter
            
        Returns:
            dict: Gradients for all parameters
        """
        m = Y_true.shape[1]
        num_layers = len(self.hidden_layers) + 1
        
        # Initialize gradients
        self.gradients = {}
        
        # Compute initial gradient for output layer based on loss type
        if loss == 'cross_entropy':
            dZ = Y_pred - Y_true
        elif loss == 'squared_error' or loss == 'mean_squared_error':
            dA = Y_pred - Y_true
            # For the output layer with softmax, need to compute proper gradient
            # Here simplified as linear for MSE - this is a simplification
            dZ = dA
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        # Backpropagation for output layer
        self.gradients[f'dW{num_layers}'] = np.dot(dZ, self.layer_inputs[f'A{num_layers-1}'].T) / m
        self.gradients[f'db{num_layers}'] = np.sum(dZ, axis=1, keepdims=True) / m
        
        # Add L2 regularization if weight_decay > 0
        if weight_decay > 0:
            self.gradients[f'dW{num_layers}'] += (weight_decay / m) * self.parameters[f'W{num_layers}']
        
        # Backpropagation for hidden layers
        for i in range(num_layers - 1, 0, -1):
            dA = np.dot(self.parameters[f'W{i+1}'].T, dZ)
            dZ = self._activation_backward(dA, self.layer_inputs[f'Z{i}'])
            
            self.gradients[f'dW{i}'] = np.dot(dZ, self.layer_inputs[f'A{i-1}'].T) / m
            self.gradients[f'db{i}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            # Add L2 regularization if weight_decay > 0
            if weight_decay > 0:
                self.gradients[f'dW{i}'] += (weight_decay / m) * self.parameters[f'W{i}']
        
        return self.gradients
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predicted class indices
        """
        A = self.forward(X)
        return np.argmax(A, axis=0)
    
    def get_config(self):
        """Returns the configuration of the model"""
        return {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "activation": self.activation,
            "weight_init": self.weight_init
        }
    
    @classmethod
    def from_config(cls, config):
        """Create a model from configuration dictionary"""
        return cls(
            config["input_size"],
            config["hidden_layers"],
            config["output_size"],
            config["activation"],
            config["weight_init"]
        )