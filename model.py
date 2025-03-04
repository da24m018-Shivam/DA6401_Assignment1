import numpy as np
from activation import sigmoid, relu, tanh, identity
from activation import sigmoid_backward, relu_backward, tanh_backward, identity_backward

class NeuralNetwork:
    def __init__(self, layer_dims, activations, learning_rate=0.01):
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # number of layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.parameters = {}
        self.cache = {}
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """
        Initialize parameters for the neural network
        """
        np.random.seed(42)  # for reproducibility
        
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
    
    def forward_propagation(self, X):
        
        A = X
        self.cache['A0'] = X
        
        # Implement forward propagation for each layer
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            # Preactivation
            Z = np.dot(W, A_prev) + b
            self.cache[f'Z{l}'] = Z
            
            # Activation
            if self.activations[l-1] == 'sigmoid':
                A = sigmoid(Z)
            elif self.activations[l-1] == 'relu':
                A = relu(Z)
            elif self.activations[l-1] == 'tanh':
                A = tanh(Z)
            elif self.activations[l-1] == 'identity':
                A = identity(Z)
            else:
                raise ValueError(f"Unsupported activation: {self.activations[l-1]}")
            
            self.cache[f'A{l}'] = A
        
        return A
    
    def compute_cost(self, AL, Y):
        
        m = Y.shape[1]
        
        # Compute loss from AL and Y
        logprobs = np.multiply(np.log(AL + 1e-8), Y) + np.multiply(np.log(1 - AL + 1e-8), 1 - Y)
        cost = -np.sum(logprobs) / m
        
        return cost
    
    def backward_propagation(self, Y):
        m = Y.shape[1]
        grads = {}
        
        # Initialize backpropagation
        dAL = - (np.divide(Y, self.cache[f'A{self.L}'] + 1e-8) - np.divide(1 - Y, 1 - self.cache[f'A{self.L}'] + 1e-8))
        
        # Backward propagation for each layer
        dA_prev = dAL
        for l in reversed(range(1, self.L + 1)):
            dA = dA_prev
            Z = self.cache[f'Z{l}']
            A_prev = self.cache[f'A{l-1}']
            W = self.parameters[f'W{l}']
            
            # Compute gradients
            if self.activations[l-1] == 'sigmoid':
                dZ = sigmoid_backward(dA, Z)
            elif self.activations[l-1] == 'relu':
                dZ = relu_backward(dA, Z)
            elif self.activations[l-1] == 'tanh':
                dZ = tanh_backward(dA, Z)
            elif self.activations[l-1] == 'identity':
                dZ = identity_backward(dA, Z)
            else:
                raise ValueError(f"Unsupported activation: {self.activations[l-1]}")
            
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W.T, dZ)
            
            grads[f'dW{l}'] = dW
            grads[f'db{l}'] = db
        
        return grads
    
    def update_parameters(self, grads):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
    
    def predict(self, X):
        AL = self.forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions