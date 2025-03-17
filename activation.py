import numpy as np

def identity(Z):
    """
    Identity activation function
    Simply returns the input as is
    """
    return Z

def identity_backward(dA, Z):
    """
    Gradient of identity function
    Simply returns the gradient as is
    """
    return dA

def sigmoid(Z):
    """Sigmoid activation function"""
    # Adding a small epsilon to avoid overflow
    z_safe = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_safe))

def sigmoid_backward(dA, Z):
    """Gradient of sigmoid function"""
    A = sigmoid(Z)
    return dA * A * (1 - A)

def tanh(Z):
    """Hyperbolic tangent activation function"""
    return np.tanh(Z)

def tanh_backward(dA, Z):
    """Gradient of tanh function"""
    A = np.tanh(Z)
    return dA * (1 - np.square(A))

def relu(Z):
    """Rectified Linear Unit activation function"""
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    """Gradient of ReLU function"""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    """Softmax activation function for output layer"""
    # Subtracting max for numerical stability
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def get_activation(activation_name):
    """
    Returns the activation function based on name
    """
    activations = {
        'identity': identity,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu,
        
    }
    
    if activation_name.lower() not in [k.lower() for k in activations]:
        raise ValueError(f"Unsupported activation function: {activation_name}")
    
    # Case-insensitive matching
    for key in activations:
        if key.lower() == activation_name.lower():
            return activations[key]
    
    return activations[activation_name]

def get_activation_backward(activation_name):
    """
    Returns the gradient function for the given activation
    """
    backward_activations = {
        'identity': identity_backward,
        'sigmoid': sigmoid_backward,
        'tanh': tanh_backward,
        'relu': relu_backward,
        
    }
    
    if activation_name.lower() not in [k.lower() for k in backward_activations]:
        raise ValueError(f"Unsupported activation function: {activation_name}")
    
    # Case-insensitive matching
    for key in backward_activations:
        if key.lower() == activation_name.lower():
            return backward_activations[key]
    
    return backward_activations[activation_name]