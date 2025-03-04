import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def relu(Z):
    return np.maximum(0, Z)

def identity(Z):
    return Z

def sigmoid_backward(dA, Z):
    A = sigmoid(Z)
    return dA * A * (1 - A)

def tanh_backward(dA, Z):    
    return dA * (1 - np.power(np.tanh(Z), 2))

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def identity_backward(dA, Z):
    return dA