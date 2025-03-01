import numpy as np

class Optimizer:
    """Base class for optimizers"""
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        
    def update(self, params, grads):
        raise NotImplementedError("Subclass must implement abstract method")

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)
        
    def update(self, params, grads):
        """Update parameters using SGD"""
        for key in params:
            params[key] -= self.learning_rate * grads[f'd{key}']
        return params
def get_optimizer(optimizer_name, learning_rate=0.001):
    """Function to get the optimizer"""
    if optimizer_name == 'sgd':
        return SGD(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")