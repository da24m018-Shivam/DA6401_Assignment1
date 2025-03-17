import numpy as np

class SGD:
    """
    Stochastic Gradient Descent (SGD):

    Update rule:
        w = w - learning_rate * grad
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params:
            grad_key = f'd{key}'
            if grad_key in grads:
                params[key] -= self.learning_rate * grads[grad_key]
        return params


class Momentum:
    """
    Momentum-based Gradient Descent:

    Velocity update:
        v = β * v + (1 - β) * grad
    Parameter update:
        w = w - learning_rate * v
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            grad_key = f'd{key}'
            if grad_key in grads:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(params[key])  # Initialize velocity

                self.velocity[key] = self.beta * self.velocity[key] + (1 - self.beta) * grads[grad_key]
                params[key] -= self.learning_rate * self.velocity[key]
        return params


class Nesterov:
    """
    Nesterov Accelerated Gradient (NAG):

    Lookahead step:
        w_lookahead = w - β * v
    Compute gradient at lookahead position:
        g_lookahead = ∇J(w_lookahead)
    Velocity update:
        v = β * v + (1 - β) * g_lookahead
    Parameter update:
        w = w - learning_rate * v
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            grad_key = f'd{key}'
            if grad_key in grads:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(params[key])

                # Lookahead step
                lookahead_W = params[key] - self.beta * self.velocity[key]
                g_lookahead = grads[grad_key]  # Ideally recompute at lookahead_W

                self.velocity[key] = self.beta * self.velocity[key] + (1 - self.beta) * g_lookahead
                params[key] -= self.learning_rate * self.velocity[key]

        return params


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation):

    Moving average of squared gradients:
        v = β * v + (1 - β) * grad^2
    Parameter update:
        w = w - learning_rate * grad / (sqrt(v) + epsilon)
    """
    def __init__(self, learning_rate=0.01, beta=0.9, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps
        self.cache = {}

    def update(self, params, grads):
        for key in params:
            grad_key = f'd{key}'
            if grad_key in grads:
                if key not in self.cache:
                    self.cache[key] = np.zeros_like(params[key])

                self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * (grads[grad_key] ** 2)
                params[key] -= self.learning_rate * grads[grad_key] / (np.sqrt(self.cache[key]) + self.eps)
        return params


class Adam:
    """
    Adam (Adaptive Moment Estimation):

    First moment:
        m = β1 * m + (1 - β1) * grad
    Second moment:
        v = β2 * v + (1 - β2) * grad^2
    Bias correction:
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
    Parameter update:
        w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """Applies Adam update to parameters."""
        self.t += 1  # Increment time step

        for key in params:
            grad_key = f'd{key}'
            if grad_key in grads:
                if key not in self.m:
                    self.m[key] = np.zeros_like(params[key])
                    self.v[key] = np.zeros_like(params[key])

                # Compute first moment estimate (momentum)
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[grad_key]

                # Compute second moment estimate (RMSprop-like variance)
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[grad_key] ** 2)

                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                # Update parameters
                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return params


class NAdam:
    """
    NAdam (Nesterov-accelerated Adam):

    First moment:
        m = β1 * m + (1 - β1) * grad
    Second moment:
        v = β2 * v + (1 - β2) * grad^2
    Bias correction:
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
    Parameter update:
        w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """Applies NAdam update to parameters."""
        self.t += 1  # Increment time step

        for key in params:
            grad_key = f'd{key}'
            if grad_key in grads:
                if key not in self.m:
                    self.m[key] = np.zeros_like(params[key])
                    self.v[key] = np.zeros_like(params[key])

                # Compute first and second moment estimates
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[grad_key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[grad_key] ** 2)

                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                # Update parameters
                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return params


# Dictionary to access optimizers
def get_optimizer(name, learning_rate=0.01):
    """Returns an optimizer instance based on the given name."""
    optimizers = {
        'sgd': SGD(learning_rate),
        'momentum': Momentum(learning_rate),
        'nag': Nesterov(learning_rate),
        'rmsprop': RMSprop(learning_rate),
        'adam': Adam(learning_rate),
        'nadam': NAdam(learning_rate),
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name]