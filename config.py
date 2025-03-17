import argparse
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
class TrainConfig:
    """Holds training configuration settings."""
    def __init__(self):        
        self.wandb_entity = None
        self.wandb_project = None
        self.num_layers = 1
        self.hidden_size = 4
        self.hidden_layers = []
        self.activation = 'sigmoid'
        self.optimizer = 'sgd'
        self.learning_rate = 0.1
        self.batch_size = 4
        self.epochs = 1
        self.momentum = 0.5
        self.beta = 0.5
        self.beta1 = 0.5
        self.beta2 = 0.5
        self.epsilon = 0.000001
        self.weight_decay = 0.0
        self.weight_init = 'random'
        self.loss = 'cross_entropy'
        self.val_split = 0.1
        self.dataset = 'fashion_mnist'

def parse_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train neural network on Fashion-MNIST or MNIST dataset")
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname',help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',choices=['mnist', 'fashion_mnist'],help="Dataset to use")
    parser.add_argument('-e', '--epochs', type=int, default=1,help="Number of epochs to train neural network")
    parser.add_argument('-b', '--batch_size', type=int, default=4,help="Batch size used to train neural network")
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',choices=['mean_squared_error', 'cross_entropy','compare',],help="Loss function type")
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],help="Optimization algorithm")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,help="Learning rate used to optimize model parameters")
    parser.add_argument('-m', '--momentum', type=float, default=0.5,help="Momentum used by momentum and nag optimizers")
    parser.add_argument('-beta', '--beta', type=float, default=0.5,help="Beta used by rmsprop optimizer")
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,help="Epsilon used by optimizers")
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,help="Weight decay used by optimizers")
    parser.add_argument('-w_i', '--weight_init', type=str, default='random',choices=['random', 'Xavier'],help="Weight initialization method")
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,help="Number of hidden layers used in feedforward neural network")
    parser.add_argument('-sz', '--hidden_size', type=str, default='4',help="Number of hidden neurons in feedforward layers (comma-separated for different sizes)")
    parser.add_argument('-a', '--activation', type=str, default='sigmoid',choices=['identity', 'sigmoid', 'tanh', 'ReLU'],help="Activation function for hidden layers")
    parser.add_argument('--val_split', type=float, default=0.1,help="Validation split ratio from training data")
    parser.add_argument('--run_sweep', action='store_true',help="Run hyperparameter sweep instead of single training")
    parser.add_argument('--sweep_count', type=int, default=75,help='Number of sweep runs to execute (default: 3)')
    
    return parser.parse_args()

def build_train_config_from_args(args):
    
    #Creates a TrainConfig object from parsed arguments.
    config = TrainConfig()    
    # Copy all parameters from args
    config.wandb_entity = args.wandb_entity
    config.wandb_project = args.wandb_project
    config.num_layers = args.num_layers
    config.activation = args.activation.lower()  
    config.optimizer = args.optimizer
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.momentum = args.momentum
    config.beta = args.beta
    config.beta1 = args.beta1
    config.beta2 = args.beta2
    config.epsilon = args.epsilon
    config.weight_decay = args.weight_decay
    config.weight_init = args.weight_init.lower()  
    config.val_split = args.val_split
    config.dataset = args.dataset
    
    # Handle hidden_size parameter - can be single value or comma-separated list
    if ',' in args.hidden_size:
        config.hidden_layers = [int(size) for size in args.hidden_size.split(',')]
        if len(config.hidden_layers) != args.num_layers:
            print(f"Warning: Number of specified hidden sizes ({len(config.hidden_layers)}) "
                  f"doesn't match num_layers ({args.num_layers}). Using the specified sizes.")
            config.num_layers = len(config.hidden_layers)
    else:
        size = int(args.hidden_size)
        config.hidden_size = size
        config.hidden_layers = [size] * args.num_layers
    config.hidden_layers_str = ','.join(str(x) for x in config.hidden_layers)
    
    return config

def define_sweep_config():
    """Defines configuration for hyperparameter sweep using WandB."""
    return {
        'method': 'bayes',  
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'dataset': {'values': ['fashion_mnist', 'mnist']},
            'num_layers': {'values': [1, 2, 3, 4, 5]},
            'hidden_size': {'values': ['32', '64', '128', '32,16', '64,32', '128,64,32']},
            'activation': {'values': ['identity', 'sigmoid', 'tanh', 'relu']},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'learning_rate': {'values': [0.1, 0.01, 0.001, 0.0001]},
            'momentum': {'values': [0.0, 0.5, 0.9]},
            'beta': {'values': [0.9, 0.95, 0.99]},
            'beta1': {'values': [0.9, 0.95]},
            'beta2': {'values': [0.999, 0.9999]},
            'epsilon': {'value': 1e-8},
            'batch_size': {'values': [16, 32, 64, 128]},
            'epochs': {'values': [5, 10, 15]},
            'weight_decay': {'values': [0, 0.0001, 0.001, 0.01]},
            'weight_init': {'values': ['random', 'xavier']},
            'loss': {'values': ['cross_entropy', 'mean_squared_error']},
            'val_split': {'value': 0.1}
        }
    }