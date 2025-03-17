# DA6401_Assignment1
This project implements feedforward neural networks from scratch using NumPy, with support for various activation functions, optimizers, and hyperparameter tuning capabilities.
Github Link : https://github.com/da24m018-Shivam/DA6401_Assignment1 

Report Link : https://wandb.ai/shivam-da24m018-iitmaana/DA6401_Assignment1/reports/DA6401_Assignment1--VmlldzoxMTgzOTk0OA

## Features

- Build and train multilayer neural networks from scratch
- Support for various activation functions (ReLU, Sigmoid, Tanh, Identity)
- Implementation of multiple optimizers (SGD, Momentum, NAG, RMSProp, Adam, NAdam)
- Training on MNIST and Fashion-MNIST datasets
- Hyperparameter tuning with Weights & Biases (WandB)
- Saving and tracking best models
- Performance evaluation with confusion matrices

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- TensorFlow (for loading datasets)
- scikit-learn
- seaborn
- Weights & Biases (WandB)

### Setup

1. Clone the repository:

2. Install the required packages:
   ```bash
   pip install numpy matplotlib tensorflow scikit-learn seaborn wandb
   ```

3. Sign up for a free WandB account at [wandb.ai](https://wandb.ai) if you don't have one

4. Login to WandB:
   ```bash
   wandb login
   ```

## Project Structure

- `activation.py`: Implementation of activation functions
- `config.py`: Configuration parsing and management
- `experiments.py`: Pre-defined experiments for comparing techniques
- `model.py`: Neural network implementation
- `optimizers.py`: Implementation of optimization algorithms
- `sweeps.py`: Hyperparameter sweep functionality
- `train.py`: Main training script
- `trainer.py`: Core training logic
- `utils.py`: Helper functions for data processing and visualization

## Basic Usage

### Training a Simple Model

To train a neural network with default parameters on Fashion-MNIST:

```bash
python train.py --wandb_entity your_wandb_username --wandb_project your_project_name
```

### Customizing Your Model

You can customize various aspects of your model:

```bash
python train.py --hidden_size 128 --num_layers 2 --activation relu --optimizer adam --learning_rate 0.001 --batch_size 32 --epochs 10
```

### Full List of Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-wp`, `--wandb_project` | WandB project name | `myprojectname` |
| `-we`, `--wandb_entity` | WandB entity name | `myname` |
| `-d`, `--dataset` | Dataset choice (mnist, fashion_mnist) | `fashion_mnist` |
| `-e`, `--epochs` | Number of training epochs | `1` |
| `-b`, `--batch_size` | Batch size | `4` |
| `-l`, `--loss` | Loss function type | `cross_entropy` |
| `-o`, `--optimizer` | Optimization algorithm | `sgd` |
| `-lr`, `--learning_rate` | Learning rate | `0.1` |
| `-m`, `--momentum` | Momentum for momentum/nag optimizers | `0.5` |
| `-beta`, `--beta` | Beta for RMSprop | `0.5` |
| `-beta1`, `--beta1` | Beta1 for Adam/NAdam | `0.5` |
| `-beta2`, `--beta2` | Beta2 for Adam/NAdam | `0.5` |
| `-eps`, `--epsilon` | Small constant for numerical stability | `0.000001` |
| `-w_d`, `--weight_decay` | L2 regularization parameter | `0.0` |
| `-w_i`, `--weight_init` | Weight initialization method | `random` |
| `-nhl`, `--num_layers` | Number of hidden layers | `1` |
| `-sz`, `--hidden_size` | Number of neurons in hidden layers | `4` |
| `-a`, `--activation` | Activation function | `sigmoid` |
| `--val_split` | Validation split ratio | `0.1` |
| `--run_sweep` | Run hyperparameter sweep | Not enabled by default |
| `--sweep_count` | Number of sweep runs | `75` |

## Advanced Usage

### Running Hyperparameter Sweeps

Run a hyperparameter sweep to find optimal configurations:

```bash
python train.py --run_sweep --sweep_count 20 --wandb_entity your_wandb_username --wandb_project your_project_name
```

This will:
1. Start a Bayesian hyperparameter search
2. Try different combinations of parameters
3. Track the best model across all runs
4. Save artifacts for the best model

### Comparing Loss Functions

Compare cross-entropy and mean squared error loss functions:

```bash
python train.py --loss compare --wandb_entity your_wandb_username --wandb_project your_project_name
```

### Recommended MNIST Configurations

Try pre-defined recommended configurations on MNIST:

```bash
python train.py --dataset mnist --loss mnist_recommendations --wandb_entity your_wandb_username --wandb_project your_project_name
```

### Custom Network Architecture

You can specify different sizes for each hidden layer by providing comma-separated values:

```bash
python train.py --hidden_size 128,64,32 --num_layers 3
```

This creates a network with three hidden layers of sizes 128, 64, and 32 respectively.

## Understanding the Output

### Training Metrics

During training, you'll see output like:

```
Epoch 1/10 - Train Loss: 0.5623, Train Acc: 78.25%, Val Loss: 0.4872, Val Acc: 82.56%
```

This shows:
- Current epoch
- Training loss and accuracy
- Validation loss and accuracy

### Best Model Tracking

After running a hyperparameter sweep, the system will output:

```
Best model from sweep:
Run ID: abc123
Sweep ID: def456
Test Accuracy: 92.45%
Dataset: fashion_mnist
Configuration: {'input_size': 784, 'hidden_layers': [128, 64], ...}
```

### Artifacts

The system saves the following artifacts in the `sweep_results` directory:
- `best_model_info.json`: Configuration of the best model
- `best_model_params.npy`: Parameters (weights and biases) of the best model
- `best_model_confusion_matrix.png`: Confusion matrix visualization

These artifacts are also uploaded to WandB for easy access and visualization.

## Weights & Biases Integration

This project uses WandB for:
- Experiment tracking
- Hyperparameter optimization
- Artifact storage
- Visualization of results

After running experiments, you can view comprehensive dashboards with:
- Learning curves (loss and accuracy)
- Sample images from the dataset
- Confusion matrices
- Hyperparameter importance

Visit your WandB dashboard at `https://wandb.ai/your_username/your_project_name` to explore results.

## Example Use Cases

### Basic Training

```bash
python train.py --hidden_size 64 --num_layers 2 --activation relu --optimizer adam --learning_rate 0.001 --batch_size 32 --epochs 10
```

### Finding Optimal Configurations

```bash
python train.py --run_sweep --sweep_count 30 --dataset fashion_mnist
```

### Performance Comparison

```bash
python train.py --hidden_size 128 --num_layers 3 --activation relu --optimizer adam --dataset mnist --epochs 15
```
