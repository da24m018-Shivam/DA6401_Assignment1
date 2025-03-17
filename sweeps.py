import wandb
import numpy as np
import os
import json
from trainer import train_model
from config import TrainConfig
from utils import preprocess_data, plot_confusion_matrix
import matplotlib.pyplot as plt

class BestModelTracker:
    """Tracks the best model across all sweep runs."""
    
    def __init__(self):
        self.best_run_id = None
        self.best_sweep_id = None
        self.best_sweep_name = None
        self.best_accuracy = 0
        self.best_model = None
        self.best_config = None
        
        # Create directory to store best model info
        os.makedirs("sweep_results", exist_ok=True)
        self.tracking_file = "sweep_results/best_model_info.json"
        
        # Load existing best model info if available
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, "r") as f:
                info = json.load(f)
                self.best_run_id = info.get("run_id")
                self.best_sweep_id = info.get("sweep_id")
                self.best_sweep_name = info.get("sweep_name")
                self.best_accuracy = info.get("test_accuracy", 0)
                self.best_dataset = info.get("dataset")
                print(f"Loaded previous best model with accuracy: {self.best_accuracy:.2f}%")
                if self.best_sweep_name:
                    print(f"Best model from sweep: {self.best_sweep_name} (ID: {self.best_sweep_id})")
                if self.best_dataset:
                    print(f"Dataset: {self.best_dataset}")
    
    def update(self, run_id, test_accuracy, model, config, sweep_id=None, sweep_name=None, dataset_type=None):
        """Update best model if current model is better and is Fashion MNIST dataset."""
        print(f"Comparing accuracy {test_accuracy:.2f}% with best {self.best_accuracy:.2f}%")
        
        # Only update if dataset is fashion_mnist
        if dataset_type != 'fashion_mnist':
            print(f"✗ Dataset is {dataset_type}, not fashion_mnist. Model not saved.")
            return False
        
        if test_accuracy > self.best_accuracy:
            self.best_run_id = run_id
            self.best_sweep_id = sweep_id
            self.best_sweep_name = sweep_name
            self.best_accuracy = test_accuracy
            self.best_model = model
            self.best_config = config
            self.best_dataset = dataset_type
            
            # Save best model info
            with open(self.tracking_file, "w") as f:
                info = {
                    "run_id": run_id,
                    "sweep_id": sweep_id,
                    "sweep_name": sweep_name,
                    "test_accuracy": test_accuracy,
                    "config": config,
                    "dataset": dataset_type
                }
                json.dump(info, f, default=str)
            
            # Save best model parameters
            np.save("sweep_results/best_model_params.npy", model.parameters)
            
            print(f"✓ Updated best model: run_id={run_id}, accuracy={test_accuracy:.2f}%")
            if sweep_name:
                print(f"✓ From sweep: {sweep_name} (ID: {sweep_id})")
            return True
        
        print(f"✗ Current model not better than best")
        return False

# Create a singleton instance of the tracker
best_model_tracker = BestModelTracker()

def sweep_agent(args):
    """Runs a hyperparameter sweep using WandB configurations."""
    global best_model_tracker

    wandb.init()  
    wandb_config = wandb.config  
    train_config = TrainConfig()
    train_config.wandb_project = args.wandb_project
    train_config.wandb_entity = args.wandb_entity
    dataset_type = wandb_config.get('dataset', args.dataset)
    
    # Get sweep information
    sweep_id = wandb.run.sweep_id if hasattr(wandb.run, 'sweep_id') else None
    sweep_name = None
    if sweep_id:
        try:
            # Try to get the sweep name from the API
            api = wandb.Api()
            sweep = api.sweep(f"{train_config.wandb_entity}/{train_config.wandb_project}/{sweep_id}")
            sweep_name = sweep.name
        except:
            # If API call fails, use the sweep ID as the name
            sweep_name = sweep_id
    
    if 'num_layers' in wandb_config and 'hidden_size' in wandb_config:
        hidden_size = wandb_config.get('hidden_size')
        num_layers = wandb_config.get('num_layers')
        
        if ',' in hidden_size:
            train_config.hidden_layers = [int(size) for size in hidden_size.split(',')]
            train_config.num_layers = len(train_config.hidden_layers)
        else:
            size = int(hidden_size)
            train_config.hidden_size = size
            train_config.hidden_layers = [size] * num_layers
            train_config.num_layers = num_layers
        train_config.hidden_layers_str = ','.join(str(x) for x in train_config.hidden_layers)
    else:
        # Use args as fallback
        train_config.hidden_layers = args.hidden_layers
        train_config.num_layers = args.num_layers
    
    train_config.activation = wandb_config.get('activation', args.activation).lower()
    train_config.optimizer = wandb_config.get('optimizer', args.optimizer)
    train_config.learning_rate = wandb_config.get('learning_rate', args.learning_rate)
    train_config.momentum = wandb_config.get('momentum', args.momentum)
    train_config.beta = wandb_config.get('beta', args.beta)
    train_config.beta1 = wandb_config.get('beta1', args.beta1)
    train_config.beta2 = wandb_config.get('beta2', args.beta2)
    train_config.epsilon = wandb_config.get('epsilon', args.epsilon)
    train_config.batch_size = wandb_config.get('batch_size', args.batch_size)
    train_config.epochs = wandb_config.get('epochs', args.epochs)
    train_config.weight_decay = wandb_config.get('weight_decay', args.weight_decay)
    train_config.weight_init = wandb_config.get('weight_init', args.weight_init).lower()
    train_config.val_split = wandb_config.get('val_split', args.val_split)
    train_config.loss_type = wandb_config.get('loss', args.loss)
    
    # Train model with return_model=True to get the model back
    test_accuracy, model, X_test, y_test, y_pred = train_model(train_config, dataset_type=dataset_type, return_model=True)
    
    # Update best model tracker with sweep information and dataset type
    is_best = best_model_tracker.update(
        wandb.run.id, 
        test_accuracy, 
        model, 
        model.get_config(),
        sweep_id=sweep_id,
        sweep_name=sweep_name,
        dataset_type=dataset_type  # Pass dataset_type here
    )
    
    # If this is the best model so far, generate and log confusion matrix
    if is_best:
        print(f"New best model found! Run ID: {wandb.run.id}, Test accuracy: {test_accuracy:.2f}%")
        if sweep_name:
            print(f"From sweep: {sweep_name} (ID: {sweep_id})")
        
        # Get class names based on dataset type
        from config import FASHION_MNIST_CLASSES
        class_names = FASHION_MNIST_CLASSES if dataset_type == 'fashion_mnist' else [str(i) for i in range(10)]
        
        # Generate confusion matrix and save to file
        cm_fig = plot_confusion_matrix(y_test, y_pred, class_names)
        cm_fig.savefig(f"sweep_results/best_model_confusion_matrix.png")
        plt.close(cm_fig)  # Close the figure to free memory
        
        print(f"Confusion matrix saved as sweep_results/best_model_confusion_matrix.png")
        
        # Add a tag to mark this as currently the best model
        wandb.run.tags = wandb.run.tags + ("current_best_model",)
    
    return test_accuracy