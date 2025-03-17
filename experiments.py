from config import TrainConfig
from trainer import train_model

def compare_loss_functions(args):
    """Compares cross-entropy and squared error loss functions."""
    
    from config import build_train_config_from_args
    
    base_config = build_train_config_from_args(args)
    
    print("Training with Cross Entropy Loss...")
    base_config.loss = 'cross_entropy'
    ce_accuracy = train_model(base_config, dataset_type=args.dataset)

    print("Training with Squared Error Loss...")
    base_config.loss = 'mean_squared_error'
    se_accuracy = train_model(base_config, dataset_type=args.dataset)

    print(f"Cross Entropy Test Accuracy: {ce_accuracy:.2f}%")
    print(f"Squared Error Test Accuracy: {se_accuracy:.2f}%")
    print(f"Difference: {abs(ce_accuracy - se_accuracy):.2f}%")

