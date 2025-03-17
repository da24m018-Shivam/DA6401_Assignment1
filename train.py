import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import wandb
from utils import preprocess_data, create_mini_batches, compute_accuracy, plot_confusion_matrix, plot_sample_images
from trainer import train_model
from config import parse_args, define_sweep_config
from sweeps import sweep_agent
from experiments import compare_loss_functions
import json
import os

def main():
    args = parse_args()  # Parse command-line arguments
    print(args)
    wandb.login()  # Authenticate WandB

    if args.run_sweep:
        # Run hyperparameter sweep
        sweep_config = define_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        
        # Run sweep agent and wait for completion
        wandb.agent(sweep_id, function=lambda: sweep_agent(args), count=args.sweep_count)
        
        # After sweep is complete, read the best model info
        with open("sweep_results/best_model_info.json", "r") as f:
            best_model_info = json.load(f)
        
        # Print the information to console
        print(f"\nBest model from sweep:")
        print(f"Run ID: {best_model_info['run_id']}")
        print(f"Sweep ID: {best_model_info.get('sweep_id', 'N/A')}")
        print(f"Test Accuracy: {best_model_info['test_accuracy']:.2f}%")
        print(f"Dataset: {best_model_info.get('dataset', 'N/A')}")
        print(f"Configuration: {best_model_info['config']}")
        
        # Only create summary run if dataset is fashion_mnist
        if best_model_info.get('dataset') == 'fashion_mnist':
            # Create a summary run to log artifacts
            with wandb.init(project=args.wandb_project, entity=args.wandb_entity) as run:
                
                # Log the best model configuration as an artifact
                config_artifact = wandb.Artifact(
                    name=f"best-model-config-{sweep_id[:8]}",
                    type="model-info",
                    description=f"Best model configuration from sweep {sweep_id}"
                )
                
                # Add the best model info JSON file to the artifact
                config_artifact.add_file("sweep_results/best_model_info.json")
                
                # Add the confusion matrix image to the artifact
                config_artifact.add_file("sweep_results/best_model_confusion_matrix.png")
                
                # Log the artifact
                run.log_artifact(config_artifact)
                
                # Also log the confusion matrix as media for easy viewing
                confusion_matrix_img = wandb.Image(
                    "sweep_results/best_model_confusion_matrix.png", 
                    caption=f"Confusion Matrix for Best Model (Acc: {best_model_info['test_accuracy']:.2f}%)"
                )
                run.log({"best_model_confusion_matrix": confusion_matrix_img})
                
                # Log the best model parameters as an artifact
                params_artifact = wandb.Artifact(
                    name=f"best-model-params-{sweep_id[:8]}",
                    type="model-weights",
                    description=f"Best model parameters from sweep {sweep_id}"
                )
                
                # Add the model parameters file to the artifact
                params_artifact.add_file("sweep_results/best_model_params.npy")
                
                # Log the artifact
                run.log_artifact(params_artifact)
                
                # Log the sweep info and best model metrics
                run.log({
                    "best_test_accuracy": best_model_info['test_accuracy'],
                    "best_run_id": best_model_info['run_id'],
                    "sweep_id": sweep_id,
                    "total_runs": args.sweep_count,
                    "dataset": best_model_info.get('dataset', 'N/A')
                })
                
                # Add a link to the best run
                best_run_link = f"{run.get_project_url()}/runs/{best_model_info['run_id']}"
                run.notes = f"Best run: [{best_model_info['run_id']}]({best_run_link}) with accuracy {best_model_info['test_accuracy']:.2f}% on {best_model_info.get('dataset', 'N/A')}"
                
                print(f"Artifacts and results uploaded to WandB project: {args.wandb_project}")
                print(f"Summary run: {run.name} (ID: {run.id})")
        else:
            print(f"Model artifacts not saved - dataset is {best_model_info.get('dataset', 'N/A')}, not fashion_mnist")
    
    elif args.loss == 'compare':
        # Compare different loss functions
        compare_loss_functions(args)
    
    else:
        # Standard training process
        train_model(args, dataset_type=args.dataset)


if __name__ == "__main__":
    main()