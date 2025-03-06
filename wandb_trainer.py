import wandb
import numpy as np
from trainer import Trainer
from utils import compute_accuracy, create_mini_batches

class WandbTrainer(Trainer):
    def __init__(self, model, optimizer=None):
        super().__init__(model, optimizer)
    
    def train(self, X_train, Y_train, X_val, Y_val, y_train, y_val, 
              num_epochs, batch_size, print_every=1):
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            
            # Create mini-batches
            mini_batches = create_mini_batches(X_train, Y_train, batch_size)
            num_batches = len(mini_batches)
            
            # Training loop
            for batch_idx, (mini_batch_X, mini_batch_Y) in enumerate(mini_batches):
                # Forward pass
                AL = self.model.forward_propagation(mini_batch_X)
                
                # Compute cost
                batch_cost = self.model.compute_cost(AL, mini_batch_Y)
                epoch_cost += batch_cost / num_batches
                
                # Backward pass
                grads = self.model.backward_propagation(mini_batch_Y)
                
                # Update parameters
                if self.optimizer:
                    params = self.optimizer.update(self.model.parameters, grads)
                    self.model.parameters = params
                else:
                    self.model.update_parameters(grads)
            
            # Compute metrics on training and validation sets
            train_preds = self.model.predict(X_train)
            val_preds = self.model.predict(X_val)
            
            train_accuracy = compute_accuracy(train_preds, y_train)
            val_accuracy = compute_accuracy(val_preds, y_val)
            
            # Compute validation loss
            AL_val = self.model.forward_propagation(X_val)
            val_cost = self.model.compute_cost(AL_val, Y_val)
            
            # Store metrics
            self.train_losses.append(epoch_cost)
            self.val_losses.append(val_cost)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_cost,
                "val_loss": val_cost,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            })
            
            # Print metrics
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train loss: {epoch_cost:.4f}, "
                      f"Val loss: {val_cost:.4f}, "
                      f"Train accuracy: {train_accuracy:.2f}%, "
                      f"Val accuracy: {val_accuracy:.2f}%")