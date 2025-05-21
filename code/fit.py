import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from forward_loop import *

def fit(model, experiment_name, num_epochs, optimizer, pos_weight, l1_lambda, 
        train_loader, val_loader, device, max_norm, target_outcome,training_mode='combined',
        lr_patience=5,patience=100):
    
    # Use the current working directory and create a Models subdirectory
    base_dir = os.getcwd()
    models_dir = os.path.join(base_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Define file paths relative to the current working directory
    checkpoint_path = os.path.join(models_dir, f"{experiment_name}.pth")
    metrics_csv_path = os.path.join(models_dir, f"{experiment_name}.csv")

    # Load model if checkpoint exists
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded saved model weights from checkpoint.")
        metric_df = pd.read_csv(metrics_csv_path)
    else:
        metric_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_auprc", "val_auroc"])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=0.33,
        patience=lr_patience, 
        threshold=0.01,
        threshold_mode='rel'
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train(model=model,
                           train_loader=train_loader,
                           optimizer=optimizer,
                           pos_weight=pos_weight,
                           l1_lambda=l1_lambda,
                           device=device,
                           max_norm=max_norm,
                           target_outcome=target_outcome,
                           training_mode = training_mode)

        val_loss, val_auprc, val_auroc = validate(model,
                                                  val_loader,
                                                  device=device,
                                                  pos_weight=pos_weight,
                                                  target_outcome = target_outcome,
                                                  training_mode = training_mode)

        print(f"Validation AUPRC: {val_auprc:.4f}, Validation AUROC: {val_auroc:.4f}")

        # Save the main model checkpoint at the end of each epoch
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model weights saved for epoch {epoch+1}")

        # Step the scheduler using training loss
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr:.6f}")

        # Save metrics to CSV
        epoch_df = pd.DataFrame({
            "epoch": [epoch],
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "val_auprc": [val_auprc],
            "val_auroc": [val_auroc]
        })
        metric_df = pd.concat([metric_df, epoch_df], ignore_index=True)
        metric_df.to_csv(metrics_csv_path, index=False)

    print(f"Metrics saved to {metrics_csv_path}")
    return model