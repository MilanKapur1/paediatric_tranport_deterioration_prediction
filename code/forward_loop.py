import torch
import torch.nn as nn  # Neural network layers and loss functions
import torch.optim as optim  # Optimization algorithms
import time
from sklearn.metrics import precision_recall_curve, roc_auc_score,auc


# Assume your model, criterion, optimizer, and DataLoaders are already set up

# Training function
# Training function
def train(model, train_loader, optimizer, pos_weight, device, l1_lambda,max_norm, target_outcome,training_mode = 'combined'):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optimizer

    start_epoch_time = time.time()  # Start timing the epoch

    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()  # Start timing the batch

        # Unpack and move data to device
        (
            feature_tensor,
            missingness_mask_tensor,
            padding_mask,
            static_tensor,
            scalar_tensor,
            list_tensor,
            label_tensor,
            cats_id,
            start_min,
            end_min,
            prediction_window_length
        ) = batch

        # Move tensors to device
        feature_tensor = feature_tensor.to(device)
        missingness_mask_tensor = missingness_mask_tensor.to(device)
        scalar_tensor = scalar_tensor.to(device)
        list_tensor = list_tensor.to(device)
        static_tensor = static_tensor.to(device)
       
        if target_outcome =='resp':
            label_tensor = label_tensor[:, 0].unsqueeze(1)       #set to 0 if training resp and 1 if training cardiac
        elif target_outcome == 'cardiac':
            label_tensor = label_tensor[:, 1].unsqueeze(1)       #set to 0 if training resp and 1 if training cardiac
        
        
        label_tensor = label_tensor.to(device)  # Use only the first item in each label
        optimizer.zero_grad()

        # Forward pass
        # Get model output
        
        if training_mode == 'simpler_embeddings':
            
            output = model(
                dynamic_features=feature_tensor,
                static_features=static_tensor,
                missingness_mask=missingness_mask_tensor
            )
            
        
        elif training_mode == 'combined':
            output = model(
                dynamic_features=feature_tensor,
                missingness_mask = missingness_mask_tensor,
                scalar_features=scalar_tensor,
                embedding_features=list_tensor
            )
        
        
        elif training_mode =='dynamic':
            output = model(
                dynamic_features=feature_tensor,
                missingness_mask = missingness_mask_tensor,
            )
        
        elif training_mode == 'demographic':
            output = model(
                scalar_features=scalar_tensor,
                embedding_features=list_tensor
            )
        
        
        # Compute the main loss (BCE with logits)
        main_loss = criterion(output, label_tensor) # Use .squeeze() to match dimensions if necessary        
        
        
        # Add L1 regularization by summing up the absolute values of all parameters
        l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
        # Combine main loss and L1 regularization loss
        loss = main_loss + l1_lambda * l1_loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        # After backward pass
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=max_norm)
        optimizer.step()
        
        # Track time per batch
        #batch_time = time.time() - batch_start_time
        #print(f"Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")


    # Calculate epoch time
    epoch_time = time.time() - start_epoch_time
    avg_loss = total_loss / num_batches
    print(f"Epoch completed in {epoch_time:.2f}s, Average Training Loss: {avg_loss:.4f}")

    return avg_loss



# Validation function
def validate(model, val_loader, device,pos_weight,target_outcome, training_mode = 'combined'):
    
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0
    num_batches = len(val_loader)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Unpack and move data to device
            (
                feature_tensor,
                missingness_mask_tensor,
                padding_mask,
                static_tensor,
                scalar_tensor,
                list_tensor,
                label_tensor,
                cats_id,
                start_min,
                end_min,
                prediction_window_length
            ) = batch
    
            # Move tensors to device
            feature_tensor = feature_tensor.to(device)
            missingness_mask_tensor = missingness_mask_tensor.to(device)
            scalar_tensor = scalar_tensor.to(device)
            list_tensor = list_tensor.to(device)
            static_tensor = static_tensor.to(device)
           
            if target_outcome =='resp':
                label_tensor = label_tensor[:, 0].unsqueeze(1)       #set to 0 if training resp and 1 if training cardiac
            elif target_outcome == 'cardiac':
                label_tensor = label_tensor[:, 1].unsqueeze(1)       #set to 0 if training resp and 1 if training cardiac
            
            label_tensor = label_tensor.to(device)  # Use only the first item in each label
            

            if training_mode == 'simpler_embeddings':
                
                output = model(
                    dynamic_features=feature_tensor,
                    static_features=static_tensor,
                    missingness_mask=missingness_mask_tensor
                )
                
            
            elif training_mode == 'combined':
                output = model(
                    dynamic_features=feature_tensor,
                    missingness_mask = missingness_mask_tensor,
                    scalar_features=scalar_tensor,
                    embedding_features=list_tensor
                )
            
            
            elif training_mode =='dynamic':
                output = model(
                    dynamic_features=feature_tensor,
                    missingness_mask = missingness_mask_tensor,
                )
            
            elif training_mode == 'demographic':
                output = model(
                    scalar_features=scalar_tensor,
                    embedding_features=list_tensor
                )
                
                
                
            # Compute loss using only the first label
            loss = criterion(output, label_tensor)  # Use .squeeze() if necessary
            total_loss += loss.item()

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(output).cpu().squeeze()

            # Collect probabilities and true labels for AUC-PR calculation
            all_probs.append(probs)
            all_labels.append(label_tensor.cpu())

    # Calculate average loss
    avg_loss = total_loss / num_batches

    # Calculate AUC-PR for the first label
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    
    
    # Calculate AUC-PR
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall, precision)

    # Calculate AUROC
    auc_roc = roc_auc_score(all_labels, all_probs)

    print(f"Validation AUC-PR for resp: {auc_pr:.4f}")
    print(f"Validation AUROC for resp: {auc_roc:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss, auc_pr, auc_roc  # Return AUROC if needed