import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Decoder
from torch.utils.data import DataLoader
import os
import torch.optim as optim  # Optimization algorithms
from fit import fit  # Custom function for model training
import re

class DynamicModel(nn.Module):
    def __init__(self, seq_len, feature_dim, hidden_dim, batch_norm=False):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        # Since we are concatenating dynamic_features and missingness_mask,
        # the input dimension becomes seq_len * (feature_dim * 2)
        input_dim = seq_len * (feature_dim * 2)
        self.flatten = nn.Flatten()
        
        # Build the MLP with optional batch normalization
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ])
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, dynamic_features, missingness_mask):
        # Both dynamic_features and missingness_mask have shape:
        # (batch_size, seq_len, feature_dim)
        # Concatenate them along the last dimension to get shape:
        # (batch_size, seq_len, feature_dim * 2)
        features_with_missing_mask = torch.cat([dynamic_features, missingness_mask], dim=-1)
        
        # Flatten the tensor to shape (batch_size, seq_len * feature_dim * 2)
        x = self.flatten(features_with_missing_mask)
        
        # Pass through the MLP
        x = self.mlp(x)
        return x

def load_dynamic_model(dynamic_dim_in, hidden_dim, max_seq_len, batch_norm=False):
    model = DynamicModel(
        seq_len=max_seq_len,
        feature_dim=dynamic_dim_in,
        hidden_dim=hidden_dim,
        batch_norm=batch_norm
    )
    return model



class StaticModel(nn.Module):
    def __init__(self, scalar_input_dim, embedding_input_dim, scalar_mlp_hidden_dim, embedding_hidden_dim, batch_norm):
        super().__init__()
        
        self.use_bn = batch_norm
    
        # MLP for scalar data
        if self.use_bn:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(scalar_input_dim, scalar_mlp_hidden_dim),
                nn.BatchNorm1d(scalar_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(scalar_mlp_hidden_dim, scalar_mlp_hidden_dim),
                nn.BatchNorm1d(scalar_mlp_hidden_dim),
                nn.ReLU()
            )
        else:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(scalar_input_dim, scalar_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(scalar_mlp_hidden_dim, scalar_mlp_hidden_dim),
                nn.ReLU()
            )
        
        # MLP for embedding data
        if self.use_bn:
            self.embedding_mlp = nn.Sequential(
                nn.Linear(embedding_input_dim, embedding_hidden_dim),
                nn.BatchNorm1d(embedding_hidden_dim),
                nn.ReLU(),
                nn.Linear(embedding_hidden_dim, embedding_hidden_dim),
                nn.BatchNorm1d(embedding_hidden_dim),
                nn.ReLU()
            )
        else:
            self.embedding_mlp = nn.Sequential(
                nn.Linear(embedding_input_dim, embedding_hidden_dim),
                nn.ReLU(),
                nn.Linear(embedding_hidden_dim, embedding_hidden_dim),
                nn.ReLU()
            )
            
            
    def forward(self, scalar_features, embedding_features):
        # Process scalar features through MLP
        scalar_output = self.scalar_mlp(scalar_features)  # Shape: (batch_size, scalar_mlp_hidden_dim)
        
        # Process embedding features through MLP
        embedding_output = self.embedding_mlp(embedding_features)  # Shape: (batch_size, embedding_hidden_dim)
        
        # Combine the outputs
        combined_features = torch.cat([scalar_output, embedding_output], dim=-1)      
        
        

        return combined_features





def load_static_model(scalar_input_dim, scalar_mlp_hidden_dim, embedding_hidden_dim,batch_norm):
    
        # Define model parameters
    embedding_input_dim = 768 # Example static feature dimension

    # Instantiate the combined model
    model = StaticModel(
        scalar_input_dim=scalar_input_dim,
        embedding_input_dim=embedding_input_dim,
        scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
        embedding_hidden_dim=embedding_hidden_dim,
        batch_norm=batch_norm
    )        
    return model




class EnsembleModel(nn.Module):
    def __init__(self, dynamic_model, static_model, ensemble_mlp_hidden_dim, output_dim,batch_norm,scalar_mlp_hidden_dim,embedding_hidden_dim,dynamic_dim_out):
        super().__init__()  # Corrected super() call
        self.dynamic_model = dynamic_model
        self.static_model = static_model
        self.use_bn = batch_norm
        self.mlp_dim = dynamic_dim_out + scalar_mlp_hidden_dim + embedding_hidden_dim
        self.mlp_hidden_dim = ensemble_mlp_hidden_dim
        
        if self.use_bn:
            self.mlp = nn.Sequential(
                nn.Linear(self.mlp_dim, self.mlp_hidden_dim),
                nn.BatchNorm1d(self.mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_dim, output_dim)
                
            )
        # MLP for combining logits
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.mlp_dim, self.mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_dim, output_dim)
                
            )
    
    def forward(self, dynamic_features, missingness_mask, scalar_features, embedding_features):
        # Get logits from pretrained models
        dynamic_output = self.dynamic_model(
            dynamic_features=dynamic_features,
            missingness_mask=missingness_mask
        )  # Shape: (batch_size, 1)
        
        static_output = self.static_model(
            scalar_features=scalar_features,
            embedding_features=embedding_features
        )  # Shape: (batch_size, 1)
        
        # Concatenate logits
        combined_penultimate = torch.cat([static_output, dynamic_output], dim=1)  # Shape: (batch_size, 2)
        
        # Pass through the MLP
        output = self.mlp(combined_penultimate )  # Shape: (batch_size, output_dim)
        return output

###############################################################
####usage

def load_model(model_path,
            dynamic_dim_in=18,
            hidden_dim=64,
            batch_norm=False,
            scalar_input_dim=40,
            scalar_mlp_hidden_dim=256,
            embedding_hidden_dim=256,
            ensemble_mlp_hidden_dim=512,
            output_dim=1,
            max_seq_len=924,
            device="cpu"):
    """
    Run an experiment with the specified parameters.
    
    Args:
        dynamic_model_loader (callable): Function to load the dynamic model.
        static_model_loader (callable): Function to load the static model.
        ensemble_model_class (class): The class for the ensemble model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        experiment_name (str): Name of the experiment.
        mlp_hidden_dim (int): Hidden layer dimension for the ensemble MLP.
        output_dim (int): Output dimension of the model.
        l1_lambda (float): L1 regularization weight.
        l2_lambda (float): L2 regularization weight (weight decay).
        lr (float): Learning rate for the optimizer.
        pos_weight_value (float): Weight for the positive class in loss calculation.
        epochs (int): Number of training epochs.
        max_norm (float): Gradient clipping max norm value.
        patience (int): Early stopping patience.
        lr_patience (int): Learning rate scheduler patience.
        device (str or torch.device): Device to run the training (e.g., "cpu" or "cuda").
    
    Returns:  
        model: Trained model.
    """
    # Load pretrained models
    dynamic_model = load_dynamic_model(dynamic_dim_in=dynamic_dim_in, hidden_dim = hidden_dim,max_seq_len=max_seq_len, batch_norm=False)

    static_model = load_static_model(scalar_input_dim=scalar_input_dim,
                                     scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
                                     embedding_hidden_dim=embedding_hidden_dim,
                                     batch_norm=batch_norm)



    # Define ensemble model
    model = EnsembleModel( dynamic_model, static_model, ensemble_mlp_hidden_dim, output_dim,batch_norm,scalar_mlp_hidden_dim,embedding_hidden_dim,hidden_dim)
    model = torch.compile(model)
    model.to(device)
    
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only = True))
        print("Loaded saved model weights.")
        
    return model


def run_exp(experiment_name,
            train_loader,
            val_loader,

            dynamic_dim_in=18,
            max_seq_len=924,
            hidden_dim=64,

            batch_norm=False,
            scalar_input_dim=40,
            scalar_mlp_hidden_dim=256,
            embedding_hidden_dim=256,
            ensemble_mlp_hidden_dim=512,
            output_dim=1,
            l1_lambda=0,
            l2_lambda=0,
            lr=1e-5,
            pos_weight_value=30.0,
            epochs=5,
            max_norm=10,
            patience=100,
            lr_patience=100,
            device="cpu",
            target_outcome = 'cardiac'
            ):
    """
    Run an experiment with the specified parameters.
    
    Args:
        dynamic_model_loader (callable): Function to load the dynamic model.
        static_model_loader (callable): Function to load the static model.
        ensemble_model_class (class): The class for the ensemble model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        experiment_name (str): Name of the experiment.
        mlp_hidden_dim (int): Hidden layer dimension for the ensemble MLP.
        output_dim (int): Output dimension of the model.
        l1_lambda (float): L1 regularization weight.
        l2_lambda (float): L2 regularization weight (weight decay).
        lr (float): Learning rate for the optimizer.
        pos_weight_value (float): Weight for the positive class in loss calculation.
        epochs (int): Number of training epochs.
        max_norm (float): Gradient clipping max norm value.
        patience (int): Early stopping patience.
        lr_patience (int): Learning rate scheduler patience.
        device (str or torch.device): Device to run the training (e.g., "cpu" or "cuda").
    
    Returns:
        model: Trained model.
    """
    # Load pretrained models
    dynamic_model = load_dynamic_model(dynamic_dim_in=dynamic_dim_in, hidden_dim = hidden_dim, max_seq_len=max_seq_len, batch_norm=False)
    
    static_model = load_static_model(scalar_input_dim=scalar_input_dim,
                                     scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
                                     embedding_hidden_dim=embedding_hidden_dim,
                                     batch_norm=batch_norm)



    # Define ensemble model
    model = EnsembleModel( dynamic_model, static_model, ensemble_mlp_hidden_dim, output_dim,batch_norm,scalar_mlp_hidden_dim,embedding_hidden_dim,hidden_dim)
    model = torch.compile(model)
    model.to(device)

    # Set a higher weight for the positive class
    pos_weight = torch.tensor(pos_weight_value, device=device)

    # Define optimizer with L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)

    # Fit the model (assuming a fit function is defined elsewhere)
    model = fit(
        model=model,
        experiment_name=experiment_name,
        num_epochs=epochs,
        optimizer=optimizer,
        pos_weight=pos_weight,
        l1_lambda=l1_lambda,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_norm=max_norm,
        lr_patience=lr_patience,
        patience=patience,
        target_outcome = target_outcome
    )
    
    return model




            
            
            
def load_model_for_eval():
    base_path = "/home/workspace/files/MilanK/Model1/final_models"
    
    subfolder = "final_cardiac_models/combined_model_without_transformer_simpler_demographics"

    # Extract filename from subfolder (last part) and add ".pth"
    filename = re.search(r'[^/]+$', subfolder).group() + ".pth"
    
    # Construct full model path
    model_path = os.path.join(base_path, subfolder, filename)
    
    print("Loading model from:", model_path)

    # Corrected function call
    model = load_model(model_path,
            dynamic_dim_in=18,
            hidden_dim=1024,
            batch_norm=False,
            scalar_input_dim=40,
            scalar_mlp_hidden_dim=256,
            embedding_hidden_dim=256,
            ensemble_mlp_hidden_dim=512,
            output_dim=1,
            max_seq_len=924,
            device="cpu")
    
    return model
