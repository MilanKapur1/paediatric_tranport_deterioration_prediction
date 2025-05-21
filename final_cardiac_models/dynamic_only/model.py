# # Basic imports
import numpy as np  # For numerical computations and array manipulations
import pandas as pd  # For loading and handling time-series and static data
import sys
import importlib
import os
import time
from tqdm import tqdm
import re

# PyTorch imports
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network layers and loss functions
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Datasets and DataLoaders for batching
from torch.nn import Transformer, TransformerEncoderLayer  # Transformer modules

# #Tranformers import
from transformers import AutoTokenizer, AutoModel
from x_transformers import ContinuousTransformerWrapper, Decoder

module_path = '/home/workspace/files/MilanK/Model1/final_models/code'
# Add the module's directory to the system path if it's not already present
if module_path not in sys.path:
    sys.path.append(module_path)
 

from generating_datasets_for_torch import *
from load_static_data import *
from PatientDataset import *
from generate_labels_dataframe_with_dataloader import *
from load_train_test_split import *
from model import *
from load_patient_list import *
from forward_loop import *
from fit import *
from validate import *


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_in, dim_out, max_seq_len, depth, heads):
        super(TransformerEncoderLayer, self).__init__()
        self.dim_out = dim_out
        self.encoder = ContinuousTransformerWrapper(
            dim_in=dim_in * 2,  # Account for concatenated features and missingness mask
            dim_out=dim_out,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dim_in * 2,  # Match the input dimension
                depth=depth,
                heads=heads,
                rotary_pos_emb=True,        
                attn_dropout = 0.1,    # dropout post-attention
                ff_dropout = 0.1   # feedforward dropout
            )
        )

    def forward(self, features, missingness_mask=None):
        # Concatenate the missingness mask to the feature tensor along the last dimension
        features_with_missing_mask = torch.cat([features, missingness_mask], dim=-1)  # Shape: (batch_size, seq_len, 32)        

        # Pass the combined features and padding mask through the encoder model
        output = self.encoder(features_with_missing_mask)
        return output
    
    
    
    

class CombinedModel(nn.Module):
    def __init__(self, transformer_encoder, combined_mlp_hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.transformer_encoder = transformer_encoder
        


        # MLP for combined features
        self.combined_mlp = nn.Sequential(
            nn.Linear(transformer_encoder.dim_out, combined_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_mlp_hidden_dim, combined_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_mlp_hidden_dim, output_dim),
            #raw values get output and converted to zero/one later on
        )

    def forward(self, dynamic_features, missingness_mask=None):
        # Process dynamic features through transformer encoder
        dynamic_output = self.transformer_encoder(dynamic_features, missingness_mask)
        
        # Pooling: Take the mean across the sequence length dimension
        dynamic_output_pooled = dynamic_output[:, -1, :]
        

        # Pass through combined MLP for final prediction
        output = self.combined_mlp(dynamic_output_pooled)  # Shape: (batch_size, output_dim)
        return output





def run_exp(train_loader, val_loader):
    
    

    transformer_encoder = TransformerEncoderLayer(
        dim_in=18,      # Input feature dimension before concatenation
        dim_out=128,    # Output dimension
        max_seq_len=924,
        depth=4,
        heads=2
    )
    
    # Define model parameters
    combined_mlp_hidden_dim = 512
    output_dim = 1  # For two outputs: respiratory and cardiac deterioration probabilities
    
    # Instantiate the combined model
    model = CombinedModel(
        transformer_encoder=transformer_encoder,
        combined_mlp_hidden_dim=combined_mlp_hidden_dim,
        output_dim=output_dim
    )
    model = torch.compile(model)
    
    device = torch.device("cpu")
    
    
    
    pos_weight = torch.tensor(30.0, device=device)  # Adjust this value based on imbalance ratio
    
    # Regularization parameters
    l1_lambda = 0  # Adjust as necessary for L1 regularization
    l2_lambda = 0  # For L2 regularization (weight decay)
    

    
    # Initialize optimizer with only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=1e-5,
        weight_decay=l2_lambda  # Set your L2 regularization coefficient here
    )
    
    
    trained_model = fit(
        model=model,
        experiment_name='dynamic_only',
        num_epochs=10,
        optimizer=optimizer,
        pos_weight=pos_weight,
        l1_lambda=l1_lambda,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_norm=10,
        lr_patience=5,
        patience=20,
        target_outcome = 'cardiac',
        training_mode = 'dynamic'
    )
    
    return trained_model
    



def load_model_for_eval():

    
    transformer_encoder = TransformerEncoderLayer(
        dim_in=18,      # Input feature dimension before concatenation
        dim_out=128,    # Output dimension
        max_seq_len=924,
        depth=4,
        heads=2
    )
    
    # Define model parameters
    combined_mlp_hidden_dim = 512
    output_dim = 1  # For two outputs: respiratory and cardiac deterioration probabilities
    
    # Instantiate the combined model
    model = CombinedModel(
        transformer_encoder=transformer_encoder,
        combined_mlp_hidden_dim=combined_mlp_hidden_dim,
        output_dim=output_dim
    )
    model = torch.compile(model)

    
    model_path = '/home/workspace/files/MilanK/Model1/final_models/final_cardiac_models/dynamic_only/dynamic_only.pth'          

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only = True))
        print("Loaded saved model weights.")
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # Ensure model is in evaluation mode

    print("Model has been frozen.")
    
    return model
    
    