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
    
    

class CombinedModel(nn.Module):
    def __init__(self, scalar_input_dim, embedding_input_dim, scalar_mlp_hidden_dim, embedding_hidden_dim, combined_mlp_hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
    
        # MLP for static data
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_input_dim, scalar_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(scalar_mlp_hidden_dim, scalar_mlp_hidden_dim),
            nn.ReLU()
        )
        
        self.embedding_mlp = nn.Sequential(
            nn.Linear(embedding_input_dim, embedding_hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_hidden_dim, embedding_hidden_dim),
            nn.ReLU()
        )
        

        # MLP for combined features
        self.combined_mlp = nn.Sequential(
            nn.Linear( scalar_mlp_hidden_dim + embedding_hidden_dim, combined_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_mlp_hidden_dim, combined_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_mlp_hidden_dim, output_dim),
            #raw values get output and converted to zero/one later on
        )

    def forward(self,  scalar_features, embedding_features):
        # Process dynamic features through transformer encoder

        # Process static features through MLP
        scalar_output = self.scalar_mlp(scalar_features)  # Shape: (batch_size, static_mlp_hidden_dim)
        embedding_output = self.embedding_mlp(embedding_features)  # Shape: (batch_size, static_mlp_hidden_dim)
        combined_features = torch.cat([scalar_output, embedding_output], dim=-1) 

        # Pass through combined MLP for final prediction
        output = self.combined_mlp(combined_features)  # Shape: (batch_size, output_dim)
        return output
    




def run_exp(train_loader,val_loader):
    
    
    batch_size = 64
    
    
    scalar_input_dim = 140  # Example static feature dimension
    embedding_input_dim = 768 # Example static feature dimension
    scalar_mlp_hidden_dim = 256
    embedding_hidden_dim = 256
    combined_mlp_hidden_dim = 512
    output_dim = 1  # For two outputs: respiratory and cardiac deterioration probabilities
    
    # Instantiate the combined model
    model = CombinedModel(
        scalar_input_dim=scalar_input_dim,
        embedding_input_dim=embedding_input_dim,
        scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
        embedding_hidden_dim=embedding_hidden_dim,
        combined_mlp_hidden_dim=combined_mlp_hidden_dim,
        output_dim=output_dim
    )
    model = torch.compile(model)
    
    
    device = torch.device("cpu")
    
    
    pos_weight = torch.tensor(20.0, device=device)  # Adjust this value based on imbalance ratio
    
    # Regularization parameters
    l1_lambda = 0  # Adjust as necessary for L1 regularization
    l2_lambda = 0.003  # For L2 regularization (weight decay)
    
    
    # Initialize optimizer with only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=1e-5,
        weight_decay=l2_lambda  # Set your L2 regularization coefficient here
    )
    
    
    trained_model = fit(
        model=model,
        experiment_name='demographics_only_full',
        num_epochs=25,
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
        training_mode = 'demographic'
    )
    return trained_model




def load_model_for_eval():

    scalar_input_dim = 140  # Example static feature dimension
    embedding_input_dim = 768 # Example static feature dimension
    scalar_mlp_hidden_dim = 256
    embedding_hidden_dim = 256
    combined_mlp_hidden_dim = 512
    output_dim = 1  # For two outputs: respiratory and cardiac deterioration probabilities
    
    # Instantiate the combined model
    model = CombinedModel(
        scalar_input_dim=scalar_input_dim,
        embedding_input_dim=embedding_input_dim,
        scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
        embedding_hidden_dim=embedding_hidden_dim,
        combined_mlp_hidden_dim=combined_mlp_hidden_dim,
        output_dim=output_dim
    )
    model = torch.compile(model)
    
    
        
        


    
    base_path = "/home/workspace/files/MilanK/Model1/final_models"
    
    subfolder = "final_cardiac_models/demographics_only_full"

    # Extract filename from subfolder (last part) and add ".pth"
    filename = re.search(r'[^/]+$', subfolder).group() + ".pth"
    
    # Construct full model path
    model_path = os.path.join(base_path, subfolder, filename)
    
    print("Loading model from:", model_path)

    
    
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only = True))
        print("Loaded saved model weights.")
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()  # Ensure model is in evaluation mode
        
    return model
