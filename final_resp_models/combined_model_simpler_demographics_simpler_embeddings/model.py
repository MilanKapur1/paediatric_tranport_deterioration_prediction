import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Decoder
from torch.utils.data import DataLoader
import os
import torch.optim as optim  # Optimization algorithms
from fit import fit  # Custom function for model training
import re

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dynamic_dim_in, dynamic_dim_out, max_seq_len, depth, heads,attn_dropout,ff_dropout):
        super().__init__()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=dynamic_dim_in * 2,  # Account for concatenated features and missingness mask
            dim_out=dynamic_dim_out,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dynamic_dim_in * 2,  # Match the input dimension
                depth=depth,
                heads=heads,
                rotary_pos_emb=True,        
                attn_dropout = attn_dropout,    # dropout post-attention
                ff_dropout = ff_dropout       # feedforward dropout
            )
        )

    def forward(self, features, missingness_mask=None):
        # Concatenate the missingness mask to the feature tensor along the last dimension
        features_with_missing_mask = torch.cat([features, missingness_mask], dim=-1)  # Shape: (batch_size, seq_len, 32)        

        # Pass the combined features and padding mask through the encoder model
        output = self.encoder(features_with_missing_mask)
        return output
    
    
    
    

class DynamicModel(nn.Module):
    def __init__(self, transformer_encoder, batch_norm,dynamic_dim_out):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(dynamic_dim_out)

    def forward(self, dynamic_features, missingness_mask=None):
        # Process dynamic features through transformer encoder
        dynamic_output = self.transformer_encoder(dynamic_features, missingness_mask)
        
        # Pooling: Take the mean across the sequence length dimension
        dynamic_output_pooled = dynamic_output[:, -1, :]
        
        if self.batch_norm:
            dynamic_output_pooled = self.bn(dynamic_output_pooled)
            
                # Capture the intermediate output here
        return  dynamic_output_pooled


def load_dynamic_model(depth, heads, attn_dropout, ff_dropout, batch_norm,dynamic_dim_in,dynamic_dim_out,max_seq_len):
    # Instantiate the TransformerEncoderLayer
    transformer_encoder = TransformerEncoderLayer(
        dynamic_dim_in=dynamic_dim_in,      # Input feature dimension before concatenation
        dynamic_dim_out=dynamic_dim_out,    # Output dimension
        max_seq_len=max_seq_len,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout
    )

    # Instantiate the combined model
    model = DynamicModel(
        transformer_encoder=transformer_encoder,
        batch_norm=batch_norm,
        dynamic_dim_out=dynamic_dim_out
        )            
    return model



    

class StaticModel(nn.Module):
    def __init__(self, static_input_dim, scalar_mlp_hidden_dim, batch_norm):
        super().__init__()
        
        self.use_bn = batch_norm        
        
    
        # MLP for scalar data
        if self.use_bn:
            self.mlp = nn.Sequential(
                nn.Linear(static_input_dim, scalar_mlp_hidden_dim),
                nn.BatchNorm1d(scalar_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(scalar_mlp_hidden_dim, scalar_mlp_hidden_dim),
                nn.BatchNorm1d(scalar_mlp_hidden_dim),
                nn.ReLU()
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(static_input_dim, scalar_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(scalar_mlp_hidden_dim, scalar_mlp_hidden_dim),
                nn.ReLU()
            )
        

            
            
    def forward(self, static_features):

        # Combine the outputs
 
        output = self.mlp(static_features)
        
        

        return output





def load_static_model(static_input_dim, scalar_mlp_hidden_dim, batch_norm):
    


    # Instantiate the combined model
    model = StaticModel(
        static_input_dim=static_input_dim,    
        scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
        batch_norm=batch_norm
    )        
    return model




class EnsembleModel(nn.Module):
    def __init__(self, dynamic_model, static_model, ensemble_mlp_hidden_dim, output_dim,batch_norm,scalar_mlp_hidden_dim, dynamic_dim_out):
        super().__init__()  # Corrected super() call
        self.dynamic_model = dynamic_model
        self.static_model = static_model
        self.use_bn = batch_norm
        self.mlp_dim = dynamic_dim_out + scalar_mlp_hidden_dim 
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
    
    def forward(self, dynamic_features, missingness_mask, static_features):
        # Get logits from pretrained models
        dynamic_output = self.dynamic_model(
            dynamic_features=dynamic_features,
            missingness_mask=missingness_mask
        )  # Shape: (batch_size, 1)
        
        static_output = self.static_model(
            static_features=static_features
        )  # Shape: (batch_size, 1)
        
        # Concatenate logits
        combined_penultimate = torch.cat([static_output, dynamic_output], dim=1)  # Shape: (batch_size, 2)
        
        # Pass through the MLP
        output = self.mlp(combined_penultimate )  # Shape: (batch_size, output_dim)
        return output

###############################################################
####usage

def load_model(model_path,
            depth=2,
            heads=2,
            dynamic_dim_in=18,
            dynamic_dim_out=64,
            max_seq_len=924,
            attn_dropout=0.1,
            ff_dropout=0.1,
            batch_norm=False,
            static_input_dim=140,
            scalar_mlp_hidden_dim=256,
            ensemble_mlp_hidden_dim=512,
            output_dim=1,
            device="cpu",
            freeze=True):
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
    dynamic_model = load_dynamic_model(depth=depth,
                                       heads=heads,
                                       attn_dropout=attn_dropout,
                                       ff_dropout=ff_dropout,
                                       batch_norm=batch_norm,
                                       dynamic_dim_in=dynamic_dim_in,
                                        dynamic_dim_out=dynamic_dim_out,
                                        max_seq_len=max_seq_len
                                       )
    
    static_model = load_static_model(static_input_dim=static_input_dim,
                                     scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
                                     batch_norm=batch_norm)



    # Define ensemble model
    model = EnsembleModel( dynamic_model, static_model, ensemble_mlp_hidden_dim, output_dim,batch_norm,scalar_mlp_hidden_dim, dynamic_dim_out)
    model = torch.compile(model)
    model.to(device)
    
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only = True))
        print("Loaded saved model weights.")
    
        # Freeze the model if `freeze=True`
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()  # Ensure model is in evaluation mode
        print("Model has been frozen.")

    return model
        



def run_exp(experiment_name,
            train_loader,
            val_loader,
            depth=2,
            heads=2,
            dynamic_dim_in=18,
            dynamic_dim_out=64,
            max_seq_len=924,
            attn_dropout=0.1,
            ff_dropout=0.1,
            batch_norm=False,
            static_input_dim=140,
            scalar_mlp_hidden_dim=256,
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
            target_outcome = 'resp',
            training_mode = 'combined'):
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
    dynamic_model = load_dynamic_model(depth=depth,
                                       heads=heads,
                                       attn_dropout=attn_dropout,
                                       ff_dropout=ff_dropout,
                                       batch_norm=batch_norm,
                                       dynamic_dim_in=dynamic_dim_in,
                                        dynamic_dim_out=dynamic_dim_out,
                                        max_seq_len=max_seq_len
                                       )
    
    static_model = load_static_model(static_input_dim=static_input_dim,
                                     scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
                                     batch_norm=batch_norm)



    # Define ensemble model
    model = EnsembleModel( dynamic_model, static_model, ensemble_mlp_hidden_dim, output_dim,batch_norm,scalar_mlp_hidden_dim,dynamic_dim_out)
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
        target_outcome = target_outcome,
        training_mode=training_mode
    )
    
    return model




def load_model_for_eval():
    base_path = "/home/workspace/files/MilanK/Model1/final_models"
    
    subfolder = "final_resp_models/combined_model_simpler_demographics_simpler_embeddings"

    # Extract filename from subfolder (last part) and add ".pth"
    filename = re.search(r'[^/]+$', subfolder).group() + ".pth"
    
    # Construct full model path
    model_path = os.path.join(base_path, subfolder, filename)
    
    print("Loading model from:", model_path)

    # Corrected function call
    model = load_model(
        model_path=model_path,
        depth=2,
        heads=2,
        dynamic_dim_in=18,
        dynamic_dim_out=64,
        max_seq_len=924,
        attn_dropout=0,
        ff_dropout=0,
        batch_norm=False,
        static_input_dim=47,
        scalar_mlp_hidden_dim=256,
        ensemble_mlp_hidden_dim=512,
        output_dim=1,
        device="cpu",
        freeze=True
    )
        
    return model

