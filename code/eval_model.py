# Basic imports
import numpy as np  # For numerical computations and array manipulations
import pandas as pd  # For loading and handling time-series and static data
import math  # For positional encoding computations (optional)
import sys
import importlib
import os
import time
import shutil
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_auc_score
import gc

# PyTorch imports
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network layers and loss functions
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Datasets and DataLoaders for batching
from torch.nn import Transformer, TransformerEncoderLayer  # Transformer modules

#Tranformers import
from transformers import AutoTokenizer, AutoModel



def eval_model(model_folder,val,test,target_metric,target_value):
    
    
    path = "/home/workspace/files/MilanK/code"

    # Check if the path exists in sys.path and remove it
    if path in sys.path:
        sys.path.remove(path)
        print(f"Removed {path} from sys.path")
    else:
        print(f"{path} was not found in sys.path")


    

  ########################################################################
    ###load specific modules for a given model.
    if model_folder not in sys.path:
        sys.path.append(model_folder)
        
    #modules to be dynamically loaded
    model_module = importlib.import_module("model")     
    static_data_module = importlib.import_module("load_static_data")
    dataset_module = importlib.import_module("PatientDataset")
    validation_module = importlib.import_module("validate")

    #########################################################################

    

    model = model_module.load_model_for_eval()

    val_dataset = dataset_module.PatientDataset(patient_list = val, min_window_min=15, step_min=15,max_window_min=15,
                            prediction_window_length=15)


    test_dataset = dataset_module.PatientDataset(patient_list = test, min_window_min=15, step_min=15,max_window_min=15,
                             prediction_window_length=15)

    batch_size=128
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


    device = 'cpu'
    exp_name = model_folder.rsplit("/", 1)[-1]
    pos_weight = torch.tensor(30, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (_, _, _, _, _, best_threshold,
     _,_, _, _, _, _,_,_) = validation_module.validate(model,
                                                      val_loader,
                                                      criterion=criterion,
                                                      device=device,
                                                      target_metric=target_metric,
                                                      target_value=target_value)

    
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, _,
     fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validation_module.validate(model,
                                                                                                            test_loader,
                                                                                                            criterion=criterion,
                                                                                                            device=device,
                                                                                                            threshold=best_threshold
                                                                                                            )


    ##########################################################################
    #clear up env for fresh modules

    del sys.modules["model"]  # Remove from loaded modules
    del sys.modules["load_static_data"]
    del sys.modules['PatientDataset']
    del sys.modules['validate']
    sys.path.remove(model_folder)  # Remove path from sys.path
            # Force garbage collection to free memory

    gc.collect()
    
    
    return exp_name, avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1
