import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score

def stratified_bootstrap_metrics(y_true, y_prob, threshold, n_bootstraps=1000, alpha=0.05, random_state=None):
    """
    Performs stratified bootstrapping to compute confidence intervals for:
        AUROC, AUCPR, Sensitivity, Specificity, PPV, NPV,
        Balanced Accuracy, and F1-Score.
    
    Args:
        y_true (np.array): True binary labels.
        y_prob (np.array): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to binary predictions.
        n_bootstraps (int): Number of bootstrap iterations.
        alpha (float): Significance level (e.g., 0.05 for 95% CI).
        random_state (int, optional): Seed for reproducibility.
    
    Returns:
        dict: Dictionary with keys for each metric mapping to a tuple 
              (mean, lower CI, upper CI).
    """
    rng = np.random.default_rng(random_state)
    # Create a DataFrame from y_true and y_prob
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    
    # Split into positive and negative subsets
    df_pos = df[df['y_true'] == 1]
    df_neg = df[df['y_true'] == 0]
    
    n_pos = len(df_pos)
    n_neg = len(df_neg)
    
    # Lists to collect metric values for each bootstrap replicate
    auroc_list = []
    aucpr_list = []
    sensitivity_list = []
    specificity_list = []
    ppv_list = []
    npv_list = []
    balanced_acc_list = []
    f1_list = []
    
    for _ in range(n_bootstraps):
        # Stratified resampling: sample positives and negatives separately
        resampled_pos = df_pos.sample(n=n_pos, replace=True, random_state=rng.integers(1e9))
        resampled_neg = df_neg.sample(n=n_neg, replace=True, random_state=rng.integers(1e9))
        resampled_df = pd.concat([resampled_pos, resampled_neg], ignore_index=True)
        
        # Only proceed if both classes are present
        if 0 in resampled_df['y_true'].values and 1 in resampled_df['y_true'].values:
            yt = resampled_df['y_true'].values
            yp = resampled_df['y_prob'].values
            
            # Compute AUROC and AUCPR
            auroc_list.append(roc_auc_score(yt, yp))
            aucpr_list.append(average_precision_score(yt, yp))
            
            # Compute binary predictions using the chosen threshold
            pred = (yp >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
            
            # Sensitivity (Recall)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            # Specificity
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            # PPV (Precision)
            ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            # NPV
            npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0
            # Balanced Accuracy
            balanced_acc = (sens + spec) / 2
            # F1-Score
            f1_val = f1_score(yt, pred)
            
            sensitivity_list.append(sens)
            specificity_list.append(spec)
            ppv_list.append(ppv_val)
            npv_list.append(npv_val)
            balanced_acc_list.append(balanced_acc)
            f1_list.append(f1_val)
    
    # Helper function to compute mean and CI from an array of values
    def get_ci(values):
        arr = np.array(values)
        mean_val = np.median(arr)
        lower = np.percentile(arr, 100 * (alpha / 2))
        upper = np.percentile(arr, 100 * (1 - alpha / 2))
        return mean_val, lower, upper

    # Build the results dictionary for the desired metrics
    results = {
        'auroc': get_ci(auroc_list),
        'aucpr': get_ci(aucpr_list),
        'sensitivity': get_ci(sensitivity_list),
        'specificity': get_ci(specificity_list),
        'ppv': get_ci(ppv_list),
        'npv': get_ci(npv_list),
        'balanced_accuracy': get_ci(balanced_acc_list),
        'f1': get_ci(f1_list)
    }
    
    return results


