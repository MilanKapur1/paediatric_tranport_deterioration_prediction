import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, auc,
    confusion_matrix, f1_score, brier_score_loss
)
import torch.nn as nn
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression  # For isotonic calibration
from sklearn.linear_model import LogisticRegression  # For Platt scaling
import joblib
import inspect
            

def validate(model, val_loader, criterion, device, target_metric = 'fpr', target_value = 0.2, threshold=None):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
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

            feature_tensor = feature_tensor.to(device)
            missingness_mask_tensor = missingness_mask_tensor.to(device)
            static_tensor = static_tensor.to(device)
            
            label_tensor = label_tensor[:, 1].unsqueeze(1).to(device)

            # Call the model with all possible inputs
            output = model(
                dynamic_features=feature_tensor,
                missingness_mask=missingness_mask_tensor,
                static_features = static_tensor
            )

            loss = criterion(output, label_tensor)
            total_loss += loss.item()

            probs = torch.sigmoid(output).cpu().squeeze()
            all_probs.append(probs)
            all_labels.append(label_tensor.cpu().squeeze())

    avg_loss = total_loss / num_batches
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute AUC metrics
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall_vals, precision_vals)
    auc_roc = roc_auc_score(all_labels, all_probs)

    # Compute ROC curve
    fpr_vals, tpr_vals, thresholds = roc_curve(all_labels, all_probs)
    

    # If no threshold is provided, select the best threshold based on the target metric
    if threshold is None:
        if target_metric.lower() == "fpr":
            # Choose thresholds where FPR is no greater than target_value
            valid_indices = np.where(fpr_vals <= target_value)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(tpr_vals[valid_indices])]
            else:
                best_idx = np.argmin(np.abs(fpr_vals - target_value))
        elif target_metric.lower() == "tpr":
            # Choose thresholds where TPR is at least target_value
            valid_indices = np.where(tpr_vals >= target_value)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmin(fpr_vals[valid_indices])]
            else:
                best_idx = np.argmin(np.abs(tpr_vals - target_value))
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = threshold

    # Additional metrics computation...
    pred_labels = (all_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    f1 = f1_score(all_labels, pred_labels)

    # Return the arrays for the ROC curve along with metrics
    return (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1)
            
            

def plot_curves(device, model, data_loader, pos_weight, exp_name, exp_dataset):
    """
    Plots AUROC and AUPRC curves and prints evaluation metrics.
    Uses the best threshold where FPR ≈ 0.2, determined by `validate`.
    """
    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Get validation metrics
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)

    base_rate = np.mean(all_labels)

    # Plot AUROC
    fpr_vals, tpr_vals, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_vals, tpr_vals, label=f"AUROC = {auc_roc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{exp_name} AUROC {exp_dataset}")
    plt.legend(loc="lower right")

    # Plot AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, label=f"AUPRC = {auc_pr:.4f}")
    plt.axhline(y=base_rate, color='darkred', linestyle='--', label=f"Base Rate = {base_rate:.4f}")
    plt.text(0.99, base_rate + 0.02, f"{base_rate:.4f}", color='darkred', ha='right', va='bottom')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{exp_name} AUPRC {exp_dataset}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    print(f"Best Threshold (FPR ≈ 0.2): {best_threshold:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"TPR: {tpr:.4f}")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1-score: {f1:.4f}")

def plot_calibration_curve(device, model, data_loader, pos_weight, exp_name, exp_dataset, n_bins=10):
    """
    Plots a calibration curve to assess how well-calibrated the base model is.
    """
    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=n_bins)
    brier_score = brier_score_loss(all_labels, all_probs)
    print(f"Brier Score: {brier_score:.4f} (Lower is better)")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Base Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{exp_name} Calibration Curve - {exp_dataset}")
    plt.legend()
    plt.grid()
    plt.show()

def train_and_save_isotonic(device, model, val_loader, pos_weight, exp_name, exp_dataset, n_bins=10, save_path="isotonic_reg.pkl"):
    """
    Trains Isotonic Regression on the validation set, saves the model, and plots calibration.
    """
    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)    
    iso_scaler = IsotonicRegression(out_of_bounds="clip")
    iso_scaler.fit(all_probs, all_labels)
    joblib.dump(iso_scaler, save_path)
    print(f"Isotonic Regression model saved to {save_path}")

    # Apply isotonic regression on validation set
    all_probs_iso = iso_scaler.transform(all_probs)
    prob_true_before, prob_pred_before = calibration_curve(all_labels, all_probs, n_bins=n_bins)
    prob_true_after, prob_pred_after = calibration_curve(all_labels, all_probs_iso, n_bins=n_bins)
    brier_score_before = brier_score_loss(all_labels, all_probs)
    brier_score_after = brier_score_loss(all_labels, all_probs_iso)
    print(f"Brier Score Before Isotonic Regression: {brier_score_before:.4f}")
    print(f"Brier Score After Isotonic Regression: {brier_score_after:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, marker='o', label="Before Isotonic Regression")
    plt.plot(prob_pred_after, prob_true_after, marker='s', label="After Isotonic Regression")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{exp_name} Calibration Curve - {exp_dataset}")
    plt.legend()
    plt.grid()
    plt.show()

    return iso_scaler

def apply_isotonic_to_test(device, model, test_loader, pos_weight, save_path="isotonic_reg.pkl"):
    """
    Loads the Isotonic Regression model and applies it to the test set.
    """
    iso_scaler = joblib.load(save_path)
    print(f"Loaded Isotonic Regression model from {save_path}")

    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)
     
    # Apply isotonic regression on test set
    all_probs_iso = iso_scaler.transform(all_probs)
    prob_true_before, prob_pred_before = calibration_curve(all_labels, all_probs, n_bins=10)
    prob_true_after, prob_pred_after = calibration_curve(all_labels, all_probs_iso, n_bins=10)
    brier_score_before = brier_score_loss(all_labels, all_probs)
    brier_score_after = brier_score_loss(all_labels, all_probs_iso)
    print(f"Brier Score Before Isotonic Regression (Test Set): {brier_score_before:.4f}")
    print(f"Brier Score After Isotonic Regression (Test Set): {brier_score_after:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, marker='o', label="Before Isotonic Regression (Test)")
    plt.plot(prob_pred_after, prob_true_after, marker='s', label="After Isotonic Regression (Test)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Test Set")
    plt.legend()
    plt.grid()
    plt.show()

def train_and_save_platt(device, model, val_loader, pos_weight, exp_name, exp_dataset, save_path="platt_scaler.joblib"):
    """
    Trains a logistic regression model (Platt scaling) on the validation set,
    saves the model, and plots calibration curves.
    """
    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)
     
    # Reshape probabilities to 2D array for logistic regression
    all_probs_reshaped = all_probs.reshape(-1, 1)
    platt_scaler = LogisticRegression(solver='lbfgs')
    platt_scaler.fit(all_probs_reshaped, all_labels)
    joblib.dump(platt_scaler, save_path)
    print(f"Platt scaling model saved to {save_path}")

    # Apply Platt scaling on validation set
    all_probs_platt = platt_scaler.predict_proba(all_probs_reshaped)[:, 1]
    prob_true_before, prob_pred_before = calibration_curve(all_labels, all_probs, n_bins=10)
    prob_true_after, prob_pred_after = calibration_curve(all_labels, all_probs_platt, n_bins=10)
    brier_score_before = brier_score_loss(all_labels, all_probs)
    brier_score_after = brier_score_loss(all_labels, all_probs_platt)
    print(f"Brier Score Before Platt Scaling: {brier_score_before:.4f}")
    print(f"Brier Score After Platt Scaling: {brier_score_after:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, marker='o', label="Before Platt Scaling")
    plt.plot(prob_pred_after, prob_true_after, marker='s', label="After Platt Scaling")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{exp_name} Calibration Curve - {exp_dataset}")
    plt.legend()
    plt.grid()
    plt.show()

    return platt_scaler

def apply_platt_to_test(device, model, test_loader, pos_weight, save_path="platt_scaler.joblib"):
    """
    Loads the Platt scaling model and applies it to calibrate test set predictions.
    """
    platt_scaler = joblib.load(save_path)
    print(f"Loaded Platt scaling model from {save_path}")

    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
            fpr_vals, tpr_vals, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)
     
    # Reshape and apply Platt scaling on test set
    all_probs_reshaped = all_probs.reshape(-1, 1)
    all_probs_platt = platt_scaler.predict_proba(all_probs_reshaped)[:, 1]
    prob_true_before, prob_pred_before = calibration_curve(all_labels, all_probs, n_bins=10)
    prob_true_after, prob_pred_after = calibration_curve(all_labels, all_probs_platt, n_bins=10)
    brier_score_before = brier_score_loss(all_labels, all_probs)
    brier_score_after = brier_score_loss(all_labels, all_probs_platt)
    print(f"Brier Score Before Platt Scaling (Test Set): {brier_score_before:.4f}")
    print(f"Brier Score After Platt Scaling (Test Set): {brier_score_after:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, marker='o', label="Before Platt Scaling (Test)")
    plt.plot(prob_pred_after, prob_true_after, marker='s', label="After Platt Scaling (Test)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Test Set")
    plt.legend()
    plt.grid()
    plt.show()




from betacal import BetaCalibration
import joblib

def train_and_save_beta(device, model, val_loader, pos_weight, exp_name, exp_dataset, save_path="beta_scaler.joblib"):
    """
    Trains a Beta Calibration model on the validation set,
    saves the model, and plots calibration curves.
    """
    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
     fpr, tpr, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)
     
    # Fit Beta Calibration model
    beta_scaler = BetaCalibration()
    beta_scaler.fit(all_probs, all_labels)
    joblib.dump(beta_scaler, save_path)
    print(f"Beta Calibration model saved to {save_path}")

    # Apply Beta Calibration on validation set
    all_probs_beta = beta_scaler.predict(all_probs)

    # Compute calibration curves before and after
    prob_true_before, prob_pred_before = calibration_curve(all_labels, all_probs, n_bins=10)
    prob_true_after, prob_pred_after = calibration_curve(all_labels, all_probs_beta, n_bins=10)

    # Compute Brier scores
    brier_score_before = brier_score_loss(all_labels, all_probs)
    brier_score_after = brier_score_loss(all_labels, all_probs_beta)
    
    print(f"Brier Score Before Beta Calibration: {brier_score_before:.4f}")
    print(f"Brier Score After Beta Calibration: {brier_score_after:.4f}")

    # Plot calibration curves
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, marker='o', label="Before Beta Calibration")
    plt.plot(prob_pred_after, prob_true_after, marker='s', label="After Beta Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{exp_name} Calibration Curve - {exp_dataset}")
    plt.legend()
    plt.grid()
    plt.show()

    return beta_scaler

def apply_beta_to_test(device, model, test_loader, pos_weight, save_path="beta_scaler.joblib"):
    """
    Loads the Beta Calibration model and applies it to the test set.
    """
    beta_scaler = joblib.load(save_path)
    print(f"Loaded Beta Calibration model from {save_path}")

    pos_weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    (avg_loss, auc_pr, auc_roc, all_probs, all_labels, best_threshold,
     fpr, tpr, sensitivity, specificity, ppv, npv, balanced_acc, f1) = validate(model, data_loader, criterion, device)
     
    # Apply Beta Calibration on test set
    all_probs_beta = beta_scaler.predict(all_probs)

    # Compute calibration curves before and after
    prob_true_before, prob_pred_before = calibration_curve(all_labels, all_probs, n_bins=10)
    prob_true_after, prob_pred_after = calibration_curve(all_labels, all_probs_beta, n_bins=10)

    # Compute Brier scores
    brier_score_before = brier_score_loss(all_labels, all_probs)
    brier_score_after = brier_score_loss(all_labels, all_probs_beta)
    
    print(f"Brier Score Before Beta Calibration (Test Set): {brier_score_before:.4f}")
    print(f"Brier Score After Beta Calibration (Test Set): {brier_score_after:.4f}")

    # Plot calibration curves
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, marker='o', label="Before Beta Calibration (Test)")
    plt.plot(prob_pred_after, prob_true_after, marker='s', label="After Beta Calibration (Test)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y=x)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Test Set")
    plt.legend()
    plt.grid()
    plt.show()







# ============================================================
# Example Usage Instructions:
# ============================================================
#
# 1. Calibration on the Validation Set:
#    Choose either Isotonic or Platt scaling.
#
#    For Isotonic Scaling:
#       iso_model = train_and_save_isotonic(device, model, val_loader, pos_weight, exp_name, exp_dataset)
#
#    For Platt Scaling:
#       platt_model = train_and_save_platt(device, model, val_loader, pos_weight, exp_name, exp_dataset)
#
# 2. Obtain validation set predictions:
#       _, _, _, val_probs, val_labels, *_ = validate(model, val_loader,
#           nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device)), device)
#
#    If using Platt scaling:
#       val_probs_calibrated = platt_model.predict_proba(val_probs.reshape(-1, 1))[:, 1]
#    Or, if using Isotonic scaling:
#       val_probs_calibrated = iso_model.transform(val_probs)
#
# 3. Find the best thresholds on the validation set:
#       best_thresh_f1, best_f1 = find_best_threshold(val_labels, val_probs_calibrated, metric='f1')
#       best_thresh_bacc, best_bacc = find_best_threshold(val_labels, val_probs_calibrated, metric='balanced_accuracy')
#
#       print(f"Best Threshold for F1 Score: {best_thresh_f1:.2f} (F1: {best_f1:.4f})")
#       print(f"Best Threshold for Balanced Accuracy: {best_thresh_bacc:.2f} (Balanced Acc: {best_bacc:.4f})")
#
# 4. Apply Calibration to the Test Set:
#
#    For Isotonic Scaling:
#       apply_isotonic_to_test(device, model, test_loader, pos_weight)
#
#    For Platt Scaling:
#       apply_platt_to_test(device, model, test_loader, pos_weight)
#
#    These functions will display the calibration curves and metrics.
#
# 5. Recalculate Test Metrics Using the New Threshold:
#
#    After obtaining test set predictions (e.g., via validate()),
#    if using the calibrated probabilities (let's call them test_probs_calibrated) and true labels test_labels, use:
#
#       f1_test, balanced_acc_test, sens_test, spec_test = calculate_test_metrics(test_labels, test_probs_calibrated, best_thresh_f1)
#
#       print(f"Test Metrics with Threshold {best_thresh_f1:.2f}:")
#       print(f"F1 Score: {f1_test:.4f}, Balanced Accuracy: {balanced_acc_test:.4f}")
#
# Ensure you have defined your device, model, loaders (val_loader, test_loader), pos_weight, exp_name, and exp_dataset before using these functions.
