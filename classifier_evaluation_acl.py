
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import importlib

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from reconstruction_model import UNet3D
from diffusion_model import DDPM3D
from classifier import InteractionLogisticRegression
from feature_selector import GlobalMaskedFeatureSelector, CNNWithGlobalMasking



from dataset import MRIDataset, CombinedMRIFeatureDataset
from feature_extractor import first_order_and_shape_features
from sklearn.metrics import accuracy_score, recall_score

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import VarianceThreshold
import torch.optim as optim
import json
from sklearn.model_selection import StratifiedKFold


# ------------------------------------------
# Radiomics-like Feature Extraction (1st order)
# ------------------------------------------

# -------------------------------------------------
# Feature Extraction Pipeline Using Pretrained DDPM
# -------------------------------------------------

import torch

def extract_features_with_pretrained_ddpm(
    model,
    loader,
    device,
    voxelArrayShift=0,
    pixelSpacing=[1.0, 1.0, 1.0]
):
    """
    Merged approach:
    
    1) For each batch of 3D volumes (potentially multi-channel, e.g. 3 views),
       corrupt the central region (e.g. 60% in each dimension).
    2) Add noise at a random diffusion timestep.
    3) Reconstruct using the pretrained diffusion model.
    4) Subdivide that same center region into 2x2x2=8 mini-patches.
    5) Extract first-order stats from each sub-patch (original & reconstructed),
       adding (loc_i, loc_j, loc_k) to each set of features.
    6) Return (all_features, all_labels).
    """

    model.eval()

    all_features = []
    all_labels = []
    valid_indices = []

    with torch.no_grad():
        num_batches = len(loader)
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"[DEBUG] Processing batch {batch_idx+1} / {num_batches}")
            images = images.to(device)  # shape: [B, C, D, H, W]
            labels = labels.to(device)  # shape: [B, ...] (depends on your dataset)

            B, C, D, H, W = images.shape
            print(f"    [DEBUG] Batch {batch_idx+1} shape: B={B}, C={C}, D={D}, H={H}, W={W}")

            # We'll accumulate feature vectors (one per sample) in this batch_feats list
            batch_feats = []
            valid_batch_indices = []

            for b in range(B):
                # We can accumulate features for all channels in one row:
                # i.e., we'll end up with one large feature vector per sample.
                sample_feats_all_views = []

                for view in range(C):
                    # -----------------------------------------
                    # 1) Corrupt the center region (60% in each dim)
                    # -----------------------------------------
                    single_view_image = images[b, view:view+1]  # shape: [1, D, H, W]
                    corrupted_images  = single_view_image.clone()

                    # Dimensions to corrupt (60% in each)
                    d = int(D * 0.5)
                    h = int(H * 0.3)
                    w = int(W * 0.5)

                    x = (D - d) // 2
                    y = (H - h) // 2
                    z = (W - w) // 2

                    # zero out center region
                    corrupted_images[..., x:x + d, y:y + h, z:z + w] = 0

                    # -----------------------------------------
                    # 2) Add noise at random diffusion timestep
                    # -----------------------------------------
                    # We want a single random timestep for this single volume
                    t = torch.randint(low=0, high=model.timesteps, size=(1,), device=device)
                    
                    # The model's add_noise() likely expects [B, C, D, H, W],
                    # so we add a batch dimension => shape [1, 1, D, H, W].
                    corrupted_images_4d = corrupted_images.unsqueeze(0)  # [1, 1, D, H, W]
                    x_t, _ = model.add_noise(corrupted_images_4d, t)      # [1, 1, D, H, W]

                    # -----------------------------------------
                    # 3) Reconstruct
                    # -----------------------------------------
                    reconstructed_4d = model(x_t, t)  # [1, 1, D, H, W]
                    # remove the batch dim => shape [1, D, H, W]
                    reconstructed = reconstructed_4d[0]

                    # -----------------------------------------
                    # 4) Subdivide the same center region into 2x2x2 mini-patches
                    #    and extract features from each patch (original & recon).
                    # -----------------------------------------
                    # We'll define slices for each dimension in 2 blocks:
                    #   block 0 -> first half
                    #   block 1 -> second half (including remainder if odd).
                    #
                    # For dimension D (depth):
                    #   block i_0 => [x : x + d//2]
                    #   block i_1 => [x + d//2 : x + d]
                    #
                    # Similar for H, W.

                    for i_ in range(2):
                        d_start = x + i_ * (d // 2)
                        d_end = x + d if i_ == 1 else (x + (i_ + 1) * (d // 2))

                        for j_ in range(2):
                            h_start = y + j_ * (h // 2)
                            h_end = y + h if j_ == 1 else (y + (j_ + 1) * (h // 2))

                            for k_ in range(2):
                                w_start = z + k_ * (w // 2)
                                w_end = z + w if k_ == 1 else (z + (k_ + 1) * (w // 2))

                                # Original mini-patch (shape: [1, d_sl, h_sl, w_sl])
                                original_patch = single_view_image[
                                    ..., d_start:d_end,
                                         h_start:h_end,
                                         w_start:w_end
                                ].float()

                                # Reconstructed mini-patch (same shape)
                                recon_patch = reconstructed[
                                    ..., d_start:d_end,
                                         h_start:h_end,
                                         w_start:w_end
                                ].float()

                                if original_patch.sum() == 0 or torch.isnan(original_patch).any():
                                    print(f"[WARNING] Invalid patch at batch {batch_idx}, skipping.")
                                    continue

                                # 5) Extract first-order stats
                                orig_feats, _ = first_order_and_shape_features(
                                    original_patch,
                                    voxelArrayShift=voxelArrayShift,
                                    pixelSpacing=pixelSpacing
                                )
                                # Insert location as numeric features
                                orig_feats["loc_i"] = float(i_)
                                orig_feats["loc_j"] = float(j_)
                                orig_feats["loc_k"] = float(k_)

                                recon_feats, _ = first_order_and_shape_features(
                                    recon_patch,
                                    voxelArrayShift=voxelArrayShift,
                                    pixelSpacing=pixelSpacing
                                )
                                
                                # Insert location (recon) if you want them separate
                                recon_feats["loc_i_recon"] = float(i_)
                                recon_feats["loc_j_recon"] = float(j_)
                                recon_feats["loc_k_recon"] = float(k_)

                                for key, value in orig_feats.items():
                                    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                                        print(f"[ERROR] NaN in orig_feats at batch {batch_idx}, key={key}")
                                for key, value in recon_feats.items():
                                    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                                        print(f"[ERROR] NaN in recon_feats at batch {batch_idx}, key={key}")# Merge these patch-features into one sub-vector
                                # We order them by sorted keys for reproducibility
                                subpatch_vec = []
                                for key in sorted(orig_feats.keys()):
                                    subpatch_vec.append(orig_feats[key])
                                for key in sorted(recon_feats.keys()):
                                    subpatch_vec.append(recon_feats[key])

                                # Add sub-patch features to the entire channel's vector
                                sample_feats_all_views.extend(subpatch_vec)

                
                
                if len(sample_feats_all_views) > 0:
                    sample_feats_tensor = torch.tensor(
                        sample_feats_all_views, dtype=torch.float, device=device
                    )
                    batch_feats.append(sample_feats_tensor)
                    valid_batch_indices.append(b)
                # Done with all channels for this sample -> convert to tensor, store
                

            if len(batch_feats) > 0:
                try:
                    batch_feats_tensor = torch.stack(batch_feats, dim=0)  # [B, feature_dim]
                    all_features.append(batch_feats_tensor)
                    all_labels.append(labels)
                    valid_indices.extend(valid_batch_indices)
                    print(f"    [DEBUG] Finished batch {batch_idx+1}, features shape = {batch_feats_tensor.shape}")
                except RuntimeError as e:
                    print(f"[ERROR] Failed to stack batch {batch_idx+1}: {e}")
            

    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_features = torch.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
        return all_features, all_labels, valid_indices
    else:
        raise ValueError("[ERROR] No valid features were extracted.")# Concatenate across all batches
    

# -------------------------------------------------------------
# 3) Visualization Function: Original vs. Corrupted vs. Recon
# -------------------------------------------------------------
def visualize_reconstructions(
    diffusion_model,
    images,
    device,
    epoch,
    save_dir
):    
    """
    - Takes a small batch of images (3 views combined).
    - Processes each view independently through the diffusion model.
    - Plots [Original, Corrupted, Reconstructed] for the mid-slice of each view.
    - Saves figure to 'save_dir/epoch_{epoch}.png'.
    """
    diffusion_model.eval()
    with torch.no_grad():
        images = images.to(device)  # Shape: [B, 3, D, H, W]
        B, C, D, H, W = images.shape

        num_to_plot = min(5, B)

        for view in range(C):
            single_view_image = images[:, view:view+1, :, :, :]
            corrupted_images = single_view_image.clone()

            d, h, w = int(D * 0.5), int(H * 0.3), int(W * 0.5)
            x = (D - d) // 2
            y = (H - h) // 2
            z = (W - w) // 2
            corrupted_images[:, :, x:x + d, y:y + h, z:z + w] = 0

            t = torch.randint(low=0, high=diffusion_model.timesteps, size=(B,), device=device)
            x_t, _ = diffusion_model.add_noise(corrupted_images, t)

            # Reconstruct the images
            reconstructed = diffusion_model(x_t, t)

            for b in range(num_to_plot):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Display original image
                axes[0].imshow(single_view_image[b, 0, D // 2, :, :].cpu(), cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis("off")

                # Display corrupted image
                axes[1].imshow(corrupted_images[b, 0, D // 2, :, :].cpu(), cmap='gray')
                axes[1].set_title("Corrupted")
                axes[1].axis("off")

                # Display reconstructed image
                axes[2].imshow(reconstructed[b, 0, D // 2, :, :].cpu(), cmap='gray')
                axes[2].set_title("Reconstructed")
                axes[2].axis("off")

                # Save the figure
                view_dir = os.path.join(save_dir, f"view_{view+1}")
                os.makedirs(view_dir, exist_ok=True)
                save_path = os.path.join(view_dir, f"epoch_{epoch}_image_{b+1}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

    print(f"[DEBUG] Visualization for epoch {epoch} saved in {save_dir}")


# ----------------------------------------------------------------
# Metrics: Sensitivity, Specificity, AUC (per label, then avg)
# ----------------------------------------------------------------

def compute_metrics(probabilities, labels):
    """
    Computes metrics for ACL binary classification.

    probabilities: shape [N,1], each in [0,1]
    labels:        shape [N,1], each 0 or 1

    Returns a dictionary with:
      - Accuracy
      - Sensitivity (Recall)
      - Specificity
      - AUC
      - F1-score
    """
    probabilities_np = probabilities.detach().cpu().numpy().flatten()  # [N]
    labels_np = labels.detach().cpu().numpy().flatten()  # [N]

    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score

    # Convert probabilities to binary predictions (threshold = 0.5)
    preds = (probabilities_np >= 0.5).astype(int)

    # Compute Accuracy
    accuracy = accuracy_score(labels_np, preds)

    # Compute Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0,1]).ravel()

    # Compute Sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Compute AUC
    try:
        auc = roc_auc_score(labels_np, probabilities_np)
    except ValueError:
        auc = 0.0  # Handle cases where only one class is present

    # Compute F1-score
    f1 = f1_score(labels_np, preds, zero_division=0)

    # Print results
    print(f"ACL Metrics: Acc={accuracy:.3f}, Sensitivity={sensitivity:.3f}, "
          f"Specificity={specificity:.3f}, AUC={auc:.3f}, F1={f1:.3f}")

    # Return metrics in a dictionary
    metrics = {
        "label_name": "ACL",
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "f1": f1
    }

    return metrics


# ----------------------------------------------------------------
# Simple classification training for your 3-class prediction
# ----------------------------------------------------------------


def find_best_thresholds_youden(probabilities, labels, min_threshold=0.1):
    """
    Finds the best threshold for ACL classification using Youden's J statistic.
    
    probabilities: Tensor of shape [N,1] - predicted probabilities for ACL
    labels:        Tensor of shape [N,1] - ground truth binary labels for ACL
    
    Returns:
    best_thr: The threshold that maximizes Youden's J statistic
    """
    probabilities_np = probabilities.detach().cpu().numpy().flatten()  # Convert to 1D array
    labels_np = labels.detach().cpu().numpy().flatten()                # Convert to 1D array

    best_thr = min_threshold
    best_j = -1

    # Iterate over possible thresholds from min_threshold to 1.0
    for thr in np.linspace(min_threshold, 1, int((1 - min_threshold) * 100) + 1):
        preds = (probabilities_np >= thr).astype(int)

        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()

        # Calculate sensitivity and specificity
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        # Youden's J statistic
        j_stat = sens + spec - 1

        # Update the best threshold based on J statistic
        if j_stat > best_j:
            best_j = j_stat
            best_thr = thr

    print(f"Best threshold for ACL = {best_thr:.2f}, Youden's J = {best_j:.3f}")

    return best_thr







def evaluate_model(
    model, 
    loader, 
    device, 
    threshold=0.47, 
    save_path="patient_feature_importance.json"
):
    model.eval()
    all_probs, all_labels = [], []
    
    # This list will store feature importances for all samples in the validation set
    all_patient_feature_importances = []
    
    with torch.no_grad():
        for batch_idx, (images_3d, feats_1824, labels) in enumerate(loader):
            images_3d = images_3d.to(device)
            feats_1824 = feats_1824.to(device)
            labels = labels[:, 1].unsqueeze(1).to(device)

            # Extract the raw feature importance vector from the CNN model
            feature_importance = model.cnn_model(images_3d)  # [B, 1824]

            # Apply hard selection: set importance < 0.40 to 0, >= 0.40 to 1
            importance_binary = torch.where(
                feature_importance < 0.40, 
                torch.zeros_like(feature_importance), 
                feature_importance
            )
            importance_binary = torch.where(
                importance_binary >= 0.40, 
                torch.ones_like(importance_binary), 
                importance_binary
            )

            nonzero_counts = (importance_binary > 0).sum(dim=1)  # count non-zero per sample
            print(f"Batch {batch_idx}: non-zero importance counts per sample: {nonzero_counts}")

            # Create masked features
            masked_feats = feats_1824 * importance_binary

            # Predict
            logits = model.lr_model(masked_feats)
            probs = torch.sigmoid(logits)

            # Collect predictions and labels for metric computation
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            # -----------------------------
            # SAVE FEATURE IMPORTANCES
            # -----------------------------
            # Loop over each sample in this batch to store its feature importance
            batch_size = feature_importance.shape[0]
            for i in range(batch_size):
                # Convert importance vector to CPU numpy and sort by descending importance
                importance_vector = feature_importance[i]
                importance_values = importance_vector.cpu().numpy().tolist()
                
                sorted_indices = torch.argsort(
                    importance_vector, descending=True
                ).cpu().numpy().tolist()
                
                # Build the list of (index, importance) pairs
                importance_pairs = [
                    {"index": idx, "importance": importance_values[idx]} 
                    for idx in sorted_indices
                ]

                # You could optionally store more info (e.g. patient ID, 
                # global index, original label, etc.). Here we just store 
                # batch/sample indices for clarity.
                patient_data = {
                    "batch_idx": batch_idx,
                    "sample_in_batch": i,
                    "importance_pairs": importance_pairs
                }
                all_patient_feature_importances.append(patient_data)
    
    # -------------
    # SAVE TO JSON
    # -------------
    if all_patient_feature_importances:
        with open(save_path, 'w') as f:
            json.dump(all_patient_feature_importances, f, indent=4)
        print(f"[INFO] Feature importance for all validation samples saved to {save_path}")
    else:
        print("[INFO] No feature importances were saved (empty loader or no data).")

    # -------------------------
    # COMPUTE FINAL METRICS
    # -------------------------
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = (all_probs >= threshold).float()

    # Example: using compute_metrics with preds & labels
    metrics_dict = compute_metrics(preds, all_labels)
    return metrics_dict



def check_for_nans_or_infs(tensor, name):
    if torch.isnan(tensor).any():
        print(f"[ERROR] {name} contains NaN values.")
    if torch.isinf(tensor).any():
        print(f"[ERROR] {name} contains Inf values.")


def cross_validate_model(model, dataset, device, output_log_file, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_metrics = []
    feature_importance_saved = False
    
    labels_np = np.array([label.numpy() for _, _, label in dataset])  # Expecting 3 values (images, features, labels)

    with open(output_log_file, 'w') as log_file:
        log_file.write("[INFO] Cross-Validation Results:\n\n")

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np.argmax(axis=1))):
            print(f"\n[INFO] Fold {fold + 1}/{n_splits}")
            log_file.write(f"Fold {fold + 1}/{n_splits}\n")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(5):  # Train for 5 epochs per fold
                for images_3d, feats_1824, labels_3 in train_loader:
                    images_3d = images_3d.to(device)
                    feats_1824 = feats_1824.to(device)
                    labels_3 = labels_3.to(device)

                    optimizer.zero_grad()
                    logits = model(images_3d, feats_1824)
                    loss = nn.BCEWithLogitsLoss()(logits, labels_3[:, 1].unsqueeze(1).float())  # Extract ACL label

                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                if not feature_importance_saved:
                    metrics_dict = evaluate_model(
                        model=model,
                        loader=val_loader,
                        device=device,
                        save_path=os.path.join(model_save_path, "patient_feature_importance.json")
                    )
                    feature_importance_saved = True  # Set flag to True after saving
                else:
                    metrics_dict = evaluate_model(
                        model=model,
                        loader=val_loader,
                        device=device
                    )
            all_metrics.append(metrics_dict)


        # Compute mean and std across folds
        final_metrics = {
            key: {'mean': np.mean([fold[key] for fold in all_metrics], axis=0),
                  'std': np.std([fold[key] for fold in all_metrics], axis=0)}
            for key in all_metrics[0] if isinstance(all_metrics[0][key], list)
        }

        log_file.write("\n[INFO] Final Cross-Validation Results:\n")
        for key, values in final_metrics.items():
            result_line = f"  {key}: {values['mean']:.3f} ± {values['std']:.3f}\n"
            print(result_line.strip())
            log_file.write(result_line)

    print(f"\n[INFO] Cross-validation results saved in: {output_log_file}")


# -----------------------
# Main Script Example
# -----------------------
if __name__ == "__main__":
    """
    Example usage:
      1. Instantiate your MRIDataset for 'train' phase.
      2. Load the pretrained DDPM model.
      3. Extract features by corrupting + reconstructing.
      4. Train LesionClassifier using those features.
    
    Adjust the paths, CSV files, and phases for your real environment.
    """

    from sklearn.model_selection import train_test_split
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_path = '.../path/to/best_diffusion_model.pth'

    diffusion_model = DDPM3D().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    diffusion_model.load_state_dict(checkpoint)
    diffusion_model.eval()
    print(f"Loaded pretrained DDPM3D model from: {model_path}")

    visualization_save_path = '.../path/to/visualizations'
    os.makedirs(visualization_save_path, exist_ok=True)
    
    model_save_path = '.../path/to/model'
    os.makedirs(model_save_path, exist_ok=True)   

    best_model_path = os.path.join(model_save_path, "best_cnn_lr_mask_model_softmax_2*2_acl_clinical.pth")
    test_features_path = os.path.join(model_save_path, "test_features_1824_acl_clinical.pt")
    test_labels_path = os.path.join(model_save_path, "test_labels_3_acl_clinical.pt")
    #thresholds_norm_path = os.path.join(model_save_path, "thresholds_and_normalization_softmax_2*2.json")
    norm_stats_path = os.path.join(model_save_path, "normalization_stats_acl_clinical.json")
 
    with open(norm_stats_path, 'r') as f:
        norm_data = json.load(f)
        means = torch.tensor(norm_data["means"], dtype=torch.float32, device=device)
        stds  = torch.tensor(norm_data["stds"], dtype=torch.float32, device=device)

    #selected_features_mask = np.load(os.path.join(model_save_path, "selected_features_mask_men.npy"))
    
    #best_thresholds = 0.5

    root_dir = ".../path/to/MRNet-v1.0_nii"
    
    labels_files_test = {
        'abnormal': os.path.join(root_dir, 'valid-abnormal.csv'),
        'acl':      os.path.join(root_dir, 'valid-acl.csv'),
        'meniscus': os.path.join(root_dir, 'valid-meniscus.csv')
    }

    test_mri_dataset = MRIDataset(
        root_dir=root_dir,
        labels_files=labels_files_test,
        phase='valid',
        views=('coronal_reg', 'axial_reg', 'sagittal_reg'),
        transform=None,
        target_size=(32, 128, 128)
    )
    print(f"[INFO] Created test dataset with {len(test_mri_dataset)} samples.")

    test_ddpm_loader = DataLoader(test_mri_dataset, batch_size=8, shuffle=False)
 
    if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
        print("[INFO] Loading saved test features and labels...")
        test_features_1824 = torch.load(test_features_path, map_location='cuda:1')
        test_labels_3 = torch.load(test_labels_path, map_location='cuda:1')
        print(f"[INFO] Loaded test features shape: {test_features_1824.shape}")
        print(f"[INFO] Loaded test labels shape: {test_labels_3.shape}")
        valid_indices_test = list(range(len(test_features_1824)))
    else:
        print("[INFO] Extracting test radiomics features...")
        test_features_1824, test_labels_3, valid_indices_test = extract_features_with_pretrained_ddpm(
            model=diffusion_model,
            loader=test_ddpm_loader,
            device=device,
            voxelArrayShift=0,
            pixelSpacing=[1.0, 1.0, 1.0]
        )

        print(f"[INFO] Extracted test features shape: {test_features_1824.shape}")
        print(f"[INFO] Extracted test labels shape: {test_labels_3.shape}")

        # Save test features and labels
        torch.save(test_features_1824, test_features_path)
        torch.save(test_labels_3, test_labels_path)
        print(f"[INFO] Test features saved to: {test_features_path}")
        print(f"[INFO] Test labels saved to: {test_labels_path}")

    #help me load means and std

    test_features_1824 = (test_features_1824 - means) / (stds + 1e-8)

    test_mri_dataset = torch.utils.data.Subset(test_mri_dataset, valid_indices_test)
    print(f"[INFO] Filtered dataset length: {len(test_mri_dataset)}")
    test_combined_dataset = CombinedMRIFeatureDataset(test_mri_dataset, test_features_1824)
    test_loader = DataLoader(test_combined_dataset, batch_size=16, shuffle=True)
    #test_combined_dataset = CombinedMRIFeatureDataset(test_mri_dataset, test_features_1842)
    #test_loader = DataLoader(test_combined_dataset, batch_size=8, shuffle=False)
    print("[INFO] Created test DataLoader.")

    cnn_model = GlobalMaskedFeatureSelector(in_channels=3, out_features=1824)
    lr_model  = InteractionLogisticRegression(input_size=1824, output_size=1)
    combined_model = CNNWithGlobalMasking(cnn_model=cnn_model, lr_model=lr_model)
    combined_model.load_state_dict(torch.load(best_model_path, map_location='cuda:1'))
    combined_model.to(device)

    output_log_file = os.path.join(model_save_path, "evaluation_metrics_softmax_2*2_acl_clinical.txt")
    
    cross_validate_model(combined_model, test_combined_dataset, device, output_log_file, n_splits=5)


    #print("[INFO] Evaluating on test set...")
    #metrics_dict, results_summary = evaluate_model(
    #    model=combined_model,
    #    loader=test_loader,
    #    device=device,
    #    thresholds=best_thresholds,
    #    selected_features_mask=selected_features_mask,
    #    output_log_file=output_log_file   # Pass the best thresholds here
    #)

    
