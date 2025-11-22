# Import all required libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.spatial.distance import jensenshannon
import os
import warnings
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_dataset_splits(data_dir='dataset'):
    """
    Load pre-split dataset files from the specified directory.
    
    Dataset structure:
    - train/val/test/test2 splits for training and evaluation
    - original: Complete feature vectors (ground truth)
    - imputed: Vectors with missing values replaced by -1 
    - missing_mask: Boolean mask (True=observed, False=missing)
    
    For test2: We only have imputed and missing_mask (no original for evaluation)
    """
    splits = {}
    
    # Define the expected split types
    split_types = ['train', 'val', 'test', 'test2']
    
    print(f"Loading dataset splits from: {data_dir}")
    
    for split_type in split_types:
        print(f"\nLoading {split_type} split...")
        splits[split_type] = {}
        
        # Load original data (ground truth) - not available for test2
        if split_type != 'test2':
            original_path = os.path.join(data_dir, f'{split_type}_original.csv')
            if os.path.exists(original_path):
                splits[split_type]['original'] = pd.read_csv(original_path).values
                print(f"  ✓ {split_type}_original.csv: {splits[split_type]['original'].shape}")
            else:
                print(f"  ✗ {original_path} not found")
        
        # Load imputed data (with -1 for missing values)
        imputed_path = os.path.join(data_dir, f'{split_type}_imputed.csv')
        if os.path.exists(imputed_path):
            splits[split_type]['imputed'] = pd.read_csv(imputed_path).values
            print(f"  ✓ {split_type}_imputed.csv: {splits[split_type]['imputed'].shape}")
        else:
            print(f"  ✗ {imputed_path} not found")
            
        # Load missing mask
        mask_path = os.path.join(data_dir, f'{split_type}_missing_mask.csv')
        if os.path.exists(mask_path):
            splits[split_type]['missing_mask'] = pd.read_csv(mask_path).values.astype(bool)
            print(f"  ✓ {split_type}_missing_mask.csv: {splits[split_type]['missing_mask'].shape}")
        else:
            print(f"  ✗ {mask_path} not found")
    
    return splits

# Calculate metrics for missing values only
def calculate_jsd_and_mean_diff(imputed_values, ground_truth_values, feature_name):
    """
    Calculate Jensen-Shannon Divergence and mean difference between imputed and ground truth values.
    """
    if len(imputed_values) == 0 or len(ground_truth_values) == 0:
        return np.nan, np.nan
    
    # Remove any NaN or infinite values
    imputed_clean = imputed_values[np.isfinite(imputed_values)]
    gt_clean = ground_truth_values[np.isfinite(ground_truth_values)]
    
    if len(imputed_clean) == 0 or len(gt_clean) == 0:
        return np.nan, np.nan
    
    # Calculate mean difference
    mean_diff = abs(imputed_clean.mean() - gt_clean.mean())
    
    # Calculate Jensen-Shannon divergence using histograms
    try:
        # Create histograms with same bins
        data_range = (min(imputed_clean.min(), gt_clean.min()), 
                     max(imputed_clean.max(), gt_clean.max()))
        
        if data_range[1] == data_range[0]:
            return mean_diff, 0.0  # No divergence if all values are the same
        
        bins = np.linspace(data_range[0], data_range[1], 50)
        
        # Get histogram probabilities
        hist_imputed, _ = np.histogram(imputed_clean, bins=bins, density=True)
        hist_gt, _ = np.histogram(gt_clean, bins=bins, density=True)
        
        # Normalize to probabilities
        hist_imputed = hist_imputed + 1e-10  # Add small epsilon to avoid zeros
        hist_gt = hist_gt + 1e-10
        hist_imputed = hist_imputed / hist_imputed.sum()
        hist_gt = hist_gt / hist_gt.sum()
        
        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(hist_imputed, hist_gt)
        
        return mean_diff, js_div
        
    except Exception as e:
        print(f"Error calculating JSD for {feature_name}: {e}")
        return mean_diff, np.nan


def plot_distribution_comparison(test_imputations_denorm, test_original_denorm, test_masks, feature_names, n_features=25):
    """
    Create distribution comparison plots for random features in a 5x5 grid.
    
    Args:
        test_imputations_denorm: Denormalized imputed values
        test_original_denorm: Denormalized ground truth values  
        test_masks: Binary masks (1=observed, 0=missing)
        feature_names: List of feature names
        n_features: Number of features to plot (default 25 for 5x5 grid)
    """
    # Find features that have missing values
    features_with_missing = []
    for i, feature_name in enumerate(feature_names):
        missing_positions = (test_masks[:, i] == 0)  # 0 = missing in model tensors
        if missing_positions.sum() > 0:
            features_with_missing.append((i, feature_name))
    
    if len(features_with_missing) < n_features:
        n_features = len(features_with_missing)
        print(f"Only {n_features} features have missing values, showing all of them.")
    
    # Randomly select features to plot
    selected_indices = range(len(features_with_missing))
    selected_features = [features_with_missing[i] for i in selected_indices]
    
    # Create grid
    fig, axes = plt.subplots(8, 5, figsize=(20, 16))
    fig.suptitle('Distribution Comparison: Dataset vs Imputed Values', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, (feature_idx, feature_name) in enumerate(selected_features):
        if idx > 40:  # Safety check for grid
            break
            
        # Get imputed and ground truth values for missing positions only
        missing_positions = (test_masks[:, feature_idx] == 0)  # 0 = missing in model tensors
        
        if missing_positions.sum() > 0:
            imputed_values = test_imputations_denorm[missing_positions, feature_idx]
            ground_truth_values = test_original_denorm[missing_positions, feature_idx]
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(imputed_values) & np.isfinite(ground_truth_values)
            imputed_clean = imputed_values[valid_mask]
            gt_clean = ground_truth_values[valid_mask]
            
            if len(imputed_clean) > 0 and len(gt_clean) > 0:
                # Create histograms
                ax = axes[idx]
                
                # Calculate bins for both distributions
                all_values = np.concatenate([imputed_clean, gt_clean])
                bins = np.linspace(all_values.min(), all_values.max(), 20)  # Fewer bins for smaller plots
                
                # Plot histograms
                ax.hist(gt_clean, bins=bins, alpha=0.7, label='Dataset', 
                    color='skyblue', density=True, edgecolor='black', linewidth=0.3)
                ax.hist(imputed_clean, bins=bins, alpha=0.7, label='Imputed', 
                    color='lightcoral', density=True, edgecolor='black', linewidth=0.3)
                
                # Add statistical information
                gt_mean, gt_std = gt_clean.mean(), gt_clean.std()
                imp_mean, imp_std = imputed_clean.mean(), imputed_clean.std()
                correlation = np.corrcoef(gt_clean, imputed_clean)[0, 1] if len(gt_clean) > 1 else 0

                # Calculate MMD (Maximum Mean Discrepancy)
                def rbf_kernel(X, Y, gamma=1.0):
                    """RBF kernel for MMD calculation"""
                    XX = np.sum(X**2, axis=1, keepdims=True)
                    YY = np.sum(Y**2, axis=1, keepdims=True)
                    XY = np.dot(X, Y.T)
                    distances = XX + YY.T - 2*XY
                    return np.exp(-gamma * distances)
                
                def mmd_rbf(X, Y, gamma=1.0):
                    """Calculate MMD with RBF kernel"""
                    X = X.reshape(-1, 1)
                    Y = Y.reshape(-1, 1)
                    
                    m, n = len(X), len(Y)
                    
                    K_XX = rbf_kernel(X, X, gamma)
                    K_YY = rbf_kernel(Y, Y, gamma)
                    K_XY = rbf_kernel(X, Y, gamma)
                    
                    mmd = (np.sum(K_XX) / (m * m) + 
                           np.sum(K_YY) / (n * n) - 
                           2 * np.sum(K_XY) / (m * n))
                    return np.sqrt(max(mmd, 0))  # Ensure non-negative
                
                try:
                    mmd_value = mmd_rbf(gt_clean, imputed_clean)
                except:
                    mmd_value = np.nan

                # Calculate Jensen-Shannon Divergence
                try:
                    # Create histograms with same bins for JSD
                    data_range = (min(gt_clean.min(), imputed_clean.min()), 
                                 max(gt_clean.max(), imputed_clean.max()))
                    
                    if data_range[1] == data_range[0]:
                        jsd_value = 0.0  # No divergence if all values are the same
                    else:
                        bins = np.linspace(data_range[0], data_range[1], 30)
                        
                        # Get histogram probabilities
                        hist_gt, _ = np.histogram(gt_clean, bins=bins, density=True)
                        hist_imp, _ = np.histogram(imputed_clean, bins=bins, density=True)
                        
                        # Normalize to probabilities
                        hist_gt = hist_gt + 1e-10  # Add small epsilon to avoid zeros
                        hist_imp = hist_imp + 1e-10
                        hist_gt = hist_gt / hist_gt.sum()
                        hist_imp = hist_imp / hist_imp.sum()
                        
                        # Calculate Jensen-Shannon divergence
                        jsd_value = jensenshannon(hist_gt, hist_imp)
                except:
                    jsd_value = np.nan

                # Add vertical lines for means
                ax.axvline(gt_mean, color='blue', linestyle='--', alpha=0.8, linewidth=1, label='Dataset Mean' if idx == 0 else "")
                ax.axvline(imp_mean, color='red', linestyle='--', alpha=0.8, linewidth=1, label='Imputed Mean' if idx == 0 else "")
                
                # Set labels and title (smaller font for 5x5 grid)
                ax.set_xlabel(f'{feature_name[:15]}', fontsize=8)  # Truncate long names
                ax.set_ylabel('Density', fontsize=8)
                ax.tick_params(labelsize=7)
                
                # Add correlation, MMD, and JSD as title
                ax.set_title(f'R²={correlation:.3f}, MMD={mmd_value:.3f}, JSD={jsd_value:.3f}', fontsize=7, fontweight='bold')
                
                # Add legend only to first plot
                if idx == 0:
                    ax.legend(fontsize=7, loc='upper right')
                
                ax.grid(True, alpha=0.3)

            else:
                axes[idx].text(0.5, 0.5, f'{feature_name[:15]}\nNo valid data', 
                            ha='center', va='center', transform=axes[idx].transAxes, fontsize=8)
                axes[idx].set_title(f'{feature_name[:15]} - No Valid Data', fontsize=8)
        else:
            axes[idx].text(0.5, 0.5, f'{feature_name[:15]}\nNo missing values', 
                        ha='center', va='center', transform=axes[idx].transAxes, fontsize=8)
            axes[idx].set_title(f'{feature_name[:15]} - No Missing Values', fontsize=8)

    # Hide any unused subplots
    for idx in range(len(selected_features), 25):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_prediction_scatter(test_imputations, test_originals, test_masks, feature_names, n_features=25):
    """
    Create scatter plots showing predicted vs ground truth values for random features in a 5x5 grid.
    
    Args:
        test_imputations: Model predictions [n_samples, n_features]
        test_originals: Ground truth values [n_samples, n_features]
        test_masks: Binary masks (1=observed, 0=missing) [n_samples, n_features]
        feature_names: List of feature names
        n_features: Number of random features to plot (default 25 for 5x5 grid)
    """
    # Create masks for missing values (where we need to evaluate imputation)
    missing_mask = (test_masks == 0)  # True where values were missing
    
    # Find features that have missing values
    features_with_missing = [i for i in range(len(feature_names)) if missing_mask[:, i].sum() > 0]
    
    if len(features_with_missing) < n_features:
        n_features = len(features_with_missing)
        print(f"Only {n_features} features have missing values, showing all of them.")
    
    selected_features = range(len(features_with_missing))
    
    # Create subplots
    fig, axes = plt.subplots(8, 5, figsize=(20, 16))
    fig.suptitle('Predicted Vs Ground Truth Values (Missing Positions Only)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(selected_features):
        if idx > 40:  # Safety check
            break
            
        ax = axes[idx]
        
        # Get missing positions for this feature
        feature_missing_mask = missing_mask[:, feature_idx]
        
        if feature_missing_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{feature_names[feature_idx][:15]}\nNo missing values', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'{feature_names[feature_idx][:15]} - No Missing', fontsize=8)
            continue
            
        # Get predicted and ground truth values for missing positions only
        predicted_values = test_imputations[feature_missing_mask, feature_idx]
        true_values = test_originals[feature_missing_mask, feature_idx]
        
        # Create scatter plot with smaller points for 5x5 grid
        ax.scatter(true_values, predicted_values, alpha=0.6, s=10, color='steelblue', edgecolors='navy', linewidth=0.3)
        
        # Add perfect prediction line (y=x)
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.5, label='Perfect' if idx == 0 else "")
        
        # Calculate and display metrics
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        try:
            correlation = np.corrcoef(true_values, predicted_values)[0, 1]
        except:
            correlation = np.nan
        
        # Set labels and title (smaller fonts for 5x5 grid)
        ax.set_xlabel('Dataset', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.set_title(f'{feature_names[feature_idx][:15]}\nR²={correlation:.3f}, MSE={mse:.4f}', 
                    fontsize=8, fontweight='bold')
        
        # Adjust tick labels
        ax.tick_params(labelsize=7)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add text box with number of missing values (smaller for 5x5)
        n_missing = feature_missing_mask.sum()
        ax.text(0.05, 0.95, f'n={n_missing}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=7)
    
    # Add legend only to the first subplot to avoid clutter
    if len(selected_features) > 0:
        axes[0].legend(loc='lower right', fontsize=7)
    
    # Hide any unused subplots
    for idx in range(len(selected_features), 25):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
