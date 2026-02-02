"""
Visualization utilities for CFD Surrogate Model
Plot predictions, compare with ground truth, analyze errors
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
from typing import Optional, Tuple, List
import json

from config import MODEL_SAVE_DIR, OUTPUT_COLUMNS
from data_loader import CFDDataLoader
from inference import CFDPredictor


def plot_field_comparison(
    coords: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    field_idx: int = 0,
    field_name: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 5),
    max_points: int = 500000  # Maximum points before downsampling
):
    """
    Plot predicted vs ground truth field comparison.
    Memory-optimized for large meshes.
    
    Args:
        coords: (N, 2) mesh coordinates
        predicted: (N, num_fields) predicted values
        ground_truth: (N, num_fields) ground truth values
        field_idx: Which field to plot (0-4)
        field_name: Name of the field (uses config if None)
        save_path: Path to save figure
        figsize: Figure size
        max_points: Maximum points to plot (downsamples if exceeded)
    """
    if field_name is None:
        field_name = OUTPUT_COLUMNS[field_idx]
    
    x = coords[:, 0]
    y = coords[:, 1]
    pred = predicted[:, field_idx]
    truth = ground_truth[:, field_idx]
    error = pred - truth
    
    # Downsample if mesh is too large
    n_points = len(x)
    downsample = False
    if n_points > max_points:
        downsample = True
        # Use random sampling for better representation
        np.random.seed(42)
        indices = np.random.choice(n_points, max_points, replace=False)
        x = x[indices]
        y = y[indices]
        pred = pred[indices]
        truth = truth[indices]
        error = error[indices]
        print(f"  Downsampled {n_points:,} points to {max_points:,} for visualization")
    
    # Create triangulation for unstructured mesh
    use_tricontourf = True
    try:
        tri = Triangulation(x, y)
    except Exception as e:
        print(f"  Warning: Triangulation failed ({e}), using scatter plot")
        tri = None
        use_tricontourf = False
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Predicted
    if tri is not None and use_tricontourf:
        # tricontourf is more memory-efficient than tripcolor
        im0 = axes[0].tricontourf(tri, pred, levels=50, cmap='jet')
    else:
        # Fallback to scatter
        im0 = axes[0].scatter(x, y, c=pred, s=0.5, cmap='jet', marker='.')
    axes[0].set_title(f'Predicted {field_name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])
    
    # Ground Truth
    if tri is not None and use_tricontourf:
        im1 = axes[1].tricontourf(tri, truth, levels=50, cmap='jet')
    else:
        im1 = axes[1].scatter(x, y, c=truth, s=0.5, cmap='jet', marker='.')
    axes[1].set_title(f'Ground Truth {field_name}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])
    
    # Error
    vmax = np.max(np.abs(error))
    if tri is not None and use_tricontourf:
        im2 = axes[2].tricontourf(tri, error, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    else:
        im2 = axes[2].scatter(x, y, c=error, s=0.5, cmap='RdBu_r', vmin=-vmax, vmax=vmax, marker='.')
    axes[2].set_title(f'Error ({field_name})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])
    
    # Add statistics (use original arrays for accuracy)
    original_error = predicted[:, field_idx] - ground_truth[:, field_idx]
    original_truth = ground_truth[:, field_idx]
    
    rel_error = np.sqrt(np.mean(original_error**2)) / (np.std(original_truth) + 1e-8)
    max_error = np.max(np.abs(original_error))
    mae = np.mean(np.abs(original_error))
    
    title = f'{field_name}: MAE={mae:.4e}, Max Error={max_error:.4e}, Rel L2={rel_error:.2%}'
    if downsample:
        title += f' [{n_points:,} points, viz downsampled to {max_points:,}]'
    fig.suptitle(title, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        # Close figure to free memory
        plt.close(fig)
    
    return fig


def plot_all_fields(
    coords: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    save_dir: Optional[Path] = None,
    prefix: str = ''
):
    """Plot comparison for all output fields."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, field_name in enumerate(OUTPUT_COLUMNS):
        save_path = save_dir / f'{prefix}{field_name}_comparison.png' if save_dir else None
        plot_field_comparison(
            coords, predicted, ground_truth,
            field_idx=i, field_name=field_name,
            save_path=save_path
        )
    
    if not save_dir:
        plt.show()


def plot_training_history(
    history_path: Path = MODEL_SAVE_DIR / 'training_history.json',
    save_path: Optional[Path] = None
):
    """Plot training and validation loss curves."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].semilogy(epochs, history['train_loss'], label='Train')
    axes[0].semilogy(epochs, history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].semilogy(epochs, history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    return fig


def compute_error_statistics(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> dict:
    """
    Compute comprehensive error statistics.
    
    Returns dict with:
        - mae: Mean Absolute Error per field
        - rmse: Root Mean Square Error per field
        - rel_l2: Relative L2 error per field
        - max_error: Maximum absolute error per field
        - r2: R² score per field
    """
    stats = {}
    
    for i, field_name in enumerate(OUTPUT_COLUMNS):
        pred = predicted[:, i]
        truth = ground_truth[:, i]
        
        mae = np.mean(np.abs(pred - truth))
        rmse = np.sqrt(np.mean((pred - truth)**2))
        rel_l2 = rmse / (np.std(truth) + 1e-8)
        max_err = np.max(np.abs(pred - truth))
        
        # R² score
        ss_res = np.sum((truth - pred)**2)
        ss_tot = np.sum((truth - np.mean(truth))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        stats[field_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'rel_l2': float(rel_l2),
            'max_error': float(max_err),
            'r2': float(r2)
        }
    
    return stats


def print_error_report(stats: dict):
    """Print formatted error statistics."""
    print("\n" + "=" * 80)
    print("ERROR STATISTICS REPORT")
    print("=" * 80)
    
    print(f"\n{'Field':<25} {'MAE':<12} {'RMSE':<12} {'Rel L2':<10} {'R²':<10}")
    print("-" * 80)
    
    for field_name, field_stats in stats.items():
        print(f"{field_name:<25} "
              f"{field_stats['mae']:<12.4e} "
              f"{field_stats['rmse']:<12.4e} "
              f"{field_stats['rel_l2']:<10.2%} "
              f"{field_stats['r2']:<10.4f}")
    
    print("=" * 80)


def validate_model(
    model_path: Path = MODEL_SAVE_DIR / 'best_model.pt',
    sample_ids: Optional[List[int]] = None,
    save_dir: Optional[Path] = None
):
    """
    Comprehensive model validation.
    
    Args:
        model_path: Path to trained model
        sample_ids: Sample IDs to validate on (uses all available if None)
        save_dir: Directory to save visualization outputs
    """
    # Load predictor
    predictor = CFDPredictor(model_path=model_path)
    
    # Load data
    data_loader = CFDDataLoader()
    data_loader.compute_normalization_stats()
    
    if sample_ids is None:
        sample_ids = data_loader.get_available_sample_ids()
    
    if not sample_ids:
        print("No simulations available for validation!")
        return
    
    all_stats = {}
    
    for sample_id in sample_ids:
        print(f"\nValidating Sample_ID {sample_id}...")
        
        # Get ground truth
        params = data_loader.get_input_params(sample_id)
        coords, ground_truth = data_loader.get_cfd_output(sample_id)
        
        # Predict
        predictions = predictor.predict(params, coords)
        
        # Compute statistics
        stats = compute_error_statistics(predictions, ground_truth)
        all_stats[sample_id] = stats
        print_error_report(stats)
        
        # Plot
        if save_dir:
            sample_save_dir = save_dir / f'sample_{sample_id}'
            plot_all_fields(
                coords, predictions, ground_truth,
                save_dir=sample_save_dir,
                prefix=f'sample{sample_id}_'
            )
    
    return all_stats


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize CFD Surrogate Model Results')
    parser.add_argument('--action', type=str, default='validate',
                        choices=['validate', 'history'],
                        help='Action to perform')
    parser.add_argument('--sample_id', type=int, default=None,
                        help='Specific sample ID to visualize')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    model_path = Path(args.model) if args.model else MODEL_SAVE_DIR / 'best_model.pt'
    save_dir = Path(args.save_dir) if args.save_dir else MODEL_SAVE_DIR / 'visualizations'
    
    if args.action == 'history':
        plot_training_history(save_path=save_dir / 'training_history.png')
    
    elif args.action == 'validate':
        sample_ids = [args.sample_id] if args.sample_id else None
        validate_model(model_path=model_path, sample_ids=sample_ids, save_dir=save_dir)


if __name__ == "__main__":
    main()
