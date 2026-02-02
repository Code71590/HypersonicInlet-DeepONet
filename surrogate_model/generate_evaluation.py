"""
Comprehensive DeepONet Model Evaluation Script
Generates all metrics, statistical graphs, and visual comparisons
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_SAVE_DIR, OUTPUT_COLUMNS, DEVICE
from data_loader import CFDDataLoader
from inference import CFDPredictor

# Output directory
OUTPUT_DIR = Path(__file__).parent / "evaluation_output"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
(OUTPUT_DIR / "contours").mkdir(exist_ok=True)


def compute_mape(predicted: np.ndarray, ground_truth: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = np.abs(ground_truth) > epsilon
    if not np.any(mask):
        return float('nan')
    return np.mean(np.abs((ground_truth[mask] - predicted[mask]) / ground_truth[mask])) * 100


def compute_all_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """Compute comprehensive error statistics for all fields."""
    metrics = {}
    
    for i, field_name in enumerate(OUTPUT_COLUMNS):
        pred = predicted[:, i]
        truth = ground_truth[:, i]
        
        # Filter out NaN and Inf values
        valid_mask = np.isfinite(pred) & np.isfinite(truth)
        
        if not np.any(valid_mask):
            print(f"  WARNING: All values are NaN/Inf for {field_name}")
            metrics[field_name] = {
                'r2': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'mape': float('nan'),
                'max_error': float('nan'),
                'rel_l2': float('nan')
            }
            continue
        
        # Use only valid values
        pred_valid = pred[valid_mask]
        truth_valid = truth[valid_mask]
        
        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0:
            print(f"  WARNING: {invalid_count} NaN/Inf values filtered out for {field_name}")
        
        # R² Score
        ss_res = np.sum((truth_valid - pred_valid)**2)
        ss_tot = np.sum((truth_valid - np.mean(truth_valid))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 1e-8 else 0.0
        
        # Clip R² to reasonable range
        r2 = np.clip(r2, -10, 1.0)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_valid - truth_valid)**2))
        
        # MAE
        mae = np.mean(np.abs(pred_valid - truth_valid))
        
        # MAPE
        mape = compute_mape(pred_valid, truth_valid)
        
        # Max Error
        max_err = np.max(np.abs(pred_valid - truth_valid))
        
        # Relative L2 Error
        truth_std = np.std(truth_valid)
        rel_l2 = rmse / truth_std if truth_std > 1e-8 else float('nan')
        
        metrics[field_name] = {
            'r2': float(r2) if np.isfinite(r2) else 0.0,
            'rmse': float(rmse) if np.isfinite(rmse) else float('nan'),
            'mae': float(mae) if np.isfinite(mae) else float('nan'),
            'mape': float(mape) if np.isfinite(mape) else float('nan'),
            'max_error': float(max_err) if np.isfinite(max_err) else float('nan'),
            'rel_l2': float(rel_l2) if np.isfinite(rel_l2) else float('nan')
        }
    
    return metrics



def measure_inference_time(predictor: CFDPredictor, params: np.ndarray, 
                           coords: np.ndarray, n_runs: int = 10) -> float:
    """Measure average inference time in milliseconds."""
    times = []
    
    # Warmup
    for _ in range(3):
        _ = predictor.predict(params, coords)
    
    # Timed runs
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = predictor.predict(params, coords)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times)


def plot_scatter_comparison(predicted: np.ndarray, ground_truth: np.ndarray, 
                            save_dir: Path):
    """Create scatter plots of Predicted vs Ground Truth for each field."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, field_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[i]
        pred = predicted[:, i]
        truth = ground_truth[:, i]
        
        # Downsample for plotting if too many points
        n_points = len(pred)
        if n_points > 50000:
            idx = np.random.choice(n_points, 50000, replace=False)
            pred_plot = pred[idx]
            truth_plot = truth[idx]
        else:
            pred_plot = pred
            truth_plot = truth
        
        ax.scatter(truth_plot, pred_plot, alpha=0.3, s=1, c='blue')
        
        # Perfect prediction line
        min_val = min(truth_plot.min(), pred_plot.min())
        max_val = max(truth_plot.max(), pred_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Compute R²
        ss_res = np.sum((truth - pred)**2)
        ss_tot = np.sum((truth - np.mean(truth))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{field_name}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "scatter_all_fields.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_all_fields.png")


def plot_error_histograms(predicted: np.ndarray, ground_truth: np.ndarray,
                          save_dir: Path):
    """Create error histograms for each field."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, field_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[i]
        error = predicted[:, i] - ground_truth[:, i]
        
        ax.hist(error, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', lw=2)
        
        mean_err = np.mean(error)
        std_err = np.std(error)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{field_name}\nμ={mean_err:.2e}, σ={std_err:.2e}')
        ax.grid(True, alpha=0.3)
    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "error_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_histograms.png")


def plot_metrics_comparison(metrics: Dict, save_dir: Path):
    """Create bar chart comparing metrics across variables."""
    fields = list(metrics.keys())
    
    # Debug: Print metrics to verify data
    print(f"  Plotting metrics for {len(fields)} fields: {fields}")
    
    r2_scores = [metrics[f]['r2'] for f in fields]
    rmse_vals = [metrics[f]['rmse'] for f in fields]
    mape_vals = [metrics[f]['mape'] for f in fields]
    
    print(f"  R² range: [{min(r2_scores):.4f}, {max(r2_scores):.4f}]")
    print(f"  RMSE range: [{min(rmse_vals):.4e}, {max(rmse_vals):.4e}]")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # R² Scores
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(fields)))
    x_pos = np.arange(len(fields))
    bars = axes[0].bar(x_pos, r2_scores, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f.replace('-', '\n') for f in fields], fontsize=9, rotation=0)
    axes[0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[0].set_title('R² Score by Variable', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(1.1, max(r2_scores) * 1.1))
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.02,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_axisbelow(True)
    
    # RMSE
    bars = axes[1].bar(x_pos, rmse_vals, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f.replace('-', '\n') for f in fields], fontsize=9, rotation=0)
    axes[1].set_ylabel('RMSE', fontsize=11, fontweight='bold')
    axes[1].set_title('RMSE by Variable', fontsize=12, fontweight='bold')
    # Use log scale only if values span multiple orders of magnitude
    if max(rmse_vals) / (min(rmse_vals) + 1e-10) > 100:
        axes[1].set_yscale('log')
    for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height * 1.05,
                     f'{val:.2e}', ha='center', va='bottom', fontsize=8, rotation=0)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_axisbelow(True)
    
    # MAPE
    mape_vals_plot = [0 if np.isnan(v) else v for v in mape_vals]
    bars = axes[2].bar(x_pos, mape_vals_plot, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f.replace('-', '\n') for f in fields], fontsize=9, rotation=0)
    axes[2].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('MAPE by Variable', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, mape_vals)):
        if not np.isnan(val) and val > 0:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2, height + max(mape_vals_plot) * 0.02,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_comparison.png")


def plot_error_distribution_analysis(predicted: np.ndarray, ground_truth: np.ndarray,
                                      save_dir: Path):
    """Create comprehensive error distribution analysis graphs."""
    
    # Downsample if dataset is too large (to avoid memory issues)
    MAX_POINTS = 500000
    n_points = len(predicted)
    
    if n_points > MAX_POINTS:
        print(f"  Downsampling from {n_points:,} to {MAX_POINTS:,} points for visualization...")
        np.random.seed(42)
        idx = np.random.choice(n_points, MAX_POINTS, replace=False)
        predicted = predicted[idx]
        ground_truth = ground_truth[idx]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Collect errors for all fields
    errors = {}
    relative_errors = {}
    
    for i, field_name in enumerate(OUTPUT_COLUMNS):
        pred = predicted[:, i]
        truth = ground_truth[:, i]
        
        # Filter out NaN/Inf values first
        valid_mask = np.isfinite(pred) & np.isfinite(truth)
        pred_valid = pred[valid_mask]
        truth_valid = truth[valid_mask]
        
        if len(pred_valid) == 0:
            print(f"  WARNING: No valid data for {field_name}")
            errors[field_name] = np.array([0.0])
            relative_errors[field_name] = np.array([0.0])
            continue
        
        error = pred_valid - truth_valid
        
        # Relative error (percentage) - vectorized operation
        nonzero_mask = np.abs(truth_valid) > 1e-8
        rel_err = np.zeros(len(error))
        if np.any(nonzero_mask):
            rel_err[nonzero_mask] = (error[nonzero_mask] / truth_valid[nonzero_mask]) * 100
        
        errors[field_name] = error
        relative_errors[field_name] = rel_err
    
    # 1. Box plot of absolute errors
    ax1 = fig.add_subplot(gs[0, 0])
    error_data = [errors[f] for f in OUTPUT_COLUMNS]
    bp = ax1.boxplot(error_data, labels=[f.replace('-', '\n') for f in OUTPUT_COLUMNS],
                     patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.8, len(OUTPUT_COLUMNS)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Absolute Error', fontweight='bold')
    ax1.set_title('Error Distribution (Box Plot)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.tick_params(axis='x', labelsize=8)
    
    # 2. Violin plot of relative errors (clip to reasonable range)
    ax2 = fig.add_subplot(gs[0, 1])
    rel_error_data = []
    for f in OUTPUT_COLUMNS:
        rel_err = relative_errors[f]
        # Clip to [-100, 100] range for visualization
        clipped = rel_err[(rel_err > -100) & (rel_err < 100)]
        rel_error_data.append(clipped if len(clipped) > 0 else np.array([0.0]))
    
    parts = ax2.violinplot(rel_error_data, positions=range(len(OUTPUT_COLUMNS)),
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plt.cm.viridis(0.2 + 0.6 * i / len(OUTPUT_COLUMNS)))
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(OUTPUT_COLUMNS)))
    ax2.set_xticklabels([f.replace('-', '\n') for f in OUTPUT_COLUMNS], fontsize=8)
    ax2.set_ylabel('Relative Error (%)', fontweight='bold')
    ax2.set_title('Relative Error Distribution (Violin Plot)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    # 3. Error percentiles
    ax3 = fig.add_subplot(gs[0, 2])
    percentiles = [50, 75, 90, 95, 99]
    percentile_data = []
    for field in OUTPUT_COLUMNS:
        abs_errors = np.abs(errors[field])
        pcts = [np.percentile(abs_errors, p) for p in percentiles]
        percentile_data.append(pcts)
    
    x = np.arange(len(OUTPUT_COLUMNS))
    width = 0.15
    for i, p in enumerate(percentiles):
        values = [percentile_data[j][i] for j in range(len(OUTPUT_COLUMNS))]
        ax3.bar(x + i * width, values, width, label=f'{p}th %ile',
                alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels([f.replace('-', '\n') for f in OUTPUT_COLUMNS], fontsize=8)
    ax3.set_ylabel('Absolute Error', fontweight='bold')
    ax3.set_title('Error Percentiles', fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')
    if max([max(pd) for pd in percentile_data]) / (min([min(pd) for pd in percentile_data]) + 1e-10) > 100:
        ax3.set_yscale('log')
    
    # 4-8. Individual field error histograms (first 5 fields)
    for idx in range(min(5, len(OUTPUT_COLUMNS))):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        field_name = OUTPUT_COLUMNS[idx]
        error = errors[field_name]
        
        # Create histogram
        n, bins, patches = ax.hist(error, bins=80, alpha=0.7, color='steelblue',
                                   edgecolor='black', linewidth=0.5)
        
        # Color bars based on distance from zero
        cm = plt.cm.RdYlGn_r
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col_vals = np.abs(bin_centers)
        col_vals = col_vals / col_vals.max()
        for c, p in zip(col_vals, patches):
            plt.setp(p, 'facecolor', cm(1 - c))
        
        # Add statistics
        mean_err = np.mean(error)
        std_err = np.std(error)
        median_err = np.median(error)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(x=mean_err, color='blue', linestyle='-', linewidth=1.5, label=f'Mean: {mean_err:.2e}')
        ax.axvline(x=median_err, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median_err:.2e}')
        
        ax.set_xlabel('Prediction Error', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{field_name}\nσ={std_err:.2e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Error Distribution Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(save_dir / "error_distribution_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_distribution_analysis.png")


def plot_field_contours(coords: np.ndarray, predicted: np.ndarray, 
                        ground_truth: np.ndarray, sample_id: int,
                        save_dir: Path, max_points: int = 100000):
    """Generate side-by-side contour plots with error maps."""
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Downsample if needed
    n_points = len(x)
    if n_points > max_points:
        np.random.seed(42)
        idx = np.random.choice(n_points, max_points, replace=False)
        x = x[idx]
        y = y[idx]
        predicted = predicted[idx]
        ground_truth = ground_truth[idx]
    
    # Create triangulation
    try:
        tri = Triangulation(x, y)
    except:
        print(f"  Warning: Could not create triangulation for sample {sample_id}")
        return
    
    # Key fields to visualize: pressure, mach-number, density
    key_fields = ['pressure', 'mach-number', 'density']
    
    for field_name in key_fields:
        if field_name not in OUTPUT_COLUMNS:
            continue
        field_idx = OUTPUT_COLUMNS.index(field_name)
        
        pred = predicted[:, field_idx]
        truth = ground_truth[:, field_idx]
        error = pred - truth
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # CFD Ground Truth
        vmin, vmax = truth.min(), truth.max()
        im0 = axes[0].tricontourf(tri, truth, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'CFD Ground Truth\n{field_name}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0])
        
        # DeepONet Prediction
        im1 = axes[1].tricontourf(tri, pred, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'DeepONet Prediction\n{field_name}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1])
        
        # Error Map
        err_max = np.max(np.abs(error))
        im2 = axes[2].tricontourf(tri, error, levels=50, cmap='RdBu_r', 
                                   vmin=-err_max, vmax=err_max)
        axes[2].set_title(f'Absolute Error\n{field_name} (MAE={np.mean(np.abs(error)):.2e})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        filename = f"sample_{sample_id}_{field_name.replace('-', '_')}_comparison.png"
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")


def plot_training_history(save_dir: Path):
    """Plot training and validation loss curves."""
    history_path = MODEL_SAVE_DIR / 'training_history.json'
    if not history_path.exists():
        print("  Training history not found, skipping...")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].semilogy(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].semilogy(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].semilogy(epochs, history['learning_rate'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_history.png")


def generate_results_table(all_metrics: Dict, inference_time: float, 
                          save_dir: Path) -> pd.DataFrame:
    """Generate results table as CSV."""
    rows = []
    for field_name, metrics in all_metrics.items():
        rows.append({
            'Variable': field_name,
            'R² Score': f"{metrics['r2']:.4f}",
            'RMSE': f"{metrics['rmse']:.4e}",
            'MAE': f"{metrics['mae']:.4e}",
            'MAPE (%)': f"{metrics['mape']:.2f}" if not np.isnan(metrics['mape']) else 'N/A',
            'Max Error': f"{metrics['max_error']:.4e}",
            'Rel L2 (%)': f"{metrics['rel_l2']*100:.2f}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "results_table.csv", index=False)
    print(f"  Saved: results_table.csv")
    return df


def generate_markdown_report(all_metrics: Dict, inference_time: float,
                             num_samples: int, num_points: int,
                             save_dir: Path):
    """Generate formatted markdown report."""
    report = f"""# DeepONet Model Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Test Samples | {num_samples} |
| Total Points | {num_points:,} |
| Avg Inference Time | {inference_time:.2f} ms |
| Speedup vs CFD | ~{1000/inference_time:.0f}x faster (estimated) |

## Quantitative Metrics

| Variable | R² Score | RMSE | MAPE (%) |
|----------|----------|------|----------|
"""
    
    for field_name, metrics in all_metrics.items():
        mape_str = f"{metrics['mape']:.2f}" if not np.isnan(metrics['mape']) else 'N/A'
        report += f"| {field_name} | {metrics['r2']:.4f} | {metrics['rmse']:.4e} | {mape_str} |\n"
    
    report += f"""
## Generated Visualizations

### Statistical Graphs
- `metrics/scatter_all_fields.png` - Predicted vs Ground Truth scatter plots
- `metrics/error_histograms.png` - Error distribution histograms
- `metrics/metrics_comparison.png` - Variable-wise metric comparison

### Flow Field Comparisons
Contour plots showing CFD (left) vs DeepONet (right) with error maps:
- `contours/sample_*_pressure_comparison.png`
- `contours/sample_*_mach_number_comparison.png`  
- `contours/sample_*_density_comparison.png`

### Training History
- `training_history.png` - Loss curves and learning rate schedule

## Notes
- R² values close to 1.0 indicate excellent prediction accuracy
- MAPE shows percentage deviation from ground truth
- Inference time measured on {DEVICE}
"""
    
    with open(save_dir / "evaluation_report.md", 'w') as f:
        f.write(report)
    print(f"  Saved: evaluation_report.md")


def main():
    print("=" * 60)
    print("DeepONet Model Comprehensive Evaluation")
    print("=" * 60)
    
    # Load predictor
    print("\n[1/6] Loading model...")
    predictor = CFDPredictor()
    print(f"  Model loaded from: {MODEL_SAVE_DIR / 'best_model.pt'}")
    print(f"  Device: {DEVICE}")
    
    # Load data
    print("\n[2/6] Loading CFD data...")
    data_loader = CFDDataLoader()
    data_loader.compute_normalization_stats()
    all_sample_ids = data_loader.get_available_sample_ids()
    
    # Limit to first 10 samples to avoid memory issues
    MAX_SAMPLES = 10
    sample_ids = all_sample_ids[:MAX_SAMPLES]
    
    print(f"  Available samples: {len(all_sample_ids)}")
    print(f"  Using {len(sample_ids)} samples for evaluation: {sample_ids}")
    
    # Collect predictions for all samples
    print("\n[3/6] Running predictions on all samples...")
    all_predictions = []
    all_ground_truth = []
    all_coords = []
    
    for sample_id in sample_ids:
        try:
            params = data_loader.get_input_params(sample_id)
            coords, ground_truth = data_loader.get_cfd_output(sample_id)
            predictions = predictor.predict(params, coords)
            
            # Check for NaN/Inf in predictions
            if not np.all(np.isfinite(predictions)):
                nan_count = np.sum(~np.isfinite(predictions))
                print(f"  WARNING: Sample {sample_id} has {nan_count} NaN/Inf predictions")
            
            all_predictions.append(predictions)
            all_ground_truth.append(ground_truth)
            all_coords.append(coords)
            print(f"  Sample {sample_id}: {len(coords):,} points")
        except Exception as e:
            print(f"  ERROR processing sample {sample_id}: {e}")
            continue
    
    # Check if we have any data
    if not all_predictions:
        print("\nERROR: No samples found. Please ensure CFD output files exist.")
        print("Expected location: cfd_outputs/d*.csv")
        return
    
    # Concatenate all data
    all_pred = np.vstack(all_predictions)
    all_truth = np.vstack(all_ground_truth)
    total_points = len(all_pred)
    print(f"  Total: {total_points:,} points across {len(sample_ids)} samples")
    
    # Compute metrics
    print("\n[4/6] Computing metrics...")
    metrics = compute_all_metrics(all_pred, all_truth)
    
    # Measure inference time
    sample_params = data_loader.get_input_params(sample_ids[0])
    sample_coords, _ = data_loader.get_cfd_output(sample_ids[0])
    inference_time = measure_inference_time(predictor, sample_params, sample_coords)
    print(f"  Average inference time: {inference_time:.2f} ms")
    
    # Print metrics table
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(f"\n{'Variable':<25} {'R² Score':<12} {'RMSE':<14} {'MAPE (%)':<12}")
    print("-" * 80)
    for field, m in metrics.items():
        mape_str = f"{m['mape']:.2f}" if not np.isnan(m['mape']) else 'N/A'
        print(f"{field:<25} {m['r2']:<12.4f} {m['rmse']:<14.4e} {mape_str:<12}")
    print("=" * 80)
    
    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    
    print("\n  Creating scatter plots...")
    plot_scatter_comparison(all_pred, all_truth, OUTPUT_DIR / "metrics")
    
    print("\n  Creating error histograms...")
    plot_error_histograms(all_pred, all_truth, OUTPUT_DIR / "metrics")
    
    print("\n  Creating metrics comparison chart...")
    plot_metrics_comparison(metrics, OUTPUT_DIR / "metrics")
    
    print("\n  Creating comprehensive error distribution analysis...")
    plot_error_distribution_analysis(all_pred, all_truth, OUTPUT_DIR / "metrics")
    
    print("\n  Creating training history plot...")
    plot_training_history(OUTPUT_DIR)
    
    # Generate contour plots for representative samples (first 3)
    print("\n  Creating flow field contour plots...")
    for i, sample_id in enumerate(sample_ids[:3]):
        print(f"\n  Processing sample {sample_id}...")
        plot_field_contours(
            all_coords[i], all_predictions[i], all_ground_truth[i],
            sample_id, OUTPUT_DIR / "contours"
        )
    
    # Generate reports
    print("\n[6/6] Generating reports...")
    generate_results_table(metrics, inference_time, OUTPUT_DIR)
    generate_markdown_report(metrics, inference_time, len(sample_ids), 
                            total_points, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nKey files:")
    print(f"  - {OUTPUT_DIR / 'results_table.csv'}")
    print(f"  - {OUTPUT_DIR / 'evaluation_report.md'}")
    print(f"  - {OUTPUT_DIR / 'metrics' / 'scatter_all_fields.png'}")
    print(f"  - {OUTPUT_DIR / 'contours'} (flow field comparisons)")


if __name__ == "__main__":
    main()
