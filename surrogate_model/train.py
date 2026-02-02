"""
Training script for CFD Surrogate Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import sys
from typing import Tuple, Optional

from config import (
    DEVICE, MODEL_SAVE_DIR, 
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, MIN_DELTA, LOG_INTERVAL, SAVE_INTERVAL,
    BATCH_SIZE, POINTS_PER_SIMULATION, TRAIN_RATIO
)
from data_loader import CFDDataLoader
from dataset import create_dataloaders
from model import create_model, count_parameters


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for params, coords, targets in train_loader:
        params = params.to(device)
        coords = coords.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(params, coords)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, dict]:
    """Validate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for params, coords, targets in val_loader:
            params = params.to(device)
            coords = coords.to(device)
            targets = targets.to(device)
            
            outputs = model(params, coords)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute per-field metrics
    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Mean Absolute Error per field
        mae_per_field = torch.mean(torch.abs(all_outputs - all_targets), dim=0).numpy()
        
        # Relative L2 error per field
        rel_l2 = torch.sqrt(torch.mean((all_outputs - all_targets)**2, dim=0))
        rel_l2 = rel_l2 / (torch.sqrt(torch.mean(all_targets**2, dim=0)) + 1e-8)
        rel_l2 = rel_l2.numpy()
        
        metrics = {
            'mae_per_field': mae_per_field.tolist(),
            'rel_l2_per_field': rel_l2.tolist()
        }
    else:
        metrics = {}
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    filepath: Path,
    data_loader: CFDDataLoader
):
    """Save model checkpoint."""
    # Save normalization stats alongside model
    norm_stats_path = filepath.parent / 'normalization_stats.npz'
    data_loader.save_normalization_stats(norm_stats_path)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': model.config if hasattr(model, 'config') else None
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> int:
    """Load model checkpoint. Returns epoch number."""
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def train(
    use_improved_model: bool = True,
    resume_from: Optional[Path] = None,
    input_params_file: Optional[Path] = None,
    cfd_outputs_dir: Optional[Path] = None,
    mapping_file: Optional[Path] = None
):
    """Main training function."""
    print("=" * 60)
    print("CFD Surrogate Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    # Use provided paths or defaults from config (which are imported at module level)
    # import defaults inside function to ensure we use what's available
    from config import INPUT_PARAMS_FILE, CFD_OUTPUTS_DIR, SIMULATION_MAPPING_FILE
    
    data_loader = CFDDataLoader(
        input_params_file=input_params_file or INPUT_PARAMS_FILE,
        cfd_outputs_dir=cfd_outputs_dir or CFD_OUTPUTS_DIR,
        mapping_file=mapping_file or SIMULATION_MAPPING_FILE
    )
    data_loader.compute_normalization_stats()
    
    available_ids = data_loader.get_available_sample_ids()
    if not available_ids:
        print("\nERROR: No CFD simulations found!")
        print("Please add CFD output files to 'cfd_outputs/' folder")
        print("and update 'simulation_mapping.csv' with the mapping.")
        sys.exit(1)
    
    print(f"\nAvailable simulations: {available_ids}")
    
    # Create dataloaders
    print("\n[2/4] Creating datasets...")
    train_loader, val_loader, train_ids, val_ids = create_dataloaders(
        data_loader,
        train_ratio=TRAIN_RATIO,
        batch_size=BATCH_SIZE,
        points_per_simulation=POINTS_PER_SIMULATION
    )
    
    # Create model
    print("\n[3/4] Building model...")
    model = create_model(improved=use_improved_model)
    model = model.to(DEVICE)
    print(f"Device: {DEVICE}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from is not None and resume_from.exists():
        start_epoch = load_checkpoint(resume_from, model, optimizer)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_model_path = MODEL_SAVE_DIR / 'best_model.pt'
    
    # Training loop
    print("\n[4/4] Training...")
    print("-" * 60)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # Log progress
        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, 
                best_model_path, data_loader
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = MODEL_SAVE_DIR / f'checkpoint_epoch_{epoch+1}.pt'
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_path, data_loader
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    final_model_path = MODEL_SAVE_DIR / 'final_model.pt'
    save_checkpoint(
        model, optimizer, epoch, train_loss, val_loss,
        final_model_path, data_loader
    )
    
    # Save training history
    history_path = MODEL_SAVE_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CFD Surrogate Model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple DeepONet (without Fourier features)')
    
    parser.add_argument('--input_params', type=str, default=None,
                        help='Path to input parameters CSV file')
    parser.add_argument('--cfd_outputs', type=str, default=None,
                        help='Path to directory containing CFD output CSV files')
    parser.add_argument('--mapping', type=str, default=None,
                        help='Path to simulation mapping CSV file')
    
    args = parser.parse_args()
    
    resume_path = Path(args.resume) if args.resume else None
    input_params = Path(args.input_params) if args.input_params else None
    cfd_outputs = Path(args.cfd_outputs) if args.cfd_outputs else None
    mapping = Path(args.mapping) if args.mapping else None
    
    train(
        use_improved_model=not args.simple,
        resume_from=resume_path,
        input_params_file=input_params,
        cfd_outputs_dir=cfd_outputs,
        mapping_file=mapping
    )
