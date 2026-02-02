"""
PyTorch Dataset for CFD Surrogate Model Training
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import random

from data_loader import CFDDataLoader
from config import POINTS_PER_SIMULATION, DEVICE


class CFDPointDataset(Dataset):
    """
    Dataset that samples (parameter, coordinate, output) triplets from CFD simulations.
    
    Each sample consists of:
    - Input parameters (18-dim): The design/operating conditions
    - Coordinates (2-dim): The (x, y) location in the mesh
    - Output fields (5-dim): The CFD solution values at that location
    """
    
    def __init__(
        self, 
        data_loader: CFDDataLoader,
        sample_ids: List[int],
        points_per_simulation: int = POINTS_PER_SIMULATION,
        normalize: bool = True,
        training: bool = True
    ):
        """
        Args:
            data_loader: CFDDataLoader instance with loaded data
            sample_ids: List of Sample_IDs to include in this dataset
            points_per_simulation: Number of points to sample from each simulation
            normalize: Whether to normalize data
            training: If True, randomly sample points. If False, use all points.
        """
        self.data_loader = data_loader
        self.sample_ids = sample_ids
        self.points_per_simulation = points_per_simulation
        self.normalize = normalize
        self.training = training
        
        # Ensure normalization stats are computed
        if normalize and self.data_loader.input_mean is None:
            self.data_loader.compute_normalization_stats()
        
        # Pre-load and prepare all data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare all data samples."""
        self.all_params = []    # (num_samples, 17)
        self.all_coords = []    # (num_samples, 2)
        self.all_outputs = []   # (num_samples, 5)
        
        for sample_id in self.sample_ids:
            try:
                # Get input parameters
                params = self.data_loader.get_input_params(sample_id)
                
                # Get CFD output
                coords, fields = self.data_loader.get_cfd_output(sample_id)
                
                # Sample points if training
                if self.training and len(coords) > self.points_per_simulation:
                    indices = np.random.choice(
                        len(coords), 
                        self.points_per_simulation, 
                        replace=False
                    )
                    coords = coords[indices]
                    fields = fields[indices]
                
                # Normalize if needed
                if self.normalize:
                    params = self.data_loader.normalize_inputs(params)
                    coords = self.data_loader.normalize_coords(coords)
                    fields = self.data_loader.normalize_outputs(fields)
                
                # Store data - parameters are repeated for each point
                num_points = len(coords)
                params_repeated = np.tile(params, (num_points, 1))
                
                self.all_params.append(params_repeated)
                self.all_coords.append(coords)
                self.all_outputs.append(fields)
                
            except Exception as e:
                print(f"Warning: Error processing sample {sample_id}: {e}")
        
        if len(self.all_params) > 0:
            self.all_params = np.vstack(self.all_params).astype(np.float32)
            self.all_coords = np.vstack(self.all_coords).astype(np.float32)
            self.all_outputs = np.vstack(self.all_outputs).astype(np.float32)
        else:
            # Create empty arrays with correct shapes - use config values
            from config import NUM_INPUT_PARAMS, NUM_OUTPUT_FIELDS
            self.all_params = np.zeros((0, NUM_INPUT_PARAMS), dtype=np.float32)
            self.all_coords = np.zeros((0, 2), dtype=np.float32)
            self.all_outputs = np.zeros((0, NUM_OUTPUT_FIELDS), dtype=np.float32)
        
        print(f"Dataset: {len(self)} points from {len(self.sample_ids)} simulations")
    
    def __len__(self) -> int:
        return len(self.all_params)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            params: (17,) tensor of input parameters
            coords: (2,) tensor of (x, y) coordinates
            outputs: (5,) tensor of output fields
        """
        params = torch.from_numpy(self.all_params[idx])
        coords = torch.from_numpy(self.all_coords[idx])
        outputs = torch.from_numpy(self.all_outputs[idx])
        return params, coords, outputs
    
    def resample(self):
        """Re-sample points from simulations (for training variety between epochs)."""
        if self.training:
            self._prepare_data()


class CFDFullFieldDataset(Dataset):
    """
    Dataset that returns full simulation fields (for validation/visualization).
    Each sample is one complete simulation.
    """
    
    def __init__(
        self, 
        data_loader: CFDDataLoader,
        sample_ids: List[int],
        normalize: bool = True
    ):
        self.data_loader = data_loader
        self.sample_ids = sample_ids
        self.normalize = normalize
        
        if normalize and self.data_loader.input_mean is None:
            self.data_loader.compute_normalization_stats()
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            params: (17,) tensor of input parameters
            coords: (N, 2) tensor of all mesh coordinates
            outputs: (N, 5) tensor of all output fields
        """
        sample_id = self.sample_ids[idx]
        
        params = self.data_loader.get_input_params(sample_id)
        coords, fields = self.data_loader.get_cfd_output(sample_id)
        
        if self.normalize:
            params = self.data_loader.normalize_inputs(params)
            coords = self.data_loader.normalize_coords(coords)
            fields = self.data_loader.normalize_outputs(fields)
        
        return (
            torch.from_numpy(params),
            torch.from_numpy(coords),
            torch.from_numpy(fields)
        )


def create_dataloaders(
    data_loader: CFDDataLoader,
    train_ratio: float = 0.8,
    batch_size: int = 4096,
    points_per_simulation: int = POINTS_PER_SIMULATION,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_loader: CFDDataLoader with loaded data
        train_ratio: Fraction of simulations for training
        batch_size: Batch size for training
        points_per_simulation: Points to sample per simulation
        num_workers: Number of data loading workers
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader  
        train_ids: Sample IDs used for training
        val_ids: Sample IDs used for validation
    """
    available_ids = data_loader.get_available_sample_ids()
    
    if len(available_ids) == 0:
        raise ValueError("No CFD simulations available! Add data to cfd_outputs/ and update simulation_mapping.csv")
    
    # Shuffle and split
    random.shuffle(available_ids)
    split_idx = max(1, int(len(available_ids) * train_ratio))
    
    train_ids = available_ids[:split_idx]
    val_ids = available_ids[split_idx:] if split_idx < len(available_ids) else available_ids[:1]
    
    print(f"Train simulations: {train_ids}")
    print(f"Validation simulations: {val_ids}")
    
    # Create datasets
    train_dataset = CFDPointDataset(
        data_loader, 
        train_ids,
        points_per_simulation=points_per_simulation,
        normalize=True,
        training=True
    )
    
    val_dataset = CFDPointDataset(
        data_loader,
        val_ids,
        points_per_simulation=points_per_simulation * 2,  # More points for validation
        normalize=True,
        training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, train_ids, val_ids


if __name__ == "__main__":
    # Test dataset creation
    from data_loader import CFDDataLoader
    
    loader = CFDDataLoader()
    loader.compute_normalization_stats()
    
    sample_ids = loader.get_available_sample_ids()
    
    if sample_ids:
        dataset = CFDPointDataset(loader, sample_ids)
        print(f"\nDataset size: {len(dataset)}")
        
        # Get a sample
        params, coords, outputs = dataset[0]
        print(f"\nSample shapes:")
        print(f"  Params: {params.shape}")
        print(f"  Coords: {coords.shape}")
        print(f"  Outputs: {outputs.shape}")
        
        # Test dataloader
        train_loader, val_loader, train_ids, val_ids = create_dataloaders(loader)
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    else:
        print("No simulations available for testing")
