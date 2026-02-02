"""
Data loading utilities for CFD Surrogate Model
Handles loading input parameters and CFD output files
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from config import (
    INPUT_PARAMS_FILE, CFD_OUTPUTS_DIR, SIMULATION_MAPPING_FILE,
    INPUT_COLUMNS, OUTPUT_COLUMNS, COORD_COLUMNS
)


class CFDDataLoader:
    """
    Loads and preprocesses CFD simulation data for surrogate model training.
    """
    
    def __init__(
        self, 
        input_params_file: Path = INPUT_PARAMS_FILE,
        cfd_outputs_dir: Path = CFD_OUTPUTS_DIR,
        mapping_file: Path = SIMULATION_MAPPING_FILE
    ):
        self.input_params_file = Path(input_params_file)
        self.cfd_outputs_dir = Path(cfd_outputs_dir)
        self.mapping_file = Path(mapping_file)
        
        # Normalization parameters (computed from data)
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.output_mean: Optional[np.ndarray] = None
        self.output_std: Optional[np.ndarray] = None
        self.coord_mean: Optional[np.ndarray] = None
        self.coord_std: Optional[np.ndarray] = None
        
        # Load data
        self.input_params_df = self._load_input_params()
        self.simulation_mapping = self._load_mapping()
        self.cfd_data = self._load_cfd_outputs()
        
    def _load_input_params(self) -> pd.DataFrame:
        """Load input parameters CSV file."""
        if not self.input_params_file.exists():
            raise FileNotFoundError(f"Input params file not found: {self.input_params_file}")
        
        df = pd.read_csv(self.input_params_file)
        print(f"Loaded {len(df)} input parameter sets from {self.input_params_file.name}")
        return df
    
    def _load_mapping(self) -> pd.DataFrame:
        """
        Auto-detect CFD output files and create mapping automatically.
        Files matching pattern d{N}.csv are linked to Sample_ID N.
        """
        # Auto-detect CFD files matching pattern d{N}.csv
        cfd_files = list(self.cfd_outputs_dir.glob("d*.csv"))
        
        if not cfd_files:
            warnings.warn(f"No CFD output files found in {self.cfd_outputs_dir}")
            return pd.DataFrame(columns=['filename', 'sample_id'])
        
        # Parse file names to extract sample IDs
        mapping_data = []
        import re
        for filepath in cfd_files:
            filename = filepath.name
            # Extract number from d{N}.csv pattern
            match = re.match(r'd(\d+)\.csv', filename)
            if match:
                sample_id = int(match.group(1))
                mapping_data.append({'filename': filename, 'sample_id': sample_id})
        
        if not mapping_data:
            warnings.warn("No files matching d{N}.csv pattern found")
            return pd.DataFrame(columns=['filename', 'sample_id'])
        
        mapping = pd.DataFrame(mapping_data)
        mapping = mapping.sort_values('sample_id').reset_index(drop=True)
        
        print(f"Auto-detected {len(mapping)} CFD output files:")
        for _, row in mapping.iterrows():
            print(f"  {row['filename']} -> Sample_ID {row['sample_id']}")
        
        return mapping
    
    def _load_cfd_outputs(self) -> Dict[int, pd.DataFrame]:
        """Load all available CFD output files based on mapping."""
        cfd_data = {}
        
        if len(self.simulation_mapping) == 0:
            # Try to load any CSV files in the directory
            csv_files = list(self.cfd_outputs_dir.glob("*.csv"))
            if csv_files:
                print(f"Found {len(csv_files)} CFD files but no mapping. "
                      f"Please create {self.mapping_file}")
            return cfd_data
        
        for _, row in self.simulation_mapping.iterrows():
            filename = row['filename']
            sample_id = int(row['sample_id'])
            filepath = self.cfd_outputs_dir / filename
            
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    # Clean column names (remove leading/trailing spaces)
                    df.columns = df.columns.str.strip()
                    cfd_data[sample_id] = df
                    print(f"Loaded {filename} -> Sample_ID {sample_id} ({len(df)} nodes)")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: CFD file not found: {filepath}")
        
        return cfd_data
    
    def get_input_params(self, sample_id: int) -> np.ndarray:
        """Get input parameters for a specific sample."""
        row = self.input_params_df[self.input_params_df['Sample_ID'] == sample_id]
        if len(row) == 0:
            raise ValueError(f"Sample_ID {sample_id} not found in input params")
        return row[INPUT_COLUMNS].values.flatten().astype(np.float32)
    
    def get_cfd_output(self, sample_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get CFD output for a specific sample.
        
        Returns:
            coords: (N, 2) array of (x, y) coordinates
            fields: (N, 5) array of output fields
        """
        if sample_id not in self.cfd_data:
            raise ValueError(f"CFD data for Sample_ID {sample_id} not loaded")
        
        df = self.cfd_data[sample_id]
        coords = df[COORD_COLUMNS].values.astype(np.float32)
        fields = df[OUTPUT_COLUMNS].values.astype(np.float32)
        return coords, fields
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization from available data."""
        # Input parameters - use all 150 samples
        all_inputs = self.input_params_df[INPUT_COLUMNS].values.astype(np.float32)
        self.input_mean = np.mean(all_inputs, axis=0)
        self.input_std = np.std(all_inputs, axis=0) + 1e-8
        
        # Coordinates and outputs - use available CFD data
        all_coords = []
        all_outputs = []
        
        for sample_id in self.cfd_data.keys():
            coords, fields = self.get_cfd_output(sample_id)
            all_coords.append(coords)
            all_outputs.append(fields)
        
        if len(all_coords) > 0:
            all_coords = np.vstack(all_coords)
            all_outputs = np.vstack(all_outputs)
            
            self.coord_mean = np.mean(all_coords, axis=0)
            self.coord_std = np.std(all_coords, axis=0) + 1e-8
            
            self.output_mean = np.mean(all_outputs, axis=0)
            self.output_std = np.std(all_outputs, axis=0) + 1e-8
        else:
            print("Warning: No CFD data available for normalization stats")
            # Use placeholder values
            self.coord_mean = np.zeros(2)
            self.coord_std = np.ones(2)
            self.output_mean = np.zeros(len(OUTPUT_COLUMNS))
            self.output_std = np.ones(len(OUTPUT_COLUMNS))
        
        print(f"Normalization stats computed from {len(self.cfd_data)} simulations")
    
    def normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize input parameters."""
        if self.input_mean is None:
            self.compute_normalization_stats()
        return (inputs - self.input_mean) / self.input_std
    
    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates."""
        if self.coord_mean is None:
            self.compute_normalization_stats()
        return (coords - self.coord_mean) / self.coord_std
    
    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize output fields."""
        if self.output_mean is None:
            self.compute_normalization_stats()
        return (outputs - self.output_mean) / self.output_std
    
    def denormalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Denormalize output fields back to physical units."""
        if self.output_mean is None:
            raise ValueError("Normalization stats not computed")
        return outputs * self.output_std + self.output_mean
    
    def get_available_sample_ids(self) -> List[int]:
        """Get list of sample IDs with available CFD data."""
        return list(self.cfd_data.keys())
    
    def save_normalization_stats(self, filepath: Path):
        """Save normalization statistics to file for inference."""
        np.savez(
            filepath,
            input_mean=self.input_mean,
            input_std=self.input_std,
            coord_mean=self.coord_mean,
            coord_std=self.coord_std,
            output_mean=self.output_mean,
            output_std=self.output_std
        )
        print(f"Normalization stats saved to {filepath}")
    
    def load_normalization_stats(self, filepath: Path):
        """Load normalization statistics from file."""
        data = np.load(filepath)
        self.input_mean = data['input_mean']
        self.input_std = data['input_std']
        self.coord_mean = data['coord_mean']
        self.coord_std = data['coord_std']
        self.output_mean = data['output_mean']
        self.output_std = data['output_std']
        print(f"Normalization stats loaded from {filepath}")


def add_simulation_to_mapping(filename: str, sample_id: int, mapping_file: Path = SIMULATION_MAPPING_FILE):
    """
    Utility function to add a new simulation to the mapping file.
    
    Usage:
        add_simulation_to_mapping("SYS-1-00316.csv", 2)
    """
    if mapping_file.exists():
        df = pd.read_csv(mapping_file)
    else:
        df = pd.DataFrame(columns=['filename', 'sample_id'])
    
    # Check for duplicates
    if sample_id in df['sample_id'].values:
        print(f"Warning: Sample_ID {sample_id} already exists, updating...")
        df = df[df['sample_id'] != sample_id]
    
    new_row = pd.DataFrame({'filename': [filename], 'sample_id': [sample_id]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(mapping_file, index=False)
    print(f"Added {filename} -> Sample_ID {sample_id} to mapping")


if __name__ == "__main__":
    # Test data loading
    loader = CFDDataLoader()
    loader.compute_normalization_stats()
    
    print("\n=== Data Summary ===")
    print(f"Total input parameter sets: {len(loader.input_params_df)}")
    print(f"Available CFD simulations: {loader.get_available_sample_ids()}")
    
    for sample_id in loader.get_available_sample_ids():
        params = loader.get_input_params(sample_id)
        coords, fields = loader.get_cfd_output(sample_id)
        print(f"\nSample {sample_id}:")
        print(f"  Input params shape: {params.shape}")
        print(f"  Coordinates shape: {coords.shape}")
        print(f"  Fields shape: {fields.shape}")
