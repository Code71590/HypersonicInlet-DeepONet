"""
Inference script for trained CFD Surrogate Model
Predict CFD fields for new parameter sets
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

from config import (
    DEVICE, MODEL_SAVE_DIR, INPUT_COLUMNS, OUTPUT_COLUMNS, COORD_COLUMNS
)
from data_loader import CFDDataLoader
from model import DeepONet, ImprovedDeepONet


class CFDPredictor:
    """
    Inference wrapper for trained surrogate model.
    """
    
    def __init__(
        self,
        model_path: Path = MODEL_SAVE_DIR / 'best_model.pt',
        norm_stats_path: Path = MODEL_SAVE_DIR / 'normalization_stats.npz',
        device: torch.device = DEVICE
    ):
        self.device = device
        
        # Load normalization stats
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
        
        self.norm_stats = np.load(norm_stats_path)
        self.input_mean = self.norm_stats['input_mean']
        self.input_std = self.norm_stats['input_std']
        self.coord_mean = self.norm_stats['coord_mean']
        self.coord_std = self.norm_stats['coord_std']
        self.output_mean = self.norm_stats['output_mean']
        self.output_std = self.norm_stats['output_std']
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Device: {device}")
    
    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type from config
        config = checkpoint.get('model_config', {})
        
        if config and 'num_fourier_features' in config:
            model = ImprovedDeepONet(**config)
        elif config:
            model = DeepONet(**config)
        else:
            # Default to ImprovedDeepONet
            model = ImprovedDeepONet()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize input parameters."""
        return (inputs - self.input_mean) / self.input_std
    
    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates."""
        return (coords - self.coord_mean) / self.coord_std
    
    def denormalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Denormalize predicted outputs to physical units."""
        return outputs * self.output_std + self.output_mean
    
    def predict(
        self,
        params: np.ndarray,
        coords: np.ndarray,
        batch_size: int = 10000,
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Predict CFD fields for given parameters at specified coordinates.
        
        Args:
            params: (17,) input parameters
            coords: (N, 2) mesh coordinates
            batch_size: Batch size for inference
            denormalize: Whether to denormalize outputs to physical units
            
        Returns:
            outputs: (N, 5) predicted field values
        """
        # Normalize inputs
        params_norm = self.normalize_inputs(params)
        coords_norm = self.normalize_coords(coords)
        
        # Convert to tensors
        params_tensor = torch.from_numpy(params_norm).float().to(self.device)
        coords_tensor = torch.from_numpy(coords_norm).float().to(self.device)
        
        # Predict using inference mode (faster than no_grad)
        with torch.inference_mode():
            outputs = self.model.predict_field(
                params_tensor, coords_tensor, batch_size=batch_size
            )
            outputs = outputs.cpu().numpy()
        
        # Denormalize
        if denormalize:
            outputs = self.denormalize_outputs(outputs)
        
        return outputs
    
    def predict_from_sample_id(
        self,
        sample_id: int,
        data_loader: Optional[CFDDataLoader] = None,
        coords: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict CFD fields for a sample from the dataset.
        
        Args:
            sample_id: Sample ID from input parameters file
            data_loader: CFDDataLoader instance (created if None)
            coords: Optional coordinates (uses available CFD mesh if None)
            
        Returns:
            predictions: (N, 5) predicted field values
            coords: (N, 2) coordinates used
        """
        if data_loader is None:
            data_loader = CFDDataLoader()
        
        # Get input parameters
        params = data_loader.get_input_params(sample_id)
        
        # Get coordinates
        if coords is None:
            if sample_id in data_loader.cfd_data:
                coords, _ = data_loader.get_cfd_output(sample_id)
            else:
                # Use first available CFD mesh as template
                available_ids = data_loader.get_available_sample_ids()
                if available_ids:
                    coords, _ = data_loader.get_cfd_output(available_ids[0])
                else:
                    raise ValueError("No coordinates provided and no CFD data available")
        
        predictions = self.predict(params, coords)
        return predictions, coords
    
    def export_to_csv(
        self,
        predictions: np.ndarray,
        coords: np.ndarray,
        output_path: Path,
        include_node_numbers: bool = True
    ):
        """
        Export predictions to CSV in same format as ANSYS output.
        
        Args:
            predictions: (N, 5) predicted field values
            coords: (N, 2) coordinates
            output_path: Path to save CSV
            include_node_numbers: Whether to include node numbers
        """
        data = {}
        
        if include_node_numbers:
            data['nodenumber'] = np.arange(1, len(coords) + 1)
        
        data['x-coordinate'] = coords[:, 0]
        data['y-coordinate'] = coords[:, 1]
        
        for i, col in enumerate(OUTPUT_COLUMNS):
            data[col] = predictions[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Predictions exported to {output_path}")


def predict_new_parameters(
    params_dict: dict,
    output_path: Path,
    model_path: Path = MODEL_SAVE_DIR / 'best_model.pt'
):
    """
    Convenience function to predict for new parameters.
    
    Usage:
        predict_new_parameters(
            params_dict={
                'First_Ramp_Angle_deg': 10.0,
                'Second_Ramp_Angle_deg': 20.0,
                'Cowl_Deflection_Angle_deg': 5.0,
                # ... etc
            },
            output_path=Path('prediction.csv')
        )
    """
    # Load predictor
    predictor = CFDPredictor(model_path=model_path)
    
    # Load data for coordinates
    data_loader = CFDDataLoader()
    
    # Get coordinates from available simulation
    available_ids = data_loader.get_available_sample_ids()
    if not available_ids:
        raise ValueError("No CFD simulations available to get coordinate template")
    
    coords, _ = data_loader.get_cfd_output(available_ids[0])
    
    # Create parameter array
    params = np.array([params_dict[col] for col in INPUT_COLUMNS], dtype=np.float32)
    
    # Predict
    predictions = predictor.predict(params, coords)
    
    # Export
    predictor.export_to_csv(predictions, coords, output_path)
    
    return predictions


def main():
    """Main inference function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CFD Surrogate Model Inference')
    parser.add_argument('--sample_id', type=int, required=True,
                        help='Sample ID from input parameters file')
    parser.add_argument('--output', type=str, default='prediction.csv',
                        help='Output CSV file path')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    model_path = Path(args.model) if args.model else MODEL_SAVE_DIR / 'best_model.pt'
    output_path = Path(args.output)
    
    # Load predictor
    predictor = CFDPredictor(model_path=model_path)
    
    # Load data
    data_loader = CFDDataLoader()
    
    # Predict
    predictions, coords = predictor.predict_from_sample_id(
        args.sample_id, 
        data_loader
    )
    
    # Export
    predictor.export_to_csv(predictions, coords, output_path)
    
    print(f"\nPrediction complete for Sample_ID {args.sample_id}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
