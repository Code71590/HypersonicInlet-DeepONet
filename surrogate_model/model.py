"""
DeepONet Model for CFD Surrogate Modeling

Architecture:
- Branch Network: Encodes input parameters (design conditions) 
- Trunk Network: Encodes spatial coordinates (x, y)
- Output: Dot product of branch and trunk outputs, decoded to field values
"""
import torch
import torch.nn as nn
from typing import List, Tuple

from config import (
    NUM_INPUT_PARAMS, NUM_OUTPUT_FIELDS,
    BRANCH_HIDDEN_DIMS, TRUNK_HIDDEN_DIMS, LATENT_DIM
)


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.0,
        output_activation: bool = False
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(activation())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchNet(nn.Module):
    """
    Branch Network: Encodes input parameters to latent space.
    
    Input: (batch, num_params) - design/operating conditions
    Output: (batch, latent_dim * num_outputs) - latent codes for each output field
    """
    
    def __init__(
        self,
        input_dim: int = NUM_INPUT_PARAMS,
        hidden_dims: List[int] = BRANCH_HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM,
        num_outputs: int = NUM_OUTPUT_FIELDS
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_outputs = num_outputs
        
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim * num_outputs,
            activation=nn.GELU,
            dropout=0.0
        )
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: (batch, input_dim) input parameters
        Returns:
            (batch, num_outputs, latent_dim) latent codes
        """
        out = self.mlp(params)  # (batch, latent_dim * num_outputs)
        return out.view(-1, self.num_outputs, self.latent_dim)


class TrunkNet(nn.Module):
    """
    Trunk Network: Encodes spatial coordinates to latent space.
    
    Input: (batch, 2) - (x, y) coordinates
    Output: (batch, latent_dim * num_outputs) - latent codes for each coordinate
    """
    
    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dims: List[int] = TRUNK_HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM,
        num_outputs: int = NUM_OUTPUT_FIELDS
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_outputs = num_outputs
        
        self.mlp = MLP(
            input_dim=coord_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim * num_outputs,
            activation=nn.GELU,
            dropout=0.0
        )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, 2) spatial coordinates
        Returns:
            (batch, num_outputs, latent_dim) latent codes
        """
        out = self.mlp(coords)  # (batch, latent_dim * num_outputs)
        return out.view(-1, self.num_outputs, self.latent_dim)


class DeepONet(nn.Module):
    """
    Deep Operator Network for CFD surrogate modeling.
    
    Combines Branch (parameter) and Trunk (coordinate) networks
    to predict field values at arbitrary locations.
    """
    
    def __init__(
        self,
        num_input_params: int = NUM_INPUT_PARAMS,
        num_output_fields: int = NUM_OUTPUT_FIELDS,
        branch_hidden_dims: List[int] = BRANCH_HIDDEN_DIMS,
        trunk_hidden_dims: List[int] = TRUNK_HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM
    ):
        super().__init__()
        
        self.branch_net = BranchNet(
            input_dim=num_input_params,
            hidden_dims=branch_hidden_dims,
            latent_dim=latent_dim,
            num_outputs=num_output_fields
        )
        
        self.trunk_net = TrunkNet(
            coord_dim=2,
            hidden_dims=trunk_hidden_dims,
            latent_dim=latent_dim,
            num_outputs=num_output_fields
        )
        
        # Learnable bias for each output field
        self.bias = nn.Parameter(torch.zeros(num_output_fields))
        
        # Save config for loading
        self.config = {
            'num_input_params': num_input_params,
            'num_output_fields': num_output_fields,
            'branch_hidden_dims': branch_hidden_dims,
            'trunk_hidden_dims': trunk_hidden_dims,
            'latent_dim': latent_dim
        }
    
    def forward(
        self, 
        params: torch.Tensor, 
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of DeepONet.
        
        Args:
            params: (batch, num_input_params) input parameters
            coords: (batch, 2) spatial coordinates
            
        Returns:
            outputs: (batch, num_output_fields) predicted field values
        """
        # Get latent representations
        branch_out = self.branch_net(params)  # (batch, num_outputs, latent_dim)
        trunk_out = self.trunk_net(coords)    # (batch, num_outputs, latent_dim)
        
        # Dot product over latent dimension
        outputs = torch.sum(branch_out * trunk_out, dim=-1)  # (batch, num_outputs)
        
        # Add bias
        outputs = outputs + self.bias
        
        return outputs
    
    def predict_field(
        self, 
        params: torch.Tensor, 
        coords: torch.Tensor,
        batch_size: int = 10000
    ) -> torch.Tensor:
        """
        Predict full field for given parameters at all coordinates.
        Handles batching internally for memory efficiency.
        
        Args:
            params: (num_input_params,) single set of input parameters
            coords: (N, 2) all mesh coordinates
            batch_size: Number of points to process at once
            
        Returns:
            outputs: (N, num_output_fields) predicted field values
        """
        self.eval()
        
        # Expand params to match all coords
        num_points = coords.shape[0]
        params = params.unsqueeze(0).expand(num_points, -1)  # (N, num_params)
        
        # Process in batches with inference_mode (faster than no_grad)
        outputs = []
        with torch.inference_mode():
            for i in range(0, num_points, batch_size):
                batch_params = params[i:i+batch_size]
                batch_coords = coords[i:i+batch_size]
                batch_out = self.forward(batch_params, batch_coords)
                outputs.append(batch_out)
        
        return torch.cat(outputs, dim=0)


class ImprovedDeepONet(nn.Module):
    """
    Enhanced DeepONet with:
    - Fourier feature encoding for coordinates
    - Residual connections
    - Layer normalization
    """
    
    def __init__(
        self,
        num_input_params: int = NUM_INPUT_PARAMS,
        num_output_fields: int = NUM_OUTPUT_FIELDS,
        branch_hidden_dims: List[int] = [256, 256, 256],
        trunk_hidden_dims: List[int] = [128, 128, 128],
        latent_dim: int = 128,
        num_fourier_features: int = 64
    ):
        super().__init__()
        
        self.num_fourier_features = num_fourier_features
        
        # Fourier feature frequencies (learnable)
        self.register_buffer(
            'freq_bands', 
            torch.linspace(1, 50, num_fourier_features // 2)
        )
        
        # Fourier encoding expands 2D coords to higher dim
        fourier_dim = 2 + 2 * num_fourier_features  # original + sin/cos
        
        self.branch_net = self._build_branch(
            num_input_params, branch_hidden_dims, latent_dim, num_output_fields
        )
        self.trunk_net = self._build_trunk(
            fourier_dim, trunk_hidden_dims, latent_dim, num_output_fields
        )
        
        self.bias = nn.Parameter(torch.zeros(num_output_fields))
        self.latent_dim = latent_dim
        self.num_outputs = num_output_fields
        
        self.config = {
            'num_input_params': num_input_params,
            'num_output_fields': num_output_fields,
            'branch_hidden_dims': branch_hidden_dims,
            'trunk_hidden_dims': trunk_hidden_dims,
            'latent_dim': latent_dim,
            'num_fourier_features': num_fourier_features
        }
    
    def _build_branch(self, input_dim, hidden_dims, latent_dim, num_outputs):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim * num_outputs))
        return nn.Sequential(*layers)
    
    def _build_trunk(self, input_dim, hidden_dims, latent_dim, num_outputs):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim * num_outputs))
        return nn.Sequential(*layers)
    
    def fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding to coordinates."""
        # coords: (batch, 2)
        # Multiply by frequency bands
        scaled = coords.unsqueeze(-1) * self.freq_bands  # (batch, 2, num_freqs)
        scaled = scaled.reshape(coords.shape[0], -1)  # (batch, 2*num_freqs)
        
        # Compute sin and cos
        encoded = torch.cat([
            coords,
            torch.sin(scaled),
            torch.cos(scaled)
        ], dim=-1)
        
        return encoded
    
    def forward(
        self, 
        params: torch.Tensor, 
        coords: torch.Tensor
    ) -> torch.Tensor:
        # Encode coordinates with Fourier features
        coords_encoded = self.fourier_encode(coords)
        
        # Get latent representations
        branch_out = self.branch_net(params)
        branch_out = branch_out.view(-1, self.num_outputs, self.latent_dim)
        
        trunk_out = self.trunk_net(coords_encoded)
        trunk_out = trunk_out.view(-1, self.num_outputs, self.latent_dim)
        
        # Dot product
        outputs = torch.sum(branch_out * trunk_out, dim=-1) + self.bias
        
        return outputs
    
    def predict_field(
        self, 
        params: torch.Tensor, 
        coords: torch.Tensor,
        batch_size: int = 10000
    ) -> torch.Tensor:
        """Predict full field - same as DeepONet."""
        self.eval()
        num_points = coords.shape[0]
        params = params.unsqueeze(0).expand(num_points, -1)
        
        # Use inference_mode for faster inference (skips version tracking)
        outputs = []
        with torch.inference_mode():
            for i in range(0, num_points, batch_size):
                batch_params = params[i:i+batch_size]
                batch_coords = coords[i:i+batch_size]
                batch_out = self.forward(batch_params, batch_coords)
                outputs.append(batch_out)
        
        return torch.cat(outputs, dim=0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(improved: bool = True) -> nn.Module:
    """Factory function to create model."""
    if improved:
        model = ImprovedDeepONet()
    else:
        model = DeepONet()
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    return model


if __name__ == "__main__":
    # Test model
    print("Testing DeepONet...")
    model = DeepONet()
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 32
    params = torch.randn(batch_size, NUM_INPUT_PARAMS)
    coords = torch.randn(batch_size, 2)
    
    outputs = model(params, coords)
    print(f"Input shapes: params={params.shape}, coords={coords.shape}")
    print(f"Output shape: {outputs.shape}")
    
    print("\nTesting ImprovedDeepONet...")
    model = ImprovedDeepONet()
    print(f"Parameters: {count_parameters(model):,}")
    
    outputs = model(params, coords)
    print(f"Output shape: {outputs.shape}")
