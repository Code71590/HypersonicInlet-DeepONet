"""
Configuration for CFD Surrogate Model Training
"""
import os
from pathlib import Path

# ============== PATHS ==============
import sys
if getattr(sys, 'frozen', False):
    # If running as a compiled executable (Nuitka/PyInstaller)
    # The executable will be in a folder, and we want paths relative to that folder
    BASE_DIR = Path(sys.executable).parent
else:
    # Development mode
    BASE_DIR = Path(__file__).parent.parent

# Allow overriding data directory via environment variable or default to BASE_DIR
DATA_DIR = Path(os.environ.get('CFD_DATA_DIR', BASE_DIR))
CFD_OUTPUTS_DIR = DATA_DIR / "cfd_outputs"
INPUT_PARAMS_FILE = DATA_DIR / "cfd_inputs" / "hypersonic_inlet_input_parameters.csv"
SIMULATION_MAPPING_FILE = DATA_DIR / "simulation_mapping.csv"
MODEL_SAVE_DIR = Path(__file__).parent / "checkpoints"

# Create directories if they don't exist
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============== INPUT PARAMETERS ==============
# Column names for input parameters (excluding Sample_ID)
# Using the 9 parameters from the new dataset format
INPUT_COLUMNS = [
    'First_Ramp_Angle_deg',
    'Second_Ramp_Angle_deg',
    'Cowl_Deflection_Angle_deg',
    'Inlet_Length_m',
    'Throat_Height_m',
    'Cowl_Lip_Position_m',
    'Mach_Number',
    'Static_Pressure_Pa',
    'Static_Temperature_K'
]
NUM_INPUT_PARAMS = len(INPUT_COLUMNS)  # 9 parameters

# ============== OUTPUT FIELDS ==============
# Column names for CFD output fields (excluding node coordinates)
# New format: pressure, pressure-coefficient, density, mach-number, total-temperature
OUTPUT_COLUMNS = [
    'pressure',
    'density',
    'velocity-magnitude',
    'mach-number',
    'temperature'
]
NUM_OUTPUT_FIELDS = len(OUTPUT_COLUMNS)  # 5 output fields

# Coordinate columns in CFD output
COORD_COLUMNS = ['x-coordinate', 'y-coordinate']

# ============== MODEL ARCHITECTURE ==============
# DeepONet configuration
BRANCH_HIDDEN_DIMS = [128, 128, 128]  # MLP layers for parameter encoding
TRUNK_HIDDEN_DIMS = [64, 64, 64]      # MLP layers for coordinate encoding
LATENT_DIM = 128                       # Dimension of latent space

# ============== TRAINING ==============
BATCH_SIZE = 4096          # Number of points per batch
NUM_EPOCHS = 500           # Maximum number of epochs
LEARNING_RATE = 1e-3       # Initial learning rate
WEIGHT_DECAY = 1e-5        # L2 regularization
PATIENCE = 100             # Early stopping patience (increased to allow more training)
MIN_DELTA = 0              # Any improvement counts (prevents early stopping from being too aggressive)

# Points to sample per simulation during training
POINTS_PER_SIMULATION = 8192

# Train/validation split ratio
TRAIN_RATIO = 0.8

# ============== DEVICE ==============
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============== LOGGING ==============
LOG_INTERVAL = 10  # Log every N epochs
SAVE_INTERVAL = 50  # Save checkpoint every N epochs
