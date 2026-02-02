# HypersonicInlet-DeepONet - Surrogate Model

A DeepONet-based surrogate model for predicting CFD simulation results of hypersonic inlet flows.

## Overview

This surrogate model uses a **Deep Operator Network (DeepONet)** architecture to learn the mapping from design parameters to CFD flow fields. The model can predict pressure, velocity, and Mach number fields at arbitrary mesh locations for any combination of input parameters.

## Why DeepONet instead of UNet?

| Aspect | UNet | DeepONet |
|--------|------|----------|
| Mesh Type | Regular grid required | **Unstructured mesh supported** |
| Parameter Conditioning | Difficult to incorporate | **Native parameter conditioning** |
| Memory Usage | High (full field at once) | **Low (point-wise)** |
| Adding New Simulations | Retrain from scratch | **Incremental training** |

## Project Structure

```
meshed file/
├── hypersonic_inlet_cfd_dataset_150_samples.csv  # Input parameters
├── simulation_mapping.csv                         # CFD file to parameter mapping
├── requirements.txt                               # Python dependencies
├── cfd_outputs/                                   # CFD simulation outputs
│   ├── SYS-1-00315.csv
│   └── ... (add more as you run them)
└── surrogate_model/
    ├── config.py          # Configuration and hyperparameters
    ├── data_loader.py     # Data loading utilities
    ├── dataset.py         # PyTorch datasets
    ├── model.py           # DeepONet architecture
    ├── train.py           # Training script
    ├── inference.py       # Prediction utilities
    ├── visualize.py       # Visualization tools
    └── checkpoints/       # Saved models
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Add Your CFD Simulations

1. Export CFD results from ANSYS to `cfd_outputs/` folder
2. Update `simulation_mapping.csv` with the mapping:

```csv
filename,sample_id
SYS-1-00315.csv,1
SYS-2-00316.csv,2
...
```

### 3. Train the Model

```bash
cd surrogate_model
python train.py
```

### 4. Make Predictions

```python
from inference import CFDPredictor

# Load trained model
predictor = CFDPredictor()

# Predict for sample ID 5
predictions, coords = predictor.predict_from_sample_id(sample_id=5)
```

### 5. Visualize Results

```bash
python visualize.py --action validate --sample_id 1
```

### 6. Run the Web Interface (HTML Testing)

To test your model through a web interface:

**Install web dependencies:**
```bash
pip install fastapi uvicorn
```

**Run the web server:**
```bash
cd surrogate_model
python app.py
```

**Open your browser:**
Navigate to `http://localhost:8000`

The web interface allows you to:
- Input the 9 design parameters
- View 2D contour plots for all 5 output fields (pressure, density, mach-number, etc.)
- Download prediction results as CSV

## Input Parameters (9 total)

The model accepts 9 input parameters:
- First_Ramp_Angle_deg
- Second_Ramp_Angle_deg  
- Cowl_Deflection_Angle_deg
- Inlet_Length_m
- Throat_Height_m
- Cowl_Lip_Position_m
- Mach_Number
- Static_Pressure_Pa
- Static_Temperature_K

## Output Fields (5 total)

The model predicts:
- pressure
- pressure-coefficient
- density
- mach-number
- total-temperature

## Adding More Simulations

As you run more ANSYS simulations:

1. Export the results to `cfd_outputs/` (e.g., `d4.csv`, `d5.csv`, etc.)
2. Add a new row to `simulation_mapping.csv`
3. Retrain the model (it will use all available data)

Or use the utility function:
```python
from data_loader import add_simulation_to_mapping
add_simulation_to_mapping("d4.csv", sample_id=4)
```

## Model Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   Branch Net    │     │    Trunk Net     │
│ (9 parameters)  │     │   (x, y coords)  │
│       ↓         │     │        ↓         │
│   256→256→256   │     │  Fourier→128→128 │
│       ↓         │     │        ↓         │
│  [b1,b2...bp]   │     │   [t1,t2...tp]   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ∑ bᵢ × tᵢ (dot product)
                     │
                     ↓
        [pressure, Cp, Vx, Vy, Mach]
```

## Tips for Best Results

1. **More simulations = better model**: The model will improve as you add more CFD results
2. **Diverse parameter coverage**: Ensure your simulations span the parameter space
3. **GPU training**: Use `--device cuda` if available for faster training
4. **Validation**: Always check predictions against ground truth before using for design

## License

MIT License
