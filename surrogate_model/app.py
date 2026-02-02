
import os
import io
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Set matplotlib backend to Agg for non-interactive server use
matplotlib.use('Agg')

from config import INPUT_COLUMNS, OUTPUT_COLUMNS, MODEL_SAVE_DIR, CFD_OUTPUTS_DIR, COORD_COLUMNS
from inference import CFDPredictor

# Global variables to hold model and template mesh
model_resources = {}

# Cache for last prediction to avoid re-running inference on CSV download
prediction_cache = {
    'params': None,
    'predictions': None,
    'coords': None
}

def load_coordinate_template():
    """Load coordinates from a single CFD file (not all files)."""
    import re
    
    # Find first available d{N}.csv file
    cfd_files = list(CFD_OUTPUTS_DIR.glob("d*.csv"))
    if not cfd_files:
        raise RuntimeError("No CFD output files found in cfd_outputs/")
    
    # Sort to get first file consistently
    cfd_files.sort(key=lambda f: int(re.search(r'd(\d+)\.csv', f.name).group(1)))
    
    first_file = cfd_files[0]
    print(f"Loading coordinate template from: {first_file.name}")
    
    df = pd.read_csv(first_file)
    df.columns = df.columns.str.strip()
    coords = df[list(COORD_COLUMNS)].values.astype(np.float32)
    
    print(f"Loaded {len(coords)} coordinate points")
    return coords

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and template mesh
    print("Loading model and template mesh...")
    try:
        predictor = CFDPredictor(model_path=MODEL_SAVE_DIR / 'best_model.pt')
        
        # Load ONLY coordinate template from first file (fast startup)
        coords = load_coordinate_template()
        
        # Create triangulation once for faster plotting
        try:
            tri = Triangulation(coords[:, 0], coords[:, 1])
        except Exception as e:
            print(f"Warning: Triangulation failed: {e}")
            tri = None

        model_resources['predictor'] = predictor
        model_resources['coords'] = coords
        model_resources['tri'] = tri
        model_resources['x'] = coords[:, 0]
        model_resources['y'] = coords[:, 1]
        
        # Warm-up inference to pre-compile CUDA kernels and avoid first-request latency
        print("Warming up model with dummy inference...")
        dummy_params = np.zeros(len(INPUT_COLUMNS), dtype=np.float32)
        _ = predictor.predict(dummy_params, coords)
        print("Warm-up complete.")
        
        print("Model and resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        # In production this might be fatal, but for dev we'll let it start so we can see errors
    
    yield
    
    # Shutdown: Clear resources
    model_resources.clear()

app = FastAPI(lifespan=lifespan)



class PredictionRequest(BaseModel):
    # Using defaults from a typical sample (approximate)
    First_Ramp_Angle_deg: float = 10.0
    Second_Ramp_Angle_deg: float = 15.0
    Cowl_Deflection_Angle_deg: float = 5.0
    Inlet_Length_m: float = 1.0
    Throat_Height_m: float = 0.1
    Cowl_Lip_Position_m: float = 0.5
    Mach_Number: float = 5.0
    Static_Pressure_Pa: float = 2000.0
    Static_Temperature_K: float = 220.0

def generate_contour_plot(x, y, tri, data, title):
    """Generate a contour plot (heatmap)."""
    # Use a wider figure to match the inlet aspect ratio (approx 2:1)
    # Reduced size and DPI for faster rendering and lower memory usage
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    
    if tri is not None:
        # shading='gouraud' makes it smoother than 'flat'
        im = ax.tripcolor(tri, data, shading='gouraud', cmap='jet')
    else:
        im = ax.scatter(x, y, c=data, s=1, cmap='jet')
        
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, aspect=30, pad=0.02)
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.post("/predict")
async def predict(request: PredictionRequest):
    if 'predictor' not in model_resources:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Convert request to numpy array in correct order
    try:
        params_dict = request.model_dump()
        params = np.array([params_dict[col] for col in INPUT_COLUMNS], dtype=np.float32)
        
        # Run inference
        predictor = model_resources['predictor']
        coords = model_resources['coords']
        
        # Output shape: (N, 5)
        predictions = predictor.predict(params, coords)
        
        # Cache the prediction for CSV download
        prediction_cache['params'] = params.copy()
        prediction_cache['predictions'] = predictions.copy()
        prediction_cache['coords'] = coords.copy()
        
        # Generate plots (Contour)
        tri = model_resources['tri']
        x = model_resources['x']
        y = model_resources['y']
        
        images = {}
        
        # Generate all 5 plots in parallel for faster response
        def generate_plot_for_field(field_idx_and_name):
            idx, field_name = field_idx_and_name
            return field_name, generate_contour_plot(x, y, tri, predictions[:, idx], field_name)
        
        # Use 2 workers to balance speed vs memory (5 causes MemoryError)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(
                generate_plot_for_field,
                enumerate(OUTPUT_COLUMNS)
            )
            for field_name, img_b64 in results:
                images[field_name] = img_b64
            
        return {"images": images}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download_csv")
async def download_csv(request: PredictionRequest):
    if 'predictor' not in model_resources:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        params_dict = request.model_dump()
        params = np.array([params_dict[col] for col in INPUT_COLUMNS], dtype=np.float32)
        
        # Check if we can use cached prediction
        if (prediction_cache['params'] is not None and 
            np.allclose(prediction_cache['params'], params)):
            # Use cached results (fast path!)
            print("Using cached prediction for CSV download")
            predictions = prediction_cache['predictions']
            coords = prediction_cache['coords']
        else:
            # Need to run new prediction
            print("Running new prediction for CSV download")
            predictor = model_resources['predictor']
            coords = model_resources['coords']
            
            # Predict
            predictions = predictor.predict(params, coords)
            
            # Update cache
            prediction_cache['params'] = params.copy()
            prediction_cache['predictions'] = predictions.copy()
            prediction_cache['coords'] = coords.copy()
        
        # Create DataFrame
        df = pd.DataFrame()
        df['x-coordinate'] = coords[:, 0]
        df['y-coordinate'] = coords[:, 1]
        
        for i, col in enumerate(OUTPUT_COLUMNS):
            df[col] = predictions[:, i]
            
        # Convert to CSV
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=prediction.csv"
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# Mount static files (catch-all for SPA/static) - Must be LAST
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
