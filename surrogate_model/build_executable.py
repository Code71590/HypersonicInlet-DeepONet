
import os
import sys
import subprocess
import shutil
from pathlib import Path

def get_site_packages_dir():
    """Find the site-packages directory where libraries are installed."""
    import site
    return Path(site.getsitepackages()[0]) # Usually the first one is the main one

def copy_package(package_name, dest_dir):
    """Copy a package directory from site-packages to destination."""
    print(f"  - Copying {package_name}...", end="", flush=True)
    try:
        # Try importing to find the location
        import importlib.util
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
             # origin is .../package/__init__.py, we want .../package
            src = Path(spec.origin).parent
            dst = dest_dir / package_name
            
            if dst.exists():
                shutil.rmtree(dst)
            
            # Use ignore patterns to skip __pycache__ and other junk
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyd-o', '.git'))
            print(" Done.")
            return True
    except Exception as e:
        print(f" Failed: {e}")
        return False
        
    print(" Not found.")
    return False

def build():
    """Build the training script into a standalone executable using fast-build mode."""
    
    # Check if Nuitka is installed
    try:
        import nuitka
    except ImportError:
        print("Error: Nuitka is not installed. Please install it with:")
        print("  pip install nuitka")
        sys.exit(1)

    # Configuration
    # Use the directory where this script is located to find train.py
    SCRIPT_DIR = Path(__file__).parent.absolute()
    MAIN_SCRIPT = SCRIPT_DIR / "train.py"
    # Create build_dist in the parent directory (project root) or same dir? 
    # Let's put it in the same directory as the script for simplicity
    OUTPUT_DIR = SCRIPT_DIR.parent / "build_dist"
    
    # Packages to skip compilation for speed (will copy manually)
    # CORE LIBRARIES
    SKIP_PACKAGES = ['torch', 'numpy', 'pandas', 'matplotlib', 'scipy']
    
    # TRANSITIVE DEPENDENCIES (Must copy these too!)
    # Torch dependencies
    SKIP_PACKAGES += ['typing_extensions', 'filelock', 'fsspec', 'jinja2', 'networkx', 'sympy', 'mpmath', 'markupsafe']
    # Pandas dependencies
    SKIP_PACKAGES += ['dateutil', 'pytz', 'six', 'packaging']
    # Matplotlib dependencies
    SKIP_PACKAGES += ['cycler', 'kiwisolver', 'pyparsing', 'pillow', 'fonttools']
    
    # Nuitka arguments
    
    # Nuitka arguments
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--follow-imports", # Follow user imports
        # Exclude heavy libs from analysis/compilation
        *[f"--nofollow-import-to={pkg}" for pkg in SKIP_PACKAGES], 
        # But we DO want to include simple plugins if possible, or just rely on manual copy
        "--enable-plugin=numpy", 
        f"--output-dir={OUTPUT_DIR}",
        "--remove-output",
        "--show-progress",
        str(MAIN_SCRIPT)
    ]
    
    # Clean previous build
    if OUTPUT_DIR.exists():
        print(f"Cleaning previous build at {OUTPUT_DIR}...")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except:
             pass # might be locked, Nuitka will complain
        
    print(f"Starting FAST compilation of {MAIN_SCRIPT}...")
    print("Optimization: Skipping heavy compilation for: " + ", ".join(SKIP_PACKAGES))
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        
        print("\n" + "="*50)
        print("COMPILATION SUCCESSFUL")
        print("Starting Post-Build Copy (copying libraries manually)...")
        print("="*50)
        
        dist_folder = OUTPUT_DIR / "train.dist"
        
        # Manually copy the skipped packages
        # We need to make sure they end up in the root of the dist folder 
        # or inside a location where python finds them (usually root of dist works for standalone)
        
        success = True
        for pkg in SKIP_PACKAGES:
            if not copy_package(pkg, dist_folder):
                success = False
                print(f"CRITICAL ERROR: Failed to copy {pkg}.")
        
        if not success:
            print("\n" + "="*50)
            print("BUILD PARTIALLY FAILED (Missing dependencies)")
            print("="*50)
            sys.exit(1)
            
        print("\n" + "="*50)
        print("BUILD COMPLETE")
        print("="*50)
        
        print(f"Executable created at: {dist_folder}")
        print("\nTo use this on another computer:")
        print(f"1. Copy the entire '{dist_folder.name}' folder")
        print("2. Copy your 'cfd_inputs' and 'cfd_outputs' folders")
        print("3. Run the valid command (see HOW_TO_RUN.md)")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*50)
        print("BUILD FAILED")
        print("="*50)
        sys.exit(e.returncode)

if __name__ == "__main__":
    build()
