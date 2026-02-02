import pandas as pd
import glob
import os

def diagnose_csv_columns(folder_path, required_columns):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Scanning {len(csv_files)} files...\n")
    
    issues_found = False

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            # Read just the header (nrows=0) to be fast
            df = pd.read_csv(file_path, nrows=0)
            
            # Clean up column names (strip whitespace just in case)
            actual_columns = [col.strip() for col in df.columns]
            
            # Check for missing columns
            missing = [col for col in required_columns if col not in actual_columns]
            
            if missing:
                issues_found = True
                print(f"❌ MISMATCH: '{filename}'")
                print(f"   Missing: {missing}")
                print(f"   Actual columns found: {actual_columns}\n")
                
        except Exception as e:
            print(f"⚠️ ERROR reading '{filename}': {e}\n")

    if not issues_found:
        print("✅ All files have the correct columns!")
    else:
        print("Check the 'Actual columns found' above. You might need to update the 'required_columns' list in the script to match the CSV headers exactly.")

# --- Configuration ---
if __name__ == "__main__":
    folder_path = '.' 
    
    # accurately list the columns you expect
    target_columns = [
        'cellnumber', 
        'x-coordinate', 
        'y-coordinate', 
        'pressure', 
        'density', 
        'velocity-magnitude', 
        'mach-number', 
        'temperature'
    ]

    diagnose_csv_columns(folder_path, target_columns)