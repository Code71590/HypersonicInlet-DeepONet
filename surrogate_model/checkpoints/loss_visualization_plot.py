import json
import matplotlib.pyplot as plt

# 1. Load the data from the file
filename = 'surrogate_model/checkpoints/training_history.json'

try:
    with open(filename, 'r') as f:
        history = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find '{filename}'. Make sure the file is in the same folder as this script.")
    exit()

# 2. Extract data
train_loss = history['train_loss']
val_loss = history.get('val_loss', []) # Use .get in case val_loss is missing
learning_rate = history.get('learning_rate', [])

epochs = range(1, len(train_loss) + 1)

# 3. Create the Plot
# We create 2 subplots sharing the same X-axis (Epochs)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# --- Subplot 1: Loss Curves ---
ax1.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=1.5)
if val_loss:
    ax1.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=1.5, alpha=0.9)

ax1.set_ylabel('Loss Value')
ax1.set_title('Training & Validation Loss History')
ax1.legend(loc='upper right')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# OPTIONAL: Use Log Scale
# Since your loss drops from 19.0 to 0.0009, a linear scale makes the end 
# of the training look like a flat line. Log scale reveals the details.
ax1.set_yscale('log') 

# --- Subplot 2: Learning Rate ---
if learning_rate:
    ax2.plot(epochs, learning_rate, label='Learning Rate', color='green', linestyle='--')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', linewidth=0.5)

# --- Final Layout Adjustments ---
plt.xlabel('Epochs')
plt.tight_layout()

# Save the plot
plt.savefig('training_history_plot.png')
print("Plot saved as 'training_history_plot.png'")

# Show the plot
plt.show()