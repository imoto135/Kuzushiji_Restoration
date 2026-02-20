import torch  # Must be imported first to avoid crash
import os
os.environ["WANDB_MODE"] = "offline"
import sys

# Ensure current directory is in path (for basicsr import if not installed)
# But if installed via pip -e ., it might be redundant but safe if strict
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    # Append to end to defer to pip installed version if present,
    # or insert at 0 if we want to override.
    # pip install -e . makes it same.
    sys.path.insert(0, current_dir)

from basicsr.train import main

if __name__ == "__main__":
    try:
        print("Starting training...", flush=True)
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}", flush=True)
