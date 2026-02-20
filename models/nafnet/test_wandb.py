import wandb
import os
print(f"Current working dir: {os.getcwd()}")
print(f"WANDB_MODE env: {os.environ.get('WANDB_MODE')}")
try:
    wandb.init(project="test_debug_mode", mode="online")
    print(f"WandB run mode: {wandb.run.mode}")
    print(f"WandB settings mode: {wandb.run.settings.mode}")
    wandb.finish()
except Exception as e:
    print(f"WandB init failed: {e}")
