import os

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "ao2844-columbia-university")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "analog_rotation_dual_loss")
WANDB_MODE = os.getenv("WANDB_MODE", "online")