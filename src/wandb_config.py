import os

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your_team_or_username")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "aimc-rotations")
WANDB_MODE = os.getenv("WANDB_MODE", "online")