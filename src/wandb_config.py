import os
import pathlib


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries from the repo `.env` when the shell has not set them."""
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_local_env()

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "ao2844-columbia-university")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "analog_rotation_dual_loss")
WANDB_MODE = os.getenv("WANDB_MODE", "online")