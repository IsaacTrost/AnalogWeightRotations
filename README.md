# My workflow

(base) ao2844_columbia_edu@instance-20260325-210342:~/AnalogWeightRotations/aimc-docker$ sudo docker build --no-cache -t aihwkit-min .

$ export WANDB_API_KEY=<your-api-key>

$ export WANDB_ENTITY=ao2844-columbia-university

$ export WANDB_PROJECT=aimc-rotations

Mind the directory I run it in isn't aimc-docker:
(base) ao2844_columbia_edu@instance-20260325-210342:~/AnalogWeightRotations$ sudo docker run --gpus all -it --rm   -p 8888:8888   -v $(pwd):/workspace   -e WANDB_API_KEY=$WANDB_API_KEY   -e WANDB_ENTITY=$WANDB_ENTITY   -e WANDB_PROJECT=$WANDB_PROJECT   aihwkit-min

$ conda run -n aihwkit python -m src.baseline_forward

## Jupyter
$ source /opt/conda/etc/profile.d/conda.sh
$ conda activate aihwkit
$ conda run -n aihwkit jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Overview
src/baseline_forward.py
Loads a Hugging Face model (currently GPT-2)
Runs a baseline forward pass
Captures intermediate activations via hooks
Logs basic metrics to W&B

Purpose: establish a reference point before applying rotations

src/baseline_forward.py
Loads a Hugging Face model (currently GPT-2)
Runs a baseline forward pass
Captures intermediate activations via hooks
Logs basic metrics to W&B

Purpose: establish a reference point before applying rotations

src/rotation_utils.py

Core math utilities:

random_orthogonal_matrix(n)
hadamard_matrix(n) (power-of-2 only)
apply_rotation(x, R)
helper functions (orthonormal checks, etc.)

Purpose: define and apply rotation matrices

src/apply_rotation.py
Runs baseline forward pass
Applies rotation → inverse rotation
Compares outputs with baseline
Logs error metrics to W&B

Purpose: verify rotations preserve model computation (sanity check)