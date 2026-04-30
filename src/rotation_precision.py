import torch

# Use float64 for offline rotation and fusion math, then cast back to the model dtype.
ROTATION_COMPUTE_DTYPE = torch.float64
