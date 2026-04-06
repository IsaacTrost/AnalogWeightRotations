import torch
from torch import Tensor
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.tiles import FloatingPointTile
from aihwkit.simulator.configs import FloatingPointRPUConfig

def run():
    print("=== ENV CHECK ===")
    print("Torch version:", torch.__version__)
    print("Torch CUDA available:", torch.cuda.is_available())
    print("AIHWKIT compiled with CUDA:", cuda.is_compiled())

    print("\n=== CPU TEST ===")
    model = AnalogLinear(2, 2)
    x = Tensor([[0.1, 0.2], [0.3, 0.4]])
    y = model(x)
    print("CPU output:", y)

    print("\n=== GPU TEST ===")
    if torch.cuda.is_available() and cuda.is_compiled():
        model = model.cuda()
        x = x.cuda()
        y = model(x)
        print("GPU output:", y)
    else:
        print("GPU path not available")

    print("\n=== TILE TEST ===")
    config = FloatingPointRPUConfig()
    tile = FloatingPointTile(10, 20, rpu_config=config)
    print("CPU tile:", type(tile))

    if cuda.is_compiled():
        tile_gpu = tile.cuda()
        print("GPU tile:", type(tile_gpu))
    else:
        print("GPU tile not available")

if __name__ == "__main__":
    run()
