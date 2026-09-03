import random
import sys

import numpy as np
import torch

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def configure_runtime():
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Python:", sys.version.split()[0]); print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available()); print("Device selected:", device)
    if torch.cuda.is_available(): print("GPU name:", torch.cuda.get_device_name(0))
    return device
