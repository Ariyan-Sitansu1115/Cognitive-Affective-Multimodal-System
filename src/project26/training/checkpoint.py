import os
import torch


def load_checkpoint(model, ckpt_path, device):
    print("Checkpoint exists:", os.path.exists(ckpt_path)); print("Checkpoint path:", ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint); model.to(device); model.eval()
    print("\nBest Hybrid checkpoint loaded successfully.")
    print("Model:", type(model).__name__); print("Device:", next(model.parameters()).device)
    print("Evaluation mode:", not model.training)
    return model
