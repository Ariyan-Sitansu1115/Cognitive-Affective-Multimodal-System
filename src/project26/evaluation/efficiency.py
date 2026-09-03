import json
import time
import torch


def measure_efficiency(model, test_loader, results_dir):
    total_params = sum(parameter.numel() for parameter in model.parameters()); trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.eval(); xb_b, _ = next(iter(test_loader)); xb_b = xb_b.to(next(model.parameters()).device)
    with torch.no_grad():
        for _ in range(3): model(xb_b)
        start = time.time()
        for _ in range(20): model(xb_b)
        elapsed = (time.time() - start) / 20
    efficiency = {'total_params': total_params, 'trainable_params': trainable_params, 'avg_batch_inference_time_sec': elapsed, 'throughput_samples_per_sec': xb_b.shape[0] / elapsed}
    with open(f'{results_dir}/efficiency.json', 'w') as file: json.dump({'hybrid': efficiency}, file, indent=2)
    return efficiency
