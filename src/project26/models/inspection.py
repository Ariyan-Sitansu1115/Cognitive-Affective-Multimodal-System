def verify_architecture(model, input_batch, device):
    print(model)
    total = sum((parameter.numel() for parameter in model.parameters()))
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    print(f"hybrid: total={total:,} trainable={trainable:,} device={next(model.parameters()).device}")
