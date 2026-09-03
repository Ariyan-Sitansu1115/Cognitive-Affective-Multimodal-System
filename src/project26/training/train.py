import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_classifier(model, train_loader, val_loader, epochs, lr, ckpt_path, device, train_targets, n_classes):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    counts = np.bincount(train_targets.to_numpy(), minlength=n_classes)
    class_weights = torch.tensor(len(train_targets) / (n_classes * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')
    for epoch in epoch_bar:
        model.train(); train_loss = 0.0; n = 0
        batch_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False, unit='batch')
        for x, y in batch_bar:
            x, y = (x.to(device), y.to(device)); optimizer.zero_grad()
            loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            train_loss += loss.item() * y.shape[0]; n += y.shape[0]
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')
        train_loss /= n; model.eval(); val_loss = 0.0; nv = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = (x.to(device), y.to(device)); val_loss += criterion(model(x), y).item() * y.shape[0]; nv += y.shape[0]
        val_loss /= nv; scheduler.step(val_loss)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        epoch_bar.set_postfix(train_loss=f'{train_loss:.4f}', val_loss=f'{val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss; torch.save(model.state_dict(), ckpt_path)
    return history
