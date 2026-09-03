import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, X_df, y_series):
        self.X = X_df.values.astype(np.float32)
        self.y = y_series.values.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def create_loaders(train_X, val_X, test_X, train_df, val_df, test_df, target, batch_size):
    train_ds = TabularDataset(train_X, train_df[target])
    val_ds = TabularDataset(val_X, val_df[target])
    test_ds = TabularDataset(test_X, test_df[target])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    xb, yb = next(iter(train_loader))
    print("features:", xb.shape, "target:", yb.shape)
    return train_loader, val_loader, test_loader, xb
