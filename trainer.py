"""
Training utilities for autoencoders.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    patience: int = 8,
    model_name: str = "autoencoder"
) -> Dict:

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val = float("inf")
    wait = 0

    for epoch in range(num_epochs):
        # ------------ Train ------------
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # ------------ Validate ------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # ------------ Early stopping ------------
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), f"{config.MODEL_PATH}/{model_name}_best.pt")
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


        print(f"Epoch {epoch}/{num_epochs} — train={train_loss:.6f}, val={val_loss:.6f}")

    # Load best weights
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/{model_name}_best.pt"))

    return history