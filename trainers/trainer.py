import torch
import torch.nn as nn
import numpy as np
import os
import copy
from torch.utils.data import DataLoader
from typing import Dict, Optional


class Trainer:
    """训练器: AdamW + 余弦退火 + 早停"""

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 grad_clip: float = 1.0, patience: int = 20,
                 checkpoint_dir: str = 'results/checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        self.criterion = nn.HuberLoss(delta=1.0)
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x_batch)
            loss = self.criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            y_hat = self.model(x_batch)
            loss = self.criterion(y_hat, y_batch)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              max_epochs: int = 200, print_freq: int = 10) -> Dict:
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % print_freq == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | Best: {self.best_val_loss:.6f}")

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 恢复最优模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple:
        """返回 (pred_returns, true_returns, base_prices)"""
        self.model.eval()
        preds, targets, bases = [], [], []

        for batch in loader:
            x_batch = batch[0].to(self.device)
            y_batch = batch[1]
            base_batch = batch[2]

            y_hat = self.model(x_batch)
            preds.append(y_hat.cpu().numpy())
            targets.append(y_batch.numpy())
            bases.append(base_batch.numpy())

        return (np.concatenate(preds, axis=0),
                np.concatenate(targets, axis=0),
                np.concatenate(bases, axis=0))

    def save_checkpoint(self, name: str):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")
