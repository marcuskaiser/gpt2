import time

import torch
from torch import nn
from torch.optim import AdamW


class SimpleTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
    ) -> None:
        self.model = model
        self.lr = lr

        self.optimizer: AdamW
        self._reset_optimizer()

    def _reset_optimizer(self) -> None:
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
        )

    def train_model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_train_steps: int = 50,
    ) -> None:
        self.model.train()

        t0 = time.time()
        n_train_steps_digits = len(str(n_train_steps))
        for i in range(n_train_steps):
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            print(f"Step: {i:{n_train_steps_digits}d}, loss: {loss.item():.3e}")

        t1 = time.time()
        print(f"Total time: {t1 - t0:.3}s.")
