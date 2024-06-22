"""Trainer class."""

from __future__ import annotations

import logging
import time

import torch
from pydantic import BaseModel
from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import ReduceOp, all_reduce
from torch.nn.parallel import DistributedDataParallel

from gpt.data_loader import SimpleDataLoader
from gpt.utils import DTYPE_MAP, TYPE_OPTIMIZER, get_optimizer

logger = logging.getLogger(name=__name__)


class TrainingConfig(BaseModel):
    lr: float = 3e-4
    num_accumulation_steps: int = 1
    use_scaler: bool = True
    optimizer: TYPE_OPTIMIZER = "adamw"


class NoScaler:
    """Dummy class that is equivalent to no scaler being present."""

    def scale(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Dummy scale function. Simply returns the input."""
        return loss

    def unscale_(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Dummy unscale_ function."""

    def update(
        self,
    ) -> None:
        """Dummy update function."""

    def step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Dummy step function. Simply calls `optimizer.step()`."""
        optimizer.step()


class SimpleTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        data_loader: SimpleDataLoader,
    ) -> None:
        """Basic implementation of a trainer class."""

        self.config = config
        assert isinstance(self.config, TrainingConfig)

        self.model = model
        assert isinstance(self.model, nn.Module)
        self._ddp = isinstance(model, DistributedDataParallel)

        self.data_loader = data_loader
        assert isinstance(self.data_loader, SimpleDataLoader)

        self.lr = config.lr
        assert self.lr > 0

        self.num_accumulation_steps = config.num_accumulation_steps
        assert self.num_accumulation_steps >= 1, self.num_accumulation_steps

        self._is_mps = self.model.config.device == "mps"
        self._is_cuda = self.model.config.device == "cuda"

        if self.config.use_scaler:
            assert self._is_cuda, (
                "use_scaler only works with device=`cuda`. "
                f"Got: {self.model.config.device}"
            )
        else:
            assert (
                self.model.config.autocast_dtype != "fp16"
            ), "Cannot use autocast_dtype=`fp16` with use_scaler=`False`!"

        self.optimizer: torch.optim.Optimizer
        self._reset_optimizer()

    def _reset_optimizer(self) -> None:
        self.optimizer = get_optimizer(
            optimizer=self.config.optimizer,
            params=self.model.parameters(),
            lr=self.lr,
        )
        # NB: Not all optimizer versions have the `fused` param:
        if self._is_cuda and hasattr(self.optimizer, "fused"):
            self.optimizer.fused = True

        self.optimizer.zero_grad()

    def _model_forward_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_mps:
            _, loss = self.model(x, y)
        else:
            with torch.autocast(
                device_type=self.model.config.device,
                dtype=DTYPE_MAP[self.model.config.autocast_dtype],
            ):
                _, loss = self.model(x, y)

        return loss

    def train_model(
        self,
        num_train_steps: int = 50,
    ) -> SimpleTrainer:
        """Training routine. Train for num_train_steps steps."""
        self.model.train()

        # TODO! Random seed
        # TODO! Learning rate scheduler
        # TODO! Weight decay
        # TODO! Validation loss

        train_logging_str = ", ".join(
            [
                f"step: %{len(str(num_train_steps))}d",
                "loss: %.3e",
                "norm: %.3e",
                "time_last_step: %.3es",
                "tokens/s: %.2f",
            ]
        )

        def _update_logging_progress() -> float:
            t_this = time.time()
            t_diff = t_this - t_last
            logger.info(
                train_logging_str,
                i_step,
                loss_est,
                norm,
                t_diff,
                total_batch_size / t_diff,
            )
            return t_this

        total_batch_size = (
            self.num_accumulation_steps * self.data_loader.eff_batch_size
        )

        scaler = NoScaler()
        if self.config.use_scaler and self.model.config.device == "cuda":
            scaler = GradScaler()

        t_init = t_last = time.time()
        for i_step in range(num_train_steps):

            # Gradient accumulation:
            loss_est = torch.zeros(1)
            for _ in range(self.num_accumulation_steps):
                x, y = self.data_loader.get_next_training_batch()

                if self._ddp:
                    self.model.require_background_grad_sync = (
                        i_step + 1 == num_train_steps
                    )

                # calculate and accumulate loss on the batch (x, y):
                loss = self._model_forward_loss(x=x, y=y)
                # normalize loss to adjust for multiple accumulation steps:
                loss = loss / self.num_accumulation_steps
                loss_est += loss.clone().detach()

                # calculate backward path and accumulate grads in leaves:
                scaler.scale(loss).backward()

            if self._ddp:
                loss_est = all_reduce(loss_est, op=ReduceOp.AVG)

            # unscales gradients inplace:
            scaler.unscale_(self.optimizer)

            # clip norms:
            norm = nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=1.0,
            )
            # NB: gradients already unscaled, so scaler.step does not unscale
            #   again, but still skips the step if gradients are NANs.
            scaler.step(self.optimizer)

            # updates the scale for next iteration:
            scaler.update()

            # reset accumulated gradients:
            self.optimizer.zero_grad()

            if self._is_cuda:
                torch.cuda.synchronize()

            t_last = _update_logging_progress()

        t_final = time.time()
        logger.info("Total time: %.3fs.", t_final - t_init)
        return self
