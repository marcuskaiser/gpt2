from __future__ import annotations
import logging
import time

from torch import nn
from torch.optim import AdamW

from gpt.data_loader import SimpleDataLoader

logger = logging.getLogger(__name__)


class SimpleTrainer:
    def __init__(
        self,
        model: nn.Module,
        data_loader: SimpleDataLoader,
        lr: float = 3e-4,
        num_accumulation_steps: int = 1,
    ) -> None:
        self.model = model
        assert isinstance(self.model, nn.Module)

        self.data_loader = data_loader
        assert isinstance(self.data_loader, SimpleDataLoader)

        self.lr = lr
        assert self.lr > 0

        self.num_accumulation_steps = num_accumulation_steps
        assert num_accumulation_steps >= 1, self.num_accumulation_steps

        self.optimizer: AdamW
        self._reset_optimizer()

    def _reset_optimizer(self) -> None:
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            fused=self.model.config.device == "cuda",
        )
        self.optimizer.zero_grad()

    def train_model(
        self,
        num_train_steps: int = 50,
    ) -> SimpleTrainer:
        """Training routine. Train for num_train_steps steps."""
        self.model.train()

        # TODO! Random seed

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

        t_init = t_last = time.time()
        for i_step in range(num_train_steps):

            # Gradient accumulation:
            loss_est = 0
            for _ in range(self.num_accumulation_steps):
                x, y = self.data_loader.get_next_training_batch()

                # with torch.autocast(
                #     device_type=self.model.config.device,
                #     dtype=self.model.config.torch_dtype,
                # ):
                # calculate and accumulate loss:
                _, loss = self.model(x, y)

                # normalize loss to adjust for multiple accumulation steps:
                loss = loss / self.num_accumulation_steps
                # calculate backward path and update optimizer:
                loss.backward()

                loss_est += loss.item()

            norm = nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=1.0,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            t_last = _update_logging_progress()

        t_final = time.time()
        logger.info("Total time: %.3fs.", t_final - t_init)
        return self
