"""Tooling for torch.distributed runs."""

from __future__ import annotations

import logging
import torch
from torch.distributed import (
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
)

from gpt2.utils import DEFAULT_DEVICE_TYPE, DeviceType

logger = logging.getLogger(__name__)


def teardown_ddp() -> None:
    """Teardown after DDP run."""
    destroy_process_group()
    logger.info("DPP Cleanup: Destroyed process group.")


class DDPManager:
    is_ddp_run: bool = False
    world_size: int = 1
    device_rank: int = 0
    device: str | DeviceType = DEFAULT_DEVICE_TYPE

    def __enter__(self) -> DDPManager:
        if torch.cuda.is_available():
            try:
                init_process_group(backend="nccl")
                world_size = get_world_size()
                device_rank = get_rank()
                device: str = f"cuda:{device_rank}"
                torch.cuda.set_device(device=device)

                self.is_ddp_run = True
                self.world_size = world_size
                self.device_rank = device_rank
                self.device = device

            except (AssertionError, ValueError, RuntimeError) as exc:
                logger.info("Cannot use DDP: exc=%s", exc)

        else:
            logger.info("CUDA not detected; Skipping DDP.")

        torch.set_default_device(device=self.device)

        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if self.is_ddp_run:
            teardown_ddp()
