import logging
from typing import Any

import torch
from torch.distributed import get_world_size, get_rank
from gpt.utils import (
    DEFAULT_DEVICE_TYPE,
    TYPE_DEVICE_TYPE,
    TYPE_DTYPE,
    TYPE_OPTIMIZER,
)
from pydantic import BaseModel

logger = logging.getLogger(name=__name__)


class GPTConfig(BaseModel):
    """GPT config."""

    block_size: int = 1024  # Max sequence length.
    vocab_size: int = 50257  # Size of vocabulary
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of heads
    n_embd: int = 768  # Latent dimension
    mlp_factor: int = 4  # multiplicative factor in MLP latent dim.
    autocast_dtype: TYPE_DTYPE = "bf16"
    device: TYPE_DEVICE_TYPE = DEFAULT_DEVICE_TYPE

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.n_embd % self.n_head == 0

    @property
    def head_dim(self) -> int:
        """Calculate head dimension."""
        return self.n_embd // self.n_head

    def get_model_kwargs(self) -> dict[str, Any]:
        """Returns model training kwargs."""
        if self.device == "mps":
            return {
                "device": self.device,
                "dtype": torch.bfloat16,
            }
        return {
            "device": self.device,
        }


class TrainingConfig(BaseModel):
    """Training config."""

    lr: float = 3e-4
    num_accumulation_steps: int = 1
    use_scaler: bool = True
    optimizer: TYPE_OPTIMIZER = "adamw"


class DataConfig(BaseModel):
    """Data config."""

    batch_size: int = 1
    seq_length: int = 1024


class DDPConfig(BaseModel):
    """DDP Config."""

    is_ddp_run: bool = False
    device_rank: int = 0
    world_size: int = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._retrieve_setup()

    def _retrieve_setup(self) -> None:
        try:
            assert torch.cuda.is_available()
            self.device_rank = get_rank()
            self.world_size = get_world_size()

        except (AssertionError, ValueError) as exc:
            logger.info("exc=%s", exc)
            self.is_ddp_run = False


class Config(BaseModel):
    """Config."""

    gpt_config: GPTConfig = GPTConfig()
    training_config: TrainingConfig = TrainingConfig()
    data_config: DataConfig = DataConfig()
    ddp_config: DDPConfig = DDPConfig()
