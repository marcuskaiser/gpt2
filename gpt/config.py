import logging
from typing import Any

import torch
from gpt.distributed import WORLD_SIZE, DEVICE_RANK, IS_DDP_RUN
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

    optimizer: TYPE_OPTIMIZER = "adamw"
    lr: float = 3e-4
    num_accumulation_steps: int = 1
    use_scaler: bool = True
    use_zero: bool = False


class DataConfig(BaseModel):
    """Data config."""

    batch_size: int = 1
    seq_length: int = 1024


class DDPConfig(BaseModel):
    """DDP Config."""

    is_ddp_run: bool = IS_DDP_RUN
    device_rank: int = DEVICE_RANK
    world_size: int = WORLD_SIZE


class Config(BaseModel):
    """Config."""

    gpt_config: GPTConfig = GPTConfig()
    training_config: TrainingConfig = TrainingConfig()
    data_config: DataConfig = DataConfig()
    ddp_config: DDPConfig = DDPConfig()
