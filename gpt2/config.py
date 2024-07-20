"""Config classes."""

import logging
from typing import Any

import torch
from pydantic import BaseModel

from gpt2.utils import (
    DEFAULT_DEVICE_TYPE,
    DeviceType,
    TorchDtype,
    OptimizerType,
)


logger = logging.getLogger(name=__name__)


class GPTConfig(BaseModel):
    """GPT config."""

    block_size: int = 1024  # Max sequence length.
    vocab_size: int = 50257  # Size of vocabulary
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of heads
    n_embd: int = 768  # Latent dimension
    mlp_factor: int = 4  # multiplicative factor in MLP latent dim.
    autocast_dtype: TorchDtype | None = None
    device: DeviceType = DEFAULT_DEVICE_TYPE

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

    optimizer: OptimizerType | None = None
    lr: float = 3e-4
    num_accumulation_steps: int = 1
    use_scaler: bool | None = None
    use_zero: bool | None = None


class DataConfig(BaseModel):
    """Data config."""

    batch_size: int = 1
    seq_length: int = 1024


class Config(BaseModel):
    """Config."""

    gpt_config: GPTConfig = GPTConfig()
    training_config: TrainingConfig = TrainingConfig()
    data_config: DataConfig = DataConfig()

    def resolve(self) -> None:
        """Apply top-level logic to resolve None default parameters."""

        is_cuda = self.gpt_config.device == DeviceType.CUDA
        is_mps = self.gpt_config.device == DeviceType.MPS

        # check gpt config:
        if self.gpt_config.autocast_dtype is None:
            self.gpt_config.autocast_dtype = TorchDtype.FP16
            if (is_cuda and torch.cuda.is_bf16_supported()) or is_mps:
                self.gpt_config.autocast_dtype = TorchDtype.BF16
            logger.info(
                "autocast_dtype: No default. Setting %s",
                self.gpt_config.autocast_dtype,
            )

        # check training config:
        is_bf16 = self.gpt_config.autocast_dtype == TorchDtype.BF16
        is_fp16 = self.gpt_config.autocast_dtype == TorchDtype.FP16

        if is_bf16:
            if self.gpt_config.device == DeviceType.CUDA:
                assert torch.cuda.is_bf16_supported()
            else:
                assert self.gpt_config.device != DeviceType.CPU

        if self.training_config.use_scaler is None:
            self.training_config.use_scaler = False
            if is_cuda and is_fp16:
                logger.info("Scaler: No default and fp16: Setting GradScaler.")
                self.training_config.use_scaler = True
            logger.info("Scaler: No default and ~fp16: Setting NoScaler.")

        if self.training_config.use_zero is None:
            self.training_config.use_zero = is_cuda

        if self.training_config.optimizer is None:
            self.training_config.optimizer = OptimizerType.ADAMW
            if is_cuda:
                self.training_config.optimizer = OptimizerType.ADAMW_8BIT
            logger.info(
                "Optimizer: No default. Setting %s",
                self.training_config.optimizer,
            )
