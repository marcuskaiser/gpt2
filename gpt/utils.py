"""Utils for GPT Model"""

import logging
from typing import Literal, cast

import torch
from torch import nn


logger = logging.getLogger(__name__)

T_DEVICE = Literal["cpu", "cuda", "mps"]
T_DTYPE = Literal["fp32", "fp16", "bf16"]


def get_device() -> T_DEVICE:
    """Extract device type."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    logger.info("Selected default_device=%s", device)
    return cast(T_DEVICE, device)


DTYPE_MAP: dict[T_DTYPE, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def empty_cache() -> None:
    """Empty cache."""
    if DEFAULT_DEVICE == "cuda":
        logger.info("Requested cuda.empty_cache")
        torch.cuda.empty_cache()
    elif DEFAULT_DEVICE == "mps":
        logger.info("Requested mps.empty_cache")
        torch.mps.empty_cache()


def set_seed(seed: int):
    """Set random seed."""

    torch.manual_seed(seed)
    for i, default_generator in enumerate(torch.cuda.default_generators):
        default_generator.manual_seed(seed + i)


def copy_model_weights(
    input_model: nn.Module,
    target_model: nn.Module,
    transpose_keys: list[str] | None = None,
) -> bool:
    """Copy model weights from one model to another."""

    if transpose_keys is None:
        transpose_keys = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

    target_model_state_dict = target_model.state_dict()
    input_model_state_dict = input_model.state_dict()

    set_diff = set(input_model_state_dict) - set(target_model_state_dict)
    assert len(set_diff) == 0, set_diff

    for key, val_input in input_model_state_dict.items():
        if key not in target_model_state_dict:
            continue
        if target_model_state_dict[key].shape not in (
            val_input.shape,
            val_input.T.shape,
        ):
            return False

    for key, val_input in input_model_state_dict.items():
        if key not in target_model_state_dict:
            continue
        if (target_model_state_dict[key].shape != val_input.shape) or any(
            key.endswith(expr) for expr in transpose_keys
        ):
            val_input = val_input.T
        with torch.no_grad():
            target_model_state_dict[key].copy_(val_input)

    logger.info("Model weights copied.")
    return True


def _check_model_copied(
    input_model: nn.Module,
    target_model: nn.Module,
) -> None:
    input_model_state_dict = input_model.state_dict()
    for key, value_input in target_model.state_dict().items():
        value_target = input_model_state_dict[key]
        if value_target.shape == value_input.shape:
            pass
        elif value_target.shape == value_input.T.shape:
            value_input = value_input.T
        else:
            assert False, (
                f"Non-compatible shapes. Got {key}: "
                f"value_input={value_input.shape}, "
                f"value_target={value_target.shape}!"
            )

        max_diff = (value_input - value_target).abs().max()
        assert max_diff < 1e-12, f"Found {key} with max_diff={max_diff}!"

    logger.info("Model weights copy check successful.")


DEFAULT_DEVICE = get_device()
