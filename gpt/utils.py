"""Utils for GPT Model"""

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


def get_device_type() -> str:
    """Extract device type."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_torch_dtype() -> str:
    """Get preferred torch.dtype."""
    device_type = get_device_type()
    if device_type in ["cuda", "mps"]:
        return "bf16"
    return "ft32"


def empty_cache() -> None:
    """Empty cache."""
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


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
