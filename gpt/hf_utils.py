"""Utils for GPT Model"""

import logging

import torch
from gpt.utils import (
    DEFAULT_DEVICE,
    DEFAULT_TORCH_DTYPE,
    DTYPE_MAP,
)
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def get_hf_tokenizer(
    model_id: str = "gpt2",
) -> PreTrainedTokenizer:
    """Get GPT2-Tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Loaded tokenizer: `%s`", model_id)
    return tokenizer


def get_hf_model(
    device_map: str = DEFAULT_DEVICE,
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
    **kwargs,
) -> PreTrainedModel:
    """Get GPT2-Model."""
    return GPT2LMHeadModel.from_pretrained(
        "gpt2",
        device_map=device_map,
        torch_dtype=DTYPE_MAP[torch_dtype],
        **kwargs,
    )


def tokenizer_string_dataset(
    text: str,
    tokenizer: PreTrainedTokenizer = get_hf_tokenizer(),
    device: str = DEFAULT_DEVICE,
) -> torch.Tensor:

    tokens = tokenizer(
        text,
        return_tensors="pt",
    )
    return tokens["input_ids"].to(device)


def tokenize_file_from_disk(
    file_path: str,
    tokenizer: PreTrainedTokenizer = get_hf_tokenizer(),
    device: str = DEFAULT_DEVICE,
) -> torch.Tensor:

    with open(file_path, "r", encoding="utf-8") as fp:
        text = fp.read()

    return tokenizer_string_dataset(
        text=text,
        tokenizer=tokenizer,
        device=device,
    )
