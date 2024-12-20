"""Utils for GPT Model"""

import logging

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from gpt2.utils import DEFAULT_DEVICE_TYPE

logger = logging.getLogger(__name__)


def get_hf_tokenizer(
    model_id: str | None = None,
) -> PreTrainedTokenizer:
    """Get GPT2-Tokenizer."""

    if model_id is None:
        model_id = "openai-community/gpt2"

    logger.info("Loading HF tokenizer: `%s`", model_id)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
    )


def get_hf_model(
    model_id: str = "openai-community/gpt2",
    device_map: str = DEFAULT_DEVICE_TYPE,
    **kwargs,
) -> PreTrainedModel:
    """Get GPT2-Model."""
    logger.info("Loading HF model: `%s`", model_id)
    return GPT2LMHeadModel.from_pretrained(
        pretrained_model_name_or_path=model_id,
        device_map=device_map,
        **kwargs,
    )


def tokenize_string_dataset(
    text: str,
    tokenizer: PreTrainedTokenizer = get_hf_tokenizer(),
    device: str = DEFAULT_DEVICE_TYPE,
) -> torch.Tensor:
    """Tokenize a string into a torch.Tensor of tokens."""

    # TODO! We do not handle BOS/EOS tokens here! Not needed for GPT2...
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(device)

    logger.info(
        "Tokenized dataset: size=%s device=`%s`",
        input_ids.size(),
        input_ids.device,
    )

    return input_ids


def tokenize_file_from_disk(
    file_path: str,
    tokenizer: PreTrainedTokenizer = get_hf_tokenizer(),
    device: str = DEFAULT_DEVICE_TYPE,
) -> torch.Tensor:
    """Tokenize a text file from disk into a torch.Tensor of tokens."""

    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        text = fp.read()

    logger.info(
        "Loaded file=`%s` with len=%d",
        file_path,
        len(text),
    )
    return tokenize_string_dataset(
        text=text,
        tokenizer=tokenizer,
        device=device,
    )
