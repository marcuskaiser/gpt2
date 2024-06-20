"""Utils for GPT Model"""

import logging

import torch
from gpt.utils import DEFAULT_DEVICE
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
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
    )

    logger.info("Loaded tokenizer: `%s`", model_id)
    return tokenizer


def get_hf_model(
    model_id: str = "gpt2",
    device_map: str = DEFAULT_DEVICE,
    **kwargs,
) -> PreTrainedModel:
    """Get GPT2-Model."""
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model_name_or_path=model_id,
        device_map=device_map,
        **kwargs,
    )

    logger.info("Loaded model: `%s`", model_id)
    return model


def tokenize_string_dataset(
    text: str,
    tokenizer: PreTrainedTokenizer = get_hf_tokenizer(),
    device: str = DEFAULT_DEVICE,
) -> torch.Tensor:

    # TODO! We do not handle BOS/EOS tokens here!
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
    device: str = DEFAULT_DEVICE,
) -> torch.Tensor:

    with open(file_path, "r", encoding="utf-8") as fp:
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
