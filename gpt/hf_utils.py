"""Utils for GPT Model"""

import logging

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from gpt.utils import (
    DTYPE_MAP,
    get_device_type,
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
    device_map: str = get_device_type(),
    torch_dtype: str = "bf16",
    **kwargs,
) -> PreTrainedModel:
    """Get GPT2-Model."""
    return GPT2LMHeadModel.from_pretrained(
        "gpt2",
        device_map=device_map,
        torch_dtype=DTYPE_MAP[torch_dtype],
        **kwargs,
    )
