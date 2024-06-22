"""Test run loading the model"""

import logging
import sys
from typing import Any

import torch
from gpt.config import Config
from gpt.data_loader import SimpleDataLoader
from gpt.hf_utils import get_hf_tokenizer, tokenize_file_from_disk
from gpt.models.gpt2 import GPT
from gpt.trainer import SimpleTrainer
from gpt.utils import (
    DEFAULT_DEVICE_TYPE,
    empty_cache,
    set_seed,
)
from gpt.distributed import teardown_ddp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)

RANDOM = False
TRAIN = True
COMPILE_MODEL = False

LR = 6e-4
SEQ_LENGTH = 1024


if DEFAULT_DEVICE_TYPE == "cuda":
    NUM_TRAIN_STEPS = 100
    NUM_ACCUMULATION_STEPS = 16
    BATCH_SIZE = 12
    OPTIMIZER = "adamw8bit"
else:
    NUM_TRAIN_STEPS = 5
    NUM_ACCUMULATION_STEPS = 4
    BATCH_SIZE = 1
    OPTIMIZER = "adamw"


def _get_config() -> Config:
    """Get config."""

    config = Config()
    config.data_config.batch_size = BATCH_SIZE
    config.data_config.seq_length = SEQ_LENGTH

    config.gpt_config.autocast_dtype = "fp16"
    config.training_config.lr = LR
    config.training_config.num_accumulation_steps = NUM_ACCUMULATION_STEPS
    config.training_config.optimizer = OPTIMIZER
    config.training_config.use_scaler = (
        config.gpt_config.autocast_dtype == "fp16"
    )

    logger.info("config=%s", config.model_dump_json())
    return config


def _load_model(
    config: Config,
) -> nn.Module:
    """Load model."""
    model_kwargs: dict[str, str] = {}
    if DEFAULT_DEVICE_TYPE == "cuda":
        model_kwargs["autocast_dtype"] = "fp16"

    model: nn.Module
    if RANDOM:
        model = GPT(config=config.gpt_config)
    else:
        model = GPT.from_pretrained(config=config.gpt_config)

    logger.info({p.device for p in model.parameters()})
    logger.info(model)

    if config.ddp_config.is_ddp_run:
        model = DistributedDataParallel(
            module=model,
            device_ids=[config.ddp_config.device_rank],
        )
        model.config = config.gpt_config

    if COMPILE_MODEL:
        try:
            model = torch.compile(model=model)
            logger.info("model.compile successful!")
        except AssertionError as exc:
            logger.info("model.compile failed: %s", exc)

    return model


def _train(
    config: Config,
    model: nn.Module,
) -> None:
    """Train model."""

    tokens: torch.Tensor = tokenize_file_from_disk(
        file_path="data/input.txt"
    ).to(device=DEFAULT_DEVICE_TYPE)

    data_loader = SimpleDataLoader(config=config, data=tokens)

    SimpleTrainer(
        config=config.training_config,
        model=model,
        data_loader=data_loader,
    ).train_model(
        num_train_steps=NUM_TRAIN_STEPS,
    )


def _eval(
    config: Config,
    model: nn.Module,
    tokenizer: Any,
) -> None:

    print("config.ddp_config.is_ddp_run", config.ddp_config.is_ddp_run)

    if not config.ddp_config.is_ddp_run:

        tokens = tokenizer(
            "Hi, my",
            return_tensors="pt",
        )
        x_eval = tokens["input_ids"].to(DEFAULT_DEVICE_TYPE)

        model.eval()
        output_tokens = model.generate(x_eval, max_new_tokens=30)
        logger.info(">> %s", output_tokens)
        output_message = tokenizer.decode(token_ids=output_tokens[0])
        output_message = output_message.replace("\n", "\\n")
        logger.info(">> %s", output_message)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    for hdlr in logging.root.handlers[:]:
        logging.root.removeHandler(hdlr=hdlr)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    logger.info(
        "effective_batch_size=%d",
        SEQ_LENGTH * NUM_ACCUMULATION_STEPS * BATCH_SIZE,
    )

    torch.set_float32_matmul_precision(precision="high")
    empty_cache()
    set_seed(seed=0)

    config = _get_config()

    tokenizer = get_hf_tokenizer()

    model = _load_model(config=config)

    _eval(config=config, model=model, tokenizer=tokenizer)
    if TRAIN:
        _train(
            config=config,
            model=model,
        )
        _eval(config=config, model=model, tokenizer=tokenizer)

    if config.ddp_config.is_ddp_run:
        teardown_ddp()
