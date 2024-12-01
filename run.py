"""Test run loading the model"""

import logging
import sys
from typing import Any

import torch
from gpt2.config import Config
from gpt2.data_loader import SimpleDataLoader
from gpt2.distributed import DDPManager
from gpt2.hf_utils import get_hf_tokenizer, tokenize_file_from_disk
from gpt2.models.gpt2 import GPT
from gpt2.trainer import SimpleTrainer
from gpt2.utils import DEFAULT_DEVICE_TYPE, empty_cache, set_seed
from safetensors.torch import load_model, save_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel


logger = logging.getLogger(__name__)

RANDOM = False
TRAIN = True
COMPILE_MODEL = False
PROFILE = False

LR = 6e-4

if DEFAULT_DEVICE_TYPE == "cuda":
    SEQ_LENGTH = 1024
    NUM_TRAIN_STEPS = 10
    NUM_ACCUMULATION_STEPS = 16
    BATCH_SIZE = 12
else:
    SEQ_LENGTH = 512
    NUM_TRAIN_STEPS = 5
    NUM_ACCUMULATION_STEPS = 8
    BATCH_SIZE = 1


def _get_config() -> Config:
    """Get config."""

    config = Config()
    config.data_config.batch_size = BATCH_SIZE
    config.data_config.seq_length = SEQ_LENGTH
    config.training_config.lr = LR
    config.training_config.num_accumulation_steps = NUM_ACCUMULATION_STEPS
    config.resolve()

    logger.info("config=%s", config.model_dump_json())
    return config


def _load_model(
    config: Config,
    ddp_manager: DDPManager,
) -> nn.Module:
    """Load model."""

    model: nn.Module
    if RANDOM:
        model = GPT(config=config.gpt_config)
    else:
        model = GPT.from_pretrained(config=config.gpt_config)

    logger.info({p.device for p in model.parameters()})
    logger.info(model)

    if ddp_manager.is_ddp_run:
        model = DistributedDataParallel(
            module=model,
            device_ids=[ddp_manager.device_rank],
        )

    if not hasattr(model, "config"):
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
    ddp_manager: DDPManager,
) -> None:
    """Train model."""

    data: torch.Tensor = tokenize_file_from_disk(
        file_path="data/input.txt",
    ).to(device=DEFAULT_DEVICE_TYPE)

    data_loader = SimpleDataLoader(
        config=config,
        data=data,
        world_size=ddp_manager.world_size,
        device_rank=ddp_manager.device_rank,
    )

    trainer = SimpleTrainer(
        config=config,
        model=model,
        data_loader=data_loader,
    )
    trainer.train_model(
        num_train_steps=NUM_TRAIN_STEPS,
    )


def _eval(
    model: nn.Module,
    tokenizer: Any,
    ddp_manager: DDPManager,
) -> None:

    if not ddp_manager.is_ddp_run:

        try:
            tokens = tokenizer("Hi, my", return_tensors="pt")
            x_eval = tokens["input_ids"].to(DEFAULT_DEVICE_TYPE)

            model.eval()
            output_tokens = model.generate(x_eval, max_new_tokens=30)
            logger.info(">> %s", output_tokens)
            output_message = tokenizer.decode(token_ids=output_tokens[0])
            output_message = output_message.replace("\n", "\\n")
            logger.info(">> %s", output_message)
        except Exception as exc:
            logger.info("ERROR %s", exc)


if __name__ == "__main__":

    for hdlr in logging.root.handlers[:]:
        logging.root.removeHandler(hdlr=hdlr)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # Torch config:
    torch.set_float32_matmul_precision(precision="high")
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    torch.backends.cuda.enable_math_sdp(enabled=False)
    empty_cache()
    set_seed(seed=0)

    # loca config, tokenizer and model:

    with DDPManager() as ddp_manager:

        config = _get_config()
        tokenizer = get_hf_tokenizer()

        model = _load_model(config=config, ddp_manager=ddp_manager)
        _eval(model=model, tokenizer=tokenizer, ddp_manager=ddp_manager)

        if TRAIN:
            _train(config=config, model=model, ddp_manager=ddp_manager)
            _eval(model=model, tokenizer=tokenizer, ddp_manager=ddp_manager)

        if ddp_manager.is_main_process:
            save_model(model=model, filename="model.safetensors")

            del model

            model = GPT(config.gpt_config)
            load_model(model=model, filename="model.safetensors")

            _eval(model=model, tokenizer=tokenizer, ddp_manager=ddp_manager)
