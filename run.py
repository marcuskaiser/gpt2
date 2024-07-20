"""Test run loading the model"""

import logging
import sys
from typing import Any

import torch
from gpt2.config import Config
from gpt2.data_loader import SimpleDataLoader
from gpt2.distributed import teardown_ddp, IS_DDP_RUN, DEVICE_RANK
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

SEQ_LENGTH = 512
NUM_TRAIN_STEPS = 5
NUM_ACCUMULATION_STEPS = 8
BATCH_SIZE = 1

if DEFAULT_DEVICE_TYPE == "cuda":
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    torch.backends.cuda.enable_math_sdp(enabled=False)

    SEQ_LENGTH = 1024
    NUM_TRAIN_STEPS = 10
    NUM_ACCUMULATION_STEPS = 16
    BATCH_SIZE = 12


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
) -> nn.Module:
    """Load model."""

    model: nn.Module
    if RANDOM:
        model = GPT(config=config.gpt_config)
    else:
        model = GPT.from_pretrained(config=config.gpt_config)

    logger.info({p.device for p in model.parameters()})
    logger.info(model)

    if IS_DDP_RUN:
        model = DistributedDataParallel(
            module=model,
            device_ids=[DEVICE_RANK],
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
) -> None:

    if not IS_DDP_RUN:

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


def _profile(
    func: callable,
    func_kwargs: dict,
    row_limit: int = 10,
    sort_by: str | None = None,
) -> Any:

    # if DEFAULT_DEVICE_TYPE != "mps":
    #     with torch.mps.profiler.profile as prof_mps:
    #         out = func(**func_kwargs)
    with torch.profiler.profile(profile_memory=True) as prof:
        out = func(**func_kwargs)

    if sort_by is None:
        sort_by = "self_cpu_time_total"
        if DEFAULT_DEVICE_TYPE == "cuda":
            sort_by = "self_cuda_time_total"
    try:
        logger.info(
            "Profiler results:\n%s",
            prof.key_averages().table(sort_by=sort_by, row_limit=row_limit),
        )
    except Exception as exc:
        logger.info("ERROR %s", exc)
    return out


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

    _eval(model=model, tokenizer=tokenizer)

    if TRAIN:
        if PROFILE:
            _profile(
                func=_train,
                func_kwargs={"config": config, "model": model},
            )
        else:
            _train(
                **{"config": config, "model": model},
            )

        _eval(model=model, tokenizer=tokenizer)

    save_model(model=model, filename="model.safetensors")

    del model
    model = GPT(config.gpt_config)
    load_model(model=model, filename="model.safetensors")

    _eval(model=model, tokenizer=tokenizer)

    if IS_DDP_RUN:
        teardown_ddp()
