"""Test run loading the model"""

import logging
import os
import sys

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

from gpt.hf_utils import get_hf_tokenizer, tokenize_file_from_disk
from gpt.models.gpt2 import GPT, GPTConfig
from gpt.data_loader import SimpleDataLoader
from gpt.trainer import SimpleTrainer, TrainingConfig
from gpt.utils import DEFAULT_DEVICE_TYPE, empty_cache, set_seed


logger = logging.getLogger(__name__)


RANDOM = False
TRAIN = True
COMPILE_MODEL = False

LR = 6e-4
SEQ_LENGTH = 1024


IS_DDP_RUN = "RANK" in os.environ
DEVICE_RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

if IS_DDP_RUN:
    assert torch.cuda.is_available()
    torch.cuda.set_device(f"cuda:{DEVICE_RANK}")
    init_process_group(backend="nccl")

if DEFAULT_DEVICE_TYPE == "cuda":
    NUM_TRAIN_STEPS = 10
    NUM_ACCUMULATION_STEPS = 16
    BATCH_SIZE = 12
    OPTIMIZER = "adamw"
else:
    NUM_TRAIN_STEPS = 5
    NUM_ACCUMULATION_STEPS = 4
    BATCH_SIZE = 1
    OPTIMIZER = "adamw8bit"


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        # format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
    )

    logger.info(
        "effective_batch_size=%d",
        SEQ_LENGTH * NUM_ACCUMULATION_STEPS * BATCH_SIZE,
    )

    empty_cache()
    set_seed(0)

    torch.set_float32_matmul_precision("high")
    # TODO! Check regarding multiple cuda devices!
    torch.set_default_device(DEFAULT_DEVICE_TYPE)

    tokenizer = get_hf_tokenizer()
    tokens = tokenizer(
        "Hi, my",
        return_tensors="pt",
    )
    x_eval = tokens["input_ids"].to(DEFAULT_DEVICE_TYPE)

    def _eval():
        model.eval()
        output_tokens = model.generate(x_eval, max_new_tokens=30)
        logger.info(">> %s", output_tokens)
        logger.info(
            ">> %s",
            tokenizer.decode(output_tokens[0]).replace("\n", "\\n"),
        )

    model_kwargs = {}
    if DEFAULT_DEVICE_TYPE == "cuda":
        model_kwargs = {
            "autocast_dtype": "fp16",
        }

    if RANDOM:
        config = GPTConfig(**model_kwargs)
        model = GPT(config=config)
    else:
        model = GPT.from_pretrained(**model_kwargs)

    logger.info({p.device for p in model.parameters()})
    logger.info(model)

    if IS_DDP_RUN:
        config = model.config
        model = DistributedDataParallel(model, device_ids=[DEVICE_RANK])
        model.config = config

    if COMPILE_MODEL:
        try:
            model = torch.compile(model)
            logger.info("model.compile successful!")
        except AssertionError as exc:
            logger.info("model.compile failed: %s", exc)

    logger.info("%s", model.config)

    if not IS_DDP_RUN:
        _eval()

    if TRAIN:
        tokens: torch.Tensor = tokenize_file_from_disk(
            file_path="data/input.txt"
        ).to(device=DEFAULT_DEVICE_TYPE)

        data_loader = SimpleDataLoader(
            data=tokens,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LENGTH,
            world_size=WORLD_SIZE,
            device_rank=DEVICE_RANK,
        )

        trainer_config = TrainingConfig(
            lr=LR,
            num_accumulation_steps=NUM_ACCUMULATION_STEPS,
            use_scaler=model.config.autocast_dtype == "fp16",
            optmizer=OPTIMIZER,
        )

        trainer = SimpleTrainer(
            config=trainer_config,
            model=model,
            data_loader=data_loader,
        ).train_model(
            num_train_steps=NUM_TRAIN_STEPS,
        )

        if not IS_DDP_RUN:
            _eval()

    if IS_DDP_RUN:
        destroy_process_group()
