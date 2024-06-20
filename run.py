"""Test run loading the model"""

import logging
import sys

import torch

from gpt.hf_utils import get_hf_tokenizer, tokenize_file_from_disk
from gpt.models.gpt2 import GPT, GPTConfig
from gpt.data_loader import SimpleDataLoader
from gpt.trainer import SimpleTrainer, TrainingConfig
from gpt.utils import DEFAULT_DEVICE, empty_cache, set_seed

logger = logging.getLogger(__name__)


RANDOM = False
TRAIN = True
COMPILE_MODEL = False

LR = 3e-4
NUM_TRAIN_STEPS = 10
NUM_ACCUMULATION_STEPS = 4
BATCH_SIZE = 1
SEQ_LENGTH = 1024


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        # format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
    )

    empty_cache()
    set_seed(0)

    # TODO!
    torch.set_float32_matmul_precision("high")
    torch.set_default_device(DEFAULT_DEVICE)

    tokenizer = get_hf_tokenizer()
    tokens = tokenizer(
        "Hi, my",
        return_tensors="pt",
    )
    x_eval = tokens["input_ids"].to(DEFAULT_DEVICE)

    def _eval():
        model.eval()
        output_tokens = model.generate(x_eval, max_new_tokens=30)
        logger.info(">> %s", output_tokens)
        logger.info(
            ">> %s",
            tokenizer.decode(output_tokens[0]).replace("\n", "\\n"),
        )

    model_kwargs = {}

    if RANDOM:
        config = GPTConfig(**model_kwargs)
        model = GPT(config=config)
    else:
        model = GPT.from_pretrained(**model_kwargs)

    logger.info({p.device for p in model.parameters()})
    logger.info(model)

    if COMPILE_MODEL:
        try:
            model = torch.compile(model)
            logger.info("model.compile successful!")
        except AssertionError as exc:
            logger.info("model.compile failed: %s", exc)

    logger.info("%s", model.config)

    _eval()

    if TRAIN:
        tokens = tokenize_file_from_disk("data/input.txt").to(DEFAULT_DEVICE)

        model = model.train()

        data_loader = SimpleDataLoader(
            data=tokens,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LENGTH,
        )

        trainer_config = TrainingConfig(
            lr=LR,
            num_accumulation_steps=NUM_ACCUMULATION_STEPS,
            use_scaler=False,
        )

        trainer = SimpleTrainer(
            config=trainer_config,
            model=model,
            data_loader=data_loader,
        ).train_model(
            num_train_steps=NUM_TRAIN_STEPS,
        )

        _eval()
