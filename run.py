"""Test run loading the model"""

import logging

import torch

from gpt.hf_utils import get_hf_tokenizer, tokenize_file_from_disk
from gpt.models.gpt2 import GPT, GPTConfig
from gpt.data_loader import SimpleDataLoader
from gpt.trainer import SimpleTrainer
from gpt.utils import DEFAULT_DEVICE, empty_cache, set_seed

logger = logging.getLogger(__name__)


RANDOM = False
TRAIN = True

LR = 3e-4
NUM_TRAIN_STEPS = 10
NUM_ACCUMULATION_STEPS = 16
BATCH_SIZE = 1
SEQ_LENGTH = 1024


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    empty_cache()
    set_seed(0)

    # TODO!
    torch.set_float32_matmul_precision("high")
    torch.set_default_device(DEFAULT_DEVICE)

    tokenizer = get_hf_tokenizer()
    tokens = tokenizer(
        "Hi, my name is Bob and",
        return_tensors="pt",
    )
    x_eval = tokens["input_ids"].to(DEFAULT_DEVICE)

    def _eval():
        model.eval()
        output_tokens = model.generate(x_eval, max_new_tokens=30)
        print(output_tokens)
        print(tokenizer.decode(output_tokens[0]).replace("\n", "\\n"))

    model_kwargs = {
        "goldfish_p": 0.75,
    }

    if RANDOM:
        config = GPTConfig(**model_kwargs)
        model = GPT(config=config)
    else:
        model = GPT.from_pretrained(**model_kwargs)

    print({p.device for p in model.parameters()})
    print(model)

    # TODO!
    # try:
    #     import torch._dynamo

    #     torch._dynamo.config.suppress_errors = True
    #     model = torch.compile(model)
    #     logger.info("model.compile successful!")

    # except AssertionError as exc:
    #     logger.info("model.compile failed: %s", exc)

    print(model.config)

    _eval()

    if TRAIN:
        tokens = tokenize_file_from_disk("data/input.txt").to(DEFAULT_DEVICE)

        model = model.train()

        data_loader = SimpleDataLoader(
            data=tokens,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LENGTH,
        )

        trainer = SimpleTrainer(
            model=model,
            data_loader=data_loader,
            lr=LR,
            num_accumulation_steps=NUM_ACCUMULATION_STEPS,
        ).train_model(
            num_train_steps=NUM_TRAIN_STEPS,
        )

        _eval()
