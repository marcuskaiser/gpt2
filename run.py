"""Test run loading the model"""

import logging

import torch

from gpt.hf_utils import get_hf_tokenizer, tokenize_file_from_disk
from gpt.models.gpt2 import GPT, GPTConfig
from gpt.data_loader import SimpleDataLoader
from gpt.trainer import SimpleTrainer
from gpt.utils import DEFAULT_DEVICE, empty_cache

RANDOM = True
TRAIN = True

LR = 3e-4
NUM_TRAIN_STEPS = 10
NUM_ACCUMULATION_STEPS = 16
BATCH_SIZE = 1
SEQ_LENGTH = 1024


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    empty_cache()

    # TODO!
    torch.set_float32_matmul_precision("high")

    tokenizer = get_hf_tokenizer()
    tokens = tokenizer(
        "Hi, my name is Bob and",
        return_tensors="pt",
    ).to(DEFAULT_DEVICE)
    x_eval = tokens["input_ids"]

    def _eval():
        model.eval()
        output_tokens = model.generate(x_eval, max_new_tokens=30)
        print(output_tokens)
        print(tokenizer.decode(output_tokens[0]).replace("\n", "\\n"))

    if RANDOM:
        config = GPTConfig()
        model = GPT(config=config)
    else:
        model = GPT.from_pretrained()

    # TODO!
    # model = torch.compile(model)

    print(model.config)

    _eval()

    if TRAIN:
        tokens = tokenize_file_from_disk("data/input.txt")

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
