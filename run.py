"""Test run loading the model"""

import logging

from gpt.models.gpt2 import GPT, GPTConfig
from gpt.utils import (
    empty_cache,
    DEFAULT_DEVICE,
)
from gpt.hf_utils import get_hf_tokenizer


RANDOM = False
TRAIN = False

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    empty_cache()
    tokenizer = get_hf_tokenizer()

    if RANDOM:
        config = GPTConfig()
        model = GPT(config=config)
    else:
        model = GPT.from_pretrained()

    if TRAIN:
        model = model.train()
        raise NotImplementedError("Not implemented yet.")

    else:
        model = model.eval()

        tokens = tokenizer(
            "Hi, my name is Bob and",
            return_tensors="pt",
        ).to(DEFAULT_DEVICE)

        x = tokens["input_ids"]
        output_tokens = model.generate(x, max_new_tokens=30)

        print(output_tokens)
        print(tokenizer.decode(output_tokens[0]).replace("\n", "\\n"))
