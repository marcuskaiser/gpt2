"""Test run loading the model"""

import logging

from gpt.models.gpt2 import GPT, GPTConfig
from gpt.utils import empty_cache, get_device_type, get_hf_tokenizer

DEVICE = get_device_type()

RANDOM = False

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    empty_cache()
    tokenizer = get_hf_tokenizer()

    if RANDOM:
        config = GPTConfig()
        model = GPT(config=config)
    else:
        model = GPT.from_pretrained().eval()

    tokens = tokenizer(
        "Hi, my name is Bob and",
        return_tensors="pt",
    ).to(DEVICE)
    x = tokens["input_ids"]
    output_tokens = model.generate(x, max_new_tokens=30)

    print(output_tokens)
    print(tokenizer.decode(output_tokens[0]).replace("\n", "\\n"))
