"""
Definition of GPT model.
Based on Karpathy's https://www.youtube.com/watch?v=l8pRSuU81PU
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from gpt.utils import (
    DEFAULT_DEVICE_TYPE,
    TYPE_DEVICE_TYPE,
    TYPE_DTYPE,
    copy_model_weights,
)

logger = logging.getLogger(__name__)


class GPTConfig(BaseModel):
    """Base class"""

    block_size: int = 1024  # Max sequence length.
    vocab_size: int = 50257  # Size of vocabulary
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of heads
    n_embd: int = 768  # Latent dimension
    mlp_factor: int = 4  # multiplicative factor in MLP latent dim.
    autocast_dtype: TYPE_DTYPE = "bf16"
    device: TYPE_DEVICE_TYPE = DEFAULT_DEVICE_TYPE

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.n_embd % self.n_head == 0

    @property
    def head_dim(self):
        """Calculate head dimension."""
        return self.n_embd // self.n_head

    def get_model_kwargs(self) -> dict[str, Any]:
        """Returns model training kwargs."""
        if self.device == "mps":
            return {
                "device": self.device,
                "dtype": torch.bfloat16,
            }
        return {
            "device": self.device,
        }


# TODO! Model initialization: Are the default initialization ranges ok?


class CausalSelfAttention(nn.Module):
    """CausalSelfAttention."""

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # Creates QKV mapping:
        self.c_attn = nn.Linear(
            in_features=self.n_embd,
            out_features=3 * self.n_embd,  # QKV
            device=self.config.device,
        )
        # Out-projection
        self.c_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_embd,
            device=self.config.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        bsz, q_len, _ = orig_shape = x.size()
        view_shape = (bsz, q_len, self.n_head, self.head_dim)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        return self.c_proj(
            F.scaled_dot_product_attention(  # pylint: disable=not-callable
                query=q.view(view_shape).transpose(1, 2),
                key=k.view(*view_shape).transpose(1, 2),
                value=v.view(*view_shape).transpose(1, 2),
                is_causal=True,
            )
            .transpose(1, 2)
            .contiguous()
            .view(*orig_shape)
        )


class MLP(nn.Module):
    """MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_mlp_dim = config.mlp_factor * config.n_embd

        self.c_fc = nn.Linear(
            in_features=config.n_embd,
            out_features=self.latent_mlp_dim,
            bias=True,
            device=self.config.device,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=self.latent_mlp_dim,
            out_features=config.n_embd,
            bias=True,
            device=self.config.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.c_proj(self.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """TransformerBlock."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(
            normalized_shape=config.n_embd,
            device=self.config.device,
        )
        self.attn = CausalSelfAttention(config=config)
        self.ln_2 = nn.LayerNorm(
            normalized_shape=config.n_embd,
            device=self.config.device,
        )
        self.mlp = MLP(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Main GPT class."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # TODO! Weight initialization
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.n_embd,
                    **self.config.get_model_kwargs(),
                ),  # Token embedding layer
                "wpe": nn.Embedding(
                    num_embeddings=config.block_size,
                    embedding_dim=config.n_embd,
                    **self.config.get_model_kwargs(),
                ),  # Positional embedding layer
                "h": nn.ModuleList(
                    [
                        TransformerBlock(config=config)
                        for _ in range(config.n_layer)
                    ]
                ),  # Transformer blocks
                "ln_f": nn.LayerNorm(
                    normalized_shape=config.n_embd,
                    **self.config.get_model_kwargs(),
                ),  # Final layer norm
            }
        )
        self.lm_head = nn.Linear(
            in_features=config.n_embd,
            out_features=config.vocab_size,
            bias=False,
            **self.config.get_model_kwargs(),
        )

        # weight sharing: initial token
        #  embedding weight == final lm head weight:
        self.transformer.wte.weight = self.lm_head.weight

        self.to(**self.config.get_model_kwargs())

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward method returning both the logits and optionally the loss
        if a target variable y is present.
        """
        assert x.ndim == 2
        assert x.shape[1] <= self.config.block_size

        pos = torch.arange(
            start=0,
            end=x.shape[1],
            dtype=torch.long,
            device=x.device,
        )
        x = self.transformer.wpe(pos) + self.transformer.wte(x)
        for block in self.transformer.h:
            x = block(x)

        logits = self.lm_head(self.transformer.ln_f(x))

        loss = None
        if y is not None:

            logits_this = logits.view(-1, logits.size(-1))
            y_this = y.view(-1)

            loss = F.cross_entropy(
                input=logits_this,
                target=y_this,
            )

        return logits, loss

    def generate(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int = 10,
    ) -> torch.Tensor:
        """Generate a reply."""
        with torch.no_grad():
            tokens = input_tokens
            for _ in range(max_new_tokens):
                output, _ = self(tokens)
                output = output[:, -1, :].argmax(axis=1, keepdim=True)
                tokens = torch.concat([tokens, output], dim=1)

        return tokens

    @classmethod
    def from_pretrained(
        cls,
        **kwargs,
    ) -> GPT:
        """Naive method to load model from pretrained."""
        from gpt.hf_utils import get_hf_model

        config = GPTConfig(**kwargs)
        model = GPT(config=config)
        model_hf = get_hf_model()

        copy_model_weights(
            input_model=model_hf,
            target_model=model,
        )
        return model
