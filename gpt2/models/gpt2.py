"""
Definition of GPT model.
Based on Karpathy's https://www.youtube.com/watch?v=l8pRSuU81PU
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from gpt2.config import GPTConfig
from gpt2.hf_utils import get_hf_model
from gpt2.utils import copy_model_weights

logger = logging.getLogger(__name__)

# TODO! Model weight initialization: Are the default initialization ranges ok?


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
                query=q.view(*view_shape).transpose(1, 2),
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
        self.gpt_config = config

        self.transformer = nn.ModuleDict(
            modules={
                "wte": nn.Embedding(
                    num_embeddings=self.gpt_config.vocab_size,
                    embedding_dim=self.gpt_config.n_embd,
                    **self.gpt_config.get_model_kwargs(),
                ),  # Token embedding layer
                "wpe": nn.Embedding(
                    num_embeddings=self.gpt_config.block_size,
                    embedding_dim=self.gpt_config.n_embd,
                    **self.gpt_config.get_model_kwargs(),
                ),  # Positional embedding layer
                "h": nn.ModuleList(
                    modules=[
                        TransformerBlock(config=self.gpt_config)
                        for _ in range(self.gpt_config.n_layer)
                    ]
                ),  # Transformer blocks
                "ln_f": nn.LayerNorm(
                    normalized_shape=self.gpt_config.n_embd,
                    **self.gpt_config.get_model_kwargs(),
                ),  # Final layer norm
            }
        )
        self.lm_head = nn.Linear(
            in_features=self.gpt_config.n_embd,
            out_features=self.gpt_config.vocab_size,
            bias=False,
            **self.gpt_config.get_model_kwargs(),
        )

        # weight sharing: initial token
        #  embedding weight == final lm head weight:
        self.transformer.wte.weight = self.lm_head.weight

        self.to(**self.gpt_config.get_model_kwargs())

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward method returning both the logits and optionally the loss
        if a target variable labels is present.
        """
        assert x.ndim == 2
        assert x.shape[1] <= self.gpt_config.block_size

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
        if labels is not None:
            loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=labels.view(-1),
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
        config: GPTConfig,
    ) -> GPT:
        """Naive method to load model from pretrained."""

        model = GPT(config=config)
        model_hf: PreTrainedModel = get_hf_model()

        copy_model_weights(
            input_model=model_hf,
            target_model=model,
        )
        return model
