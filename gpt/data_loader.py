import logging

import torch

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        tokens: torch.Tensor,
    ) -> None:
        self.tokens = tokens

        assert tokens.ndim == 2
        assert tokens.size(0) == 1
        assert tokens.size(1) > 0

    def _get_sample(
        self,
        offset: int,
        bsz: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_this = slice(
            offset,
            offset + (bsz * seq_len + 1),
        )
        tokens_this = self.tokens[:, idx_this]
        return (
            tokens_this[:, :-1].view(bsz, seq_len),
            tokens_this[:, 1:].view(bsz, seq_len),
        )
