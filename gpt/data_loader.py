import logging

import torch

logger = logging.getLogger(__name__)


# TODO! Allow to wrap when end of dataset is reached.
class SimpleDataLoader:
    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> None:
        self.data = data
        assert data.ndim == 2
        assert data.size(0) == 1
        assert data.size(1) > 0
        self._data_len = self.data.size(1)

        self.batch_size = batch_size
        assert self.batch_size > 0

        self.seq_len = seq_len
        assert self.seq_len > 0

        self.eff_batch_size = self.batch_size * self.seq_len

        self._offset = 0
        self._batch_counter = 0
        self._dataset_cycles = 0

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.size(1) == self.eff_batch_size
        ), f"{x.size(1)} != {self.eff_batch_size}"
        return x.view(self.batch_size, self.seq_len)

    def _get_token_slice(self, offset: int) -> torch.Tensor:
        idx_this = slice(
            offset,
            offset + (self.eff_batch_size + 1),
        )
        tokens_this = self.data[:, idx_this]
        assert (
            tokens_this.size(1) == self.eff_batch_size + 1
        ), f"{tokens_this.size(1)} != {self.eff_batch_size + 1}"
        return tokens_this

    def _extract_one_training_batch(
        self,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        tokens_this = self._get_token_slice(offset=offset)
        self._batch_counter += 1
        return (
            self._reshape(tokens_this[:, :-1]),
            self._reshape(tokens_this[:, 1:]),
        )

    def get_next_training_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Creates the next mini-batch for model training."""

        logger.debug(
            "Extracting training batch %d:%d (total length %d)",
            self._offset,
            self._offset + self.eff_batch_size,
            self._data_len,
        )

        x, y = self._extract_one_training_batch(offset=self._offset)

        self._offset += self.eff_batch_size
        if (
            self._offset + self.eff_batch_size + 1 >= self._data_len
        ):  # TODO! Check logic
            self._offset = 0
            self._dataset_cycles += 1

        return x, y
