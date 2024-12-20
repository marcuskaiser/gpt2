"""Simple Data Loader class"""

import logging

import torch

from gpt2.config import Config

logger = logging.getLogger(__name__)


# TODO! Add train/test data splitting.
# TODO! Packing? Not needed for single text string, but TBC.
class SimpleDataLoader:
    """SimpleDataLoader."""

    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    def __init__(
        self,
        config: Config,
        data: torch.Tensor,
        world_size: int = 1,
        device_rank: int = 0,
    ) -> None:
        self.config = config
        assert isinstance(config, Config)
        config.resolve()

        self.data = data
        assert data.ndim == 2
        assert data.size(0) == 1
        assert data.size(1) > 0
        self._data_len = self.data.size(1)

        self.batch_size = self.config.data_config.batch_size
        assert self.batch_size > 0

        self.seq_length = self.config.data_config.seq_length
        assert self.seq_length > 0

        self.eff_batch_size_per_device = self.batch_size * self.seq_length
        assert self.eff_batch_size_per_device > 0
        self.eff_batch_size = self.eff_batch_size_per_device * world_size
        assert self.eff_batch_size > 0
        logger.info("SimpleDataLoader.eff_batch_size=%d", self.eff_batch_size)

        self._offset = self.eff_batch_size_per_device * device_rank
        self._batch_counter = 0
        self._dataset_cycles = 0

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.size(1) == self.eff_batch_size_per_device
        ), f"{x.size(1)} != {self.eff_batch_size_per_device}"
        return x.view(self.batch_size, self.seq_length)

    def _get_token_slice(self, offset: int) -> torch.Tensor:
        idx_this = slice(
            offset,
            offset + (self.eff_batch_size_per_device + 1),
        )
        tokens_this = self.data[:, idx_this]
        assert (
            tokens_this.size(1) == self.eff_batch_size_per_device + 1
        ), f"{tokens_this.size(1)} != {self.eff_batch_size_per_device + 1}"
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
            self._offset + self.eff_batch_size_per_device,
            self._data_len,
        )

        x, y = self._extract_one_training_batch(offset=self._offset)

        self._offset += self.eff_batch_size
        # TODO! We are truncating part of the data. Can we fix this?
        if self._offset + self.eff_batch_size >= self._data_len:
            logger.info(
                "End of epoch %3d: Resetting offset.",
                self._dataset_cycles,
            )
            self._offset = 0
            self._dataset_cycles += 1

        return x, y
