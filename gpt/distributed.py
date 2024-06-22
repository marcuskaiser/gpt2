import logging

import torch
from torch.distributed import (
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
)
from gpt.utils import DEFAULT_DEVICE_TYPE

logger = logging.getLogger(__name__)


DEVICE_RANK = 0
WORLD_SIZE = 1
IS_DDP_RUN = False
DEVICE = DEFAULT_DEVICE_TYPE
if torch.cuda.is_available():
    try:
        init_process_group(backend="nccl")
        DEVICE_RANK = get_rank()
        DEVICE: str = f"cuda:{DEVICE_RANK}"
        WORLD_SIZE = get_world_size()
        IS_DDP_RUN = True
        torch.cuda.set_device(device=DEVICE)

    except (AssertionError, ValueError, RuntimeError) as exc:
        logger.info("Cannot use DDP: exc=%s", exc)

torch.set_default_device(device=DEVICE)


def teardown_ddp() -> None:
    """Teardown after DDP run."""
    destroy_process_group()
    logger.info("DPP Cleanup: Destroyed process group.")
