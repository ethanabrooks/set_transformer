from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import filter_torchrl_warnings

filter_torchrl_warnings()


from torchrl.data import ReplayBuffer  # noqa: E402
from torchrl.data.replay_buffers import LazyMemmapStorage  # noqa: E402


def create_replay_buffer(directory: Path, size: int, index: Optional[int]):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = directory / now
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / f"{index}.memmap"
    return ReplayBuffer(LazyMemmapStorage(size, scratch_dir=path))
