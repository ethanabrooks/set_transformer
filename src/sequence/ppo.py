from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from gymnasium.spaces import Discrete

from ppo.data_storage import DataStorage
from ppo.train import train
from sequence.base import Sequence as BaseSequence
from utils.dataclasses import Transition


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def make(
        cls,
        gamma: float,
        num_processes: int,
        num_steps: int,
        num_updates: int,
        trajectory_length: int,
        **kwargs,
    ):
        assert (num_processes * num_steps * num_updates) % trajectory_length == 0, (
            f"num_processes * num_steps * num_updates ({num_processes} * {num_steps} * {num_updates}) "
            f"must be divisible by sequence_length ({trajectory_length})"
        )
        data_storage: DataStorage
        df: pd.DataFrame

        data_storage, df = train(
            gamma=gamma,
            **kwargs,
            load_path=None,
            num_processes=num_processes,
            num_steps=num_steps,
            num_updates=num_updates,
            run=None,
        )
        if wandb.run is not None:
            plt.figure()
            image = wandb.Image(
                sns.lineplot(df, x="update", y="reward", hue=None).figure
            )
            table = wandb.Table(data=df)
            wandb.run.log({"ppo/plot": image, "ppo/table": table})

        def preprocess():
            v: np.ndarray
            for k, v in asdict(data_storage.to_transition()).items():
                v = v.swapaxes(0, 1)
                _, _, *shape = v.shape
                reshape = v.reshape(-1, trajectory_length, *shape)
                tensor = torch.from_numpy(reshape)
                yield k, tensor

        transitions = Transition(**dict(preprocess()))
        assert isinstance(data_storage.action_space, Discrete)
        pad_value = data_storage.action_space.n
        if isinstance(data_storage.observation_space, Discrete):
            pad_value += data_storage.observation_space.n

        return Sequence(
            action_space=data_storage.action_space,
            gamma=gamma,
            observation_space=data_storage.observation_space,
            pad_value=pad_value,
            transitions=transitions,
        )
