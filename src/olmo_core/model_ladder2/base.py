import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.config import Config
from olmo_core.data import DataLoaderBase
from olmo_core.train import Duration, Trainer, TrainerConfig
from olmo_core.train.train_module import TrainModule


@dataclass
class ModelLadderRunSpec(Config):
    """
    Defines a single run in a model ladder by model size and training duration.
    """

    size_descriptor: str
    """
    Approximate model size (number of parameters), usually excluding input embeddings.
    E.g. "7B".
    """
    duration_descriptor: str
    """
    The duration to train for, e.g. "3xC".
    """

    @property
    def size(self) -> int:
        size = self.size_descriptor.replace(" ", "").upper()
        if (m := re.match(r"^([\d\.]+)([KMBT])$", size)) is not None:
            value, unit = m.groups()
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
            return int(float(value) * multiplier)
        else:
            raise ValueError(f"Invalid size descriptor '{self.size_descriptor}'")

    @property
    def duration(self) -> Duration:
        duration = self.duration_descriptor.replace(" ", "")
        if (m := re.match(r"^([\d\.]+)xC$", duration)) is not None:
            chinchilla_multiple = float(m.group(1))
            return Duration.chinchilla_tokens(chinchilla_multiple, model_params=self.size)
        else:
            raise ValueError(f"Invalid duration descriptor '{duration}'")


@dataclass
class ModelLadder(Config, metaclass=ABCMeta):
    """
    An abstract base class for defining model ladders.

    This class serves as a mapping from a :class:`ModelLadderRunSpec` to the concrete components
    needed to execute such a run, including the :class:`~olmo_core.train.train_module.TrainModule`
    and :class:`~olmo_core.data.data_loader.DataLoaderBase`.
    """

    root_dir: str

    @abstractmethod
    def build_train_module(self, run_spec: ModelLadderRunSpec) -> TrainModule:
        """
        Construct a train module for a ladder run.

        :param run_spec: The spec for the run.

        :raises ValueError: If the run spec is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def build_data_loader(self, run_spec: ModelLadderRunSpec) -> DataLoaderBase:
        raise NotImplementedError

    def build_trainer_config(self, run_spec: ModelLadderRunSpec) -> TrainerConfig:
        return TrainerConfig(
            save_folder=self.get_save_folder(run_spec),
            work_dir=str(self.get_work_dir(run_spec)),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=run_spec.duration,
            callbacks={
                "gpu_monitor": callbacks.GPUMemoryMonitorCallback(),
                "config_saver": callbacks.ConfigSaverCallback(),
                "garbage_collector": callbacks.GarbageCollectorCallback(),
                "checkpointer": callbacks.CheckpointerCallback(
                    save_interval=1_000,
                    save_async=True,
                ),
            },
        )

    def get_save_folder(self, run_spec: ModelLadderRunSpec) -> str:
        return str(
            io.join_path(
                self.root_dir, f"{run_spec.size_descriptor}-{run_spec.duration_descriptor}"
            )
        )

    def get_work_dir(self, run_spec: ModelLadderRunSpec) -> Path:
        del run_spec
        if io.is_url(self.root_dir):
            return Path("./cache")
        else:
            return Path(io.join_path(self.root_dir, "cache"))


@dataclass
class ModelLadderExperiment(Config):
    """
    Represents a complete model ladder experiment, defined by a concrete :class:`ModelLadder`
    implementation and a series of :class:`ModelLadderRunSpec` to apply the ``ModelLadder`` to.
    """

    runs_specs: list[ModelLadderRunSpec]
    model_ladder: ModelLadder
