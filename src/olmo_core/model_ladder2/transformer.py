import re
from dataclasses import dataclass
from pathlib import Path

import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
import olmo_core.utils as utils
from olmo_core.config import Config, StrEnum
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import Duration, Trainer, TrainerConfig
from olmo_core.train.train_module import TrainModule, TransformerTrainModuleConfig

from .base import ModelLadderExperiment, ModelLadderRun


class TransformerSize(StrEnum):
    size_190M = "190M"
    size_370M = "370M"
    size_600M = "600M"
    size_760M = "760M"
    size_1B = "1B"
    size_3B = "3B"
    size_7B = "7B"
    size_13B = "13B"

    @property
    def num_parameters(self) -> int:
        size = self.replace(" ", "").upper()
        if (m := re.match(r"^([\d\.]+)([KMBT])$", size)) is not None:
            value, unit = m.groups()
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
            return int(float(value) * multiplier)
        else:
            raise ValueError(f"Invalid size descriptor '{self}'")


@dataclass(kw_only=True)
class TransformerLadderRunComponents(Config):
    """Defines all components required to execute a single run within transformer ladder experiment."""

    model: TransformerConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    backend: str
    seed: int


@dataclass(kw_only=True)
class TransformerLadderRun(ModelLadderRun):
    size: TransformerSize
    chinchilla_multiple: float
    sequence_length: int
    batch_size: int

    def __post_init__(self):
        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")
        if self.sequence_length <= 0:
            raise OLMoConfigurationError("'sequence_length' must be positive")
        if self.batch_size <= 0:
            raise OLMoConfigurationError("'batch_size' must be positive")
        if self.batch_size % self.sequence_length != 0:
            raise OLMoConfigurationError("'batch_size' must be a multiple of 'sequence_length'")

    @property
    def num_parameters(self) -> int:
        size = self.size.replace(" ", "").upper()
        if (m := re.match(r"^([\d\.]+)([KMBT])$", size)) is not None:
            value, unit = m.groups()
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
            return int(float(value) * multiplier)
        else:
            raise ValueError(f"Invalid size descriptor '{self.size}'")

    @property
    def duration(self) -> Duration:
        return Duration.chinchilla_tokens(
            self.chinchilla_multiple, model_params=self.num_parameters
        )

    @property
    def id(self) -> str:
        return f"{self.size}-{self.chinchilla_multiple:.2f}xC-{self.sequence_length}CL-{self.batch_size}BZ"

    def configure_model(self) -> TransformerConfig:
        pass

    def configure_train_module(self) -> TransformerTrainModuleConfig:
        pass


@dataclass(kw_only=True)
class TransformerLadderExperiment(ModelLadderExperiment[TransformerLadderRun]):
    mix: DataMix
    mix_base_dir: str = "gs://ai2-llm"
    tokenizer: TokenizerConfig
    intra_document_masking: bool = False
    instance_filter: bool = False
    backend: str = "cpu:gloo,cuda:nccl"

    def run(self, run: TransformerLadderRun, overrides: list[str] | None = None):
        components = self.configure_run_components(run, overrides)

        # Set RNG states on all devices.
        utils.seed_all(self.seed)

    def get_save_folder(self, run: TransformerLadderRun) -> str:
        return str(
            io.join_path(self.root_dir, io.make_url_safe(self.name), run.id, f"seed-{self.seed}")
        )

    def get_work_dir(self) -> str:
        return "./cache" if io.is_url(self.root_dir) else str(io.join_path(self.root_dir, "cache"))

    def configure_run_components(
        self, run: TransformerLadderRun, overrides: list[str] | None = None
    ) -> TransformerLadderRunComponents:
        if overrides:
            self = self.merge(overrides, strict=False)

        save_folder = self.get_save_folder(run)
        work_dir = self.get_work_dir()
        trainer = TrainerConfig(
            save_folder=str(save_folder),
            work_dir=str(work_dir),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=run.duration,
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
        components = TransformerLadderRunComponents(
            model=run.configure_model(),
            train_module=run.configure_train_module(),
            trainer=trainer,
            dataset=NumpyFSLDatasetConfig.from_data_mix(
                self.mix,
                mix_base_dir=self.mix_base_dir,
                tokenizer=self.tokenizer,
                work_dir=work_dir,
                sequence_length=run.sequence_length,
                generate_doc_lengths=self.intra_document_masking,
                instance_filter_config=None
                if not self.instance_filter
                else InstanceFilterConfig(
                    repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
                ),
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=run.batch_size, seed=self.seed, num_workers=4
            ),
            backend=self.backend,
            seed=self.seed,
        )
        if overrides is not None:
            components = components.merge(overrides)
        return components
