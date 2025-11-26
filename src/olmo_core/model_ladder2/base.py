import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    WSDS,
    OptimConfig,
    OptimGroupOverride,
    Scheduler,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)

from .utils import format_tokens

M = TypeVar("M", bound=Config)


@dataclass(kw_only=True)
class ModelConfigurator(Config, metaclass=ABCMeta):
    """
    Defines how to configure a model of a particular size.
    """

    @abstractmethod
    def configure_model(self, size: str) -> TransformerConfig:
        """
        Configure the model for the given size spec.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class RunConfigurator(Config, metaclass=ABCMeta):
    """
    Defines how to configure a run for a model of particular size.
    """

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """
        A unique fingerprint for this run configuration. Used for caching run results.
        Ideally human readable.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_duration(self, num_params: int) -> Duration:
        """Get the training duration for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_batch_size(self, num_params: int) -> int:
        """Get the global batch size for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_optimizer(self, num_params: int) -> OptimConfig:
        """Get the optimizer config for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_lr_scheduler(self, num_params: int) -> Scheduler:
        """Get the learning rate scheduler for a model of this size."""
        raise NotImplementedError


@dataclass(kw_only=True)
class WSDSChinchillaRunConfigurator(RunConfigurator):
    """
    A run configurator that uses WSD-S learning rate scheduling and Chinchilla scaling laws.
    """

    VERSION: ClassVar[int] = 1

    chinchilla_multiple: float
    """
    How long to train each run for, expressed as a multiple of the Chinchilla-optimal duration
    which must be a power of 2.
    """
    decay_factor: float = 2.0
    """The duration of each decay in the WSD-S schedule as factor of the initial warmup."""

    def __post_init__(self):
        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")
        log2_cm = math.log(self.chinchilla_multiple, 2)
        if not log2_cm.is_integer():
            raise OLMoConfigurationError("'chinchilla_multiple' must be a power of 2")

    @property
    def fingerprint(self) -> str:
        return f"WSD-S_{self.chinchilla_multiple}xC_DF{self.decay_factor}_v{self.VERSION}"

    def configure_duration(self, num_params: int) -> Duration:
        return Duration.chinchilla_tokens(
            self.chinchilla_multiple,
            model_params=num_params,
        )

    def configure_batch_size(self, num_params: int) -> int:
        # Calculate global batch size according to https://api.semanticscholar.org/CorpusID:270764838
        # which assumes a sequence length of 2048.
        return round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))

    def configure_optimizer(self, num_params: int) -> SkipStepAdamWConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        # but divide by 2 for WSD schedule (seems to work emperically).
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        lr /= 2.0
        return SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(
                0.9,
                0.95,  # NOTE: paper above suggest using larger beta2 (~0.99) for small batch sizes.
            ),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        )

    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, int, list[float]]:
        # Warm up 1 token per parameter according to https://api.semanticscholar.org/CorpusID:270764838
        warmup = num_params
        decay = round(warmup * self.decay_factor)

        # Determine minimum spacing for decay periods.
        # Assume we should be in stable era in the first period for at least 80% of the period,
        # meaning warmup + decay is no more than 20% of the period.
        min_tokens_per_period = (warmup + decay) / 0.2
        # Then convert that to a Chinchilla multiple.
        min_chinchilla_period = math.ceil(
            min_tokens_per_period / Duration.chinchilla_tokens(1.0, model_params=num_params).value
        )
        if self.chinchilla_multiple < min_chinchilla_period:
            raise OLMoConfigurationError(
                f"'chinchilla_multiple={self.chinchilla_multiple}' is too small relative to warmup and decay. "
                f"You'll need a chinchilla multiple of at least {min_chinchilla_period} or a smaller decay factor."
            )

        # Generate Chinchilla (decay) periods as multiples of two, but at least the minimum.
        chinchilla_periods: list[float] = []
        max_pow = math.log(self.chinchilla_multiple, 2)
        assert max_pow.is_integer()  # checked in `__post_init__()` as well.
        for p in range(int(max_pow) + 1):
            period = 2**p
            if period >= min_chinchilla_period:
                chinchilla_periods.append(period)

        return warmup, decay, chinchilla_periods

    def configure_lr_scheduler(self, num_params: int) -> Scheduler:
        warmup, decay, chinchilla_periods = self.configure_chinchilla_periods(num_params)
        period_lengths = []
        for pidx, c in enumerate(chinchilla_periods):
            period = Duration.chinchilla_tokens(c, model_params=num_params).value
            if pidx == 0:
                period_lengths.append(period)
            else:
                period_lengths.append(
                    period
                    - Duration.chinchilla_tokens(
                        chinchilla_periods[pidx - 1], model_params=num_params
                    ).value
                )

        return WSDS(
            units=SchedulerUnits.tokens,
            warmup=warmup,
            decay=decay,
            period_lengths=period_lengths,
        )

    def plot_lr_schedule(self, num_params: int):
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore

        optim = self.configure_optimizer(num_params)
        scheduler = self.configure_lr_scheduler(num_params)
        warmup, decay, chinchilla_periods = self.configure_chinchilla_periods(num_params)
        t_max = self.configure_duration(num_params).value
        batch_size = self.configure_batch_size(num_params)
        tokens_seen = 0
        tokens = []
        lrs = []
        while tokens_seen < t_max:
            tokens_seen += batch_size
            lr = scheduler.get_lr(optim.lr, tokens_seen, t_max)
            tokens.append(tokens_seen)
            lrs.append(lr)

        df = pd.DataFrame({"tokens": tokens, "lr": lrs})
        df.plot(x="tokens", y="lr")
        plt.grid(True)

        for c in chinchilla_periods:
            if c > self.chinchilla_multiple:
                break
            period = Duration.chinchilla_tokens(c, model_params=num_params).value
            plt.axvline(x=period, color="red", linestyle="--", alpha=0.5, label=f"{c}xC")
            plt.text(
                period - 0.5 * decay,
                0.0,
                f"{c}xC",
                color="red",
                alpha=0.5,
                horizontalalignment="right",
            )

        chinchilla_tokens = Duration.chinchilla_tokens(1.0, model_params=num_params).value
        caption = f"warmup={format_tokens(warmup)}, 1xC={format_tokens(chinchilla_tokens)}, duration={format_tokens(t_max)}"
        plt.xlabel(f"Tokens\n{caption}")
        plt.title(f"Learning rate schedule out to {self.chinchilla_multiple}xC")

        plt.tight_layout()
        plt.show()


@dataclass(kw_only=True)
class ModelLadder(Config, Generic[M], metaclass=ABCMeta):
    """
    Represents a complete model ladder of runs.
    """

    name: str
    """A name to assign to the ladder."""
    sizes: list[str]
    """A list of model size specs to run as part of the ladder."""
    chinchilla_multiple: float
    """How long to train each run for, expressed as a multiple of the Chinchilla-optimal duration."""
    sequence_length: int = 4096
    """The sequence length to train each run on."""
    tokenizer: TokenizerConfig
    """The tokenizer to use."""
    mix: DataMix
    """The mix to train on."""
    mix_base_dir: str = "gs://ai2-llm"
    """The base directory for the data mix."""
    root_dir: str
    """The root directory where ladder run results should be saved."""
    seed: int = 42
    """The initial random seed to use for all runs in the ladder."""
    intra_document_masking: bool = False
    """A flag indicating whether to use intra-document masking."""
    instance_filter: bool = False
    """A flag indicating whether to use the instance filter with the data loader."""
    backend: str = "cpu:gloo,cuda:nccl"
    """The distributed backend to use for each run."""

    def __post_init__(self):
        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")

    @property
    def work_dir(self) -> PathOrStr:
        return "./cache" if io.is_url(self.root_dir) else str(io.join_path(self.root_dir, "cache"))

    @abstractmethod
    def run(self, model_spec: M, overrides: list[str] | None = None):
        """
        Execute a particular model run of the experiment locally and store the results.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics_for_run(
        self, model_spec: M, overrides: list[str] | None = None
    ) -> dict[str, float] | None:
        """
        Retrieve the final metrics for a particular model run of the experiment.
        If the experiment hasn't completed this should return ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_model(self, size: str) -> TransformerConfig:
        """
        Configure a model spec for a particular size.
        """
        raise NotImplementedError

    def configure_global_batch_size(self, num_params: int) -> int:
        # Calculate global batch size according to https://api.semanticscholar.org/CorpusID:270764838
        # which assumes a sequence length of 2048.
        return round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))

    def configure_dataset(self) -> NumpyFSLDatasetConfig:
        return NumpyFSLDatasetConfig.from_data_mix(
            self.mix,
            mix_base_dir=self.mix_base_dir,
            tokenizer=self.tokenizer,
            work_dir=str(self.work_dir),
            sequence_length=self.sequence_length,
            generate_doc_lengths=self.intra_document_masking,
            instance_filter_config=None
            if not self.instance_filter
            else InstanceFilterConfig(
                repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
            ),
        )

    def configure_data_loader(self, global_batch_size: int) -> NumpyDataLoaderConfig:
        return NumpyDataLoaderConfig(
            global_batch_size=global_batch_size, seed=self.seed, num_workers=4
        )

    def configure_trainer(self, run_id: str, num_params: int) -> TrainerConfig:
        run_name = f"{self.name}-{run_id}"
        save_folder = io.join_path(self.root_dir, io.make_url_safe(self.name), run_id)

        # Calculate training duration based on Chinchilla scaling laws.
        duration = Duration.chinchilla_tokens(
            self.chinchilla_multiple,
            model_params=num_params,
        )

        return TrainerConfig(
            save_folder=str(save_folder),
            work_dir=str(self.work_dir),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=duration,
            callbacks={
                "gpu_monitor": callbacks.GPUMemoryMonitorCallback(),
                "config_saver": callbacks.ConfigSaverCallback(),
                "garbage_collector": callbacks.GarbageCollectorCallback(),
                "checkpointer": callbacks.CheckpointerCallback(
                    save_interval=1_000,
                    save_async=True,
                ),
                "profiler": callbacks.ProfilerCallback(enabled=False),
                "gap_monitor": callbacks.GAPMonitorCallback(enabled=False),
                "slack_notifier": callbacks.SlackNotifierCallback(name=run_name, enabled=False),
                "beaker": callbacks.BeakerCallback(enabled=False),
                "wandb": callbacks.WandBCallback(
                    name=run_name,
                    group=run_name,
                    project=self.name,
                    cancel_check_interval=10,
                ),
                "downstream_evaluator": callbacks.DownstreamEvaluatorCallbackConfig(
                    tokenizer=self.tokenizer,
                    tasks=self.get_in_loop_eval_tasks(),
                    eval_interval=1_000,
                ),
                "metric_saver": callbacks.MetricSaverCallback(
                    metrics_to_capture=["train/*", "optim/*", "eval/*"]
                ),
            },
        )

    def get_in_loop_eval_tasks(self) -> list[str]:
        # For training runs where we don't expect the model to acquire MC (e.g., 1B-5xC, short 7B training runs).
        tasks_small_compute = [
            # OLMES Core 9(-ish) RC
            "arc_challenge_test_rc_5shot",
            "arc_easy_test_rc_5shot",
            "hellaswag_rc_5shot",  # 1K subset of HellaSwag
            "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
            "csqa_val_rc_5shot",
            "piqa_val_rc_5shot",
            "socialiqa_val_rc_5shot",
            # Too noisy to be worth tracking
            # "boolq_val_rc_5shot",
            # "openbookqa_test_rc_5shot",
            # MMLU RC
            "mmlu_stem_val_rc_5shot",
            "mmlu_humanities_val_rc_5shot",
            "mmlu_social_sciences_val_rc_5shot",
            "mmlu_other_val_rc_5shot",
            "mmlu_stem_test_rc_5shot",
            "mmlu_humanities_test_rc_5shot",
            "mmlu_social_sciences_test_rc_5shot",
            "mmlu_other_test_rc_5shot",
            # Gen tasks BPB
            "gsm8k_gold_bpb_5shot",
            "minerva_math_algebra_gold_bpb_0shot",
            "minerva_math_counting_and_probability_gold_bpb_0shot",
            "minerva_math_geometry_gold_bpb_0shot",
            "minerva_math_intermediate_algebra_gold_bpb_0shot",
            "minerva_math_number_theory_gold_bpb_0shot",
            "minerva_math_prealgebra_gold_bpb_0shot",
            "minerva_math_precalculus_gold_bpb_0shot",
            "codex_humaneval_gold_bpb_0shot",
            "codex_mbpp_gold_bpb_0shot",
            # Sanity check for MCQA ability
            "copycolors_10way",
        ]

        # For training runs where we expect the model to acquire MC
        tasks_large_compute = [
            # OLMES Core 9(-ish) MC
            "arc_challenge_test_mc_5shot",
            "arc_easy_test_mc_5shot",
            "hellaswag_rc_5shot",  # 1K subset of HellaSwag
            "csqa_val_mc_5shot",
            "piqa_val_mc_5shot",
            "socialiqa_val_mc_5shot",
            "winogrande_val_rc_5shot",
            # Too noisy to be worth tracking
            # "boolq_val_mc_5shot",
            # "openbookqa_test_mc_5shot",
            # MMLU MC BPB
            "mmlu_stem_val_mc_5shot",
            "mmlu_humanities_val_mc_5shot",
            "mmlu_social_sciences_val_mc_5shot",
            "mmlu_other_val_mc_5shot",
            "mmlu_stem_test_mc_5shot",
            "mmlu_humanities_test_mc_5shot",
            "mmlu_social_sciences_test_mc_5shot",
            "mmlu_other_test_mc_5shot",
            # Gen tasks BPB
            "gsm8k_gold_bpb_5shot",
            "minerva_math_algebra_gold_bpb_0shot",
            "minerva_math_counting_and_probability_gold_bpb_0shot",
            "minerva_math_geometry_gold_bpb_0shot",
            "minerva_math_intermediate_algebra_gold_bpb_0shot",
            "minerva_math_number_theory_gold_bpb_0shot",
            "minerva_math_prealgebra_gold_bpb_0shot",
            "minerva_math_precalculus_gold_bpb_0shot",
            "codex_humaneval_gold_bpb_0shot",
            "codex_mbpp_gold_bpb_0shot",
            # Sanity check for MCQA ability
            "copycolors_10way",
        ]

        # Unfortunately we need the same metrics for everything, so we run them all.
        tasks = list(set(tasks_small_compute + tasks_large_compute))
        tasks.sort()
        return tasks
