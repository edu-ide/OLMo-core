import json
import math
import re
import typing
import warnings
from dataclasses import dataclass

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmupAndLinearDecay,
    OptimConfig,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

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
    """Defines all components required to execute a single run within a transformer ladder experiment."""

    model: TransformerConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    backend: str
    seed: int


@dataclass(kw_only=True)
class TransformerLadderRun(ModelLadderRun):
    """
    Defines a single run within a :class:`TransformerLadderExperiment`.

    By default this will use the OLMo3 architecture, but subclasses can override :meth:`configure_model()`
    to customize it.
    """

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
    def target_num_parameters(self) -> int:
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
            self.chinchilla_multiple, model_params=self.target_num_parameters
        )

    @property
    def id(self) -> str:
        return f"{self.size}-{self.chinchilla_multiple:.2f}xC-{self.sequence_length}CL-{self.batch_size}BZ"

    def configure_model(self, vocab_size: int) -> TransformerConfig:
        model: TransformerConfig
        if self.size == TransformerSize.size_190M:
            model = TransformerConfig.olmo3_190M(vocab_size)
        elif self.size == TransformerSize.size_370M:
            model = TransformerConfig.olmo3_370M(vocab_size)
        elif self.size == TransformerSize.size_600M:
            model = TransformerConfig.olmo3_600M(vocab_size)
        elif self.size == TransformerSize.size_760M:
            model = TransformerConfig.olmo3_760M(vocab_size)
        elif self.size == TransformerSize.size_1B:
            model = TransformerConfig.olmo3_1B(vocab_size)
        elif self.size == TransformerSize.size_3B:
            model = TransformerConfig.olmo3_3B(vocab_size)
        elif self.size == TransformerSize.size_7B:
            model = TransformerConfig.olmo3_7B(vocab_size)
        elif self.size == TransformerSize.size_13B:
            model = TransformerConfig.olmo3_13B(vocab_size)
        else:
            raise OLMoConfigurationError(f"Unsupported model size '{self.size}'")
        # Make sure actual number of params is close to target number.
        if (
            pct_diff := (
                math.fabs(model.num_params - self.target_num_parameters)
                / self.target_num_parameters
            )
        ) > 0.05:
            warnings.warn(
                f"Configured model has {model.num_params:,d} parameters, "
                f"which differs from target of {self.size} by ~{100 * pct_diff:.1f}%.",
                UserWarning,
            )
        return model

    def configure_optimizer(self) -> OptimConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        #  assert self.sequence_length in {2048, 4096}
        lr = 0.0047 * (self.target_num_parameters / 108000000) ** (-1 / 3)

        return SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        )

    def configure_mbz(self) -> int:
        # Assumes H100 with 80GB memory.
        if self.target_num_parameters <= 370e6:
            return 16 * 4096
        elif self.target_num_parameters <= 760e6:
            return 10 * 4096
        elif self.target_num_parameters <= 1e9:
            return 8 * 4096
        elif self.target_num_parameters <= 7e9:
            return 4 * 4096
        else:
            return 2 * 4096

    def configure_train_module(self) -> TransformerTrainModuleConfig:
        mbz = max(self.sequence_length, self.configure_mbz())
        return TransformerTrainModuleConfig(
            rank_microbatch_size=mbz,
            max_sequence_length=self.sequence_length,
            optim=self.configure_optimizer(),
            compile_model=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            ),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=CosWithWarmupAndLinearDecay(
                warmup_steps=round(self.target_num_parameters / self.batch_size)
            ),
        )


@dataclass(kw_only=True)
class TransformerLadderExperiment(ModelLadderExperiment[TransformerLadderRun]):
    mix: DataMix
    mix_base_dir: str = "gs://ai2-llm"
    tokenizer: TokenizerConfig
    intra_document_masking: bool = False
    instance_filter: bool = False
    backend: str = "cpu:gloo,cuda:nccl"

    def run(self, run: TransformerLadderRun, overrides: list[str] | None = None):
        config = self.configure_run_components(run, overrides)
        prepare_training_environment(backend=config.backend, seed=self.seed)

        try:
            # Build components.
            model = config.model.build(init_device="meta")
            train_module = config.train_module.build(model)
            dataset = config.dataset.build()
            data_loader = config.data_loader.build(
                dataset, dp_process_group=train_module.dp_process_group
            )
            trainer = config.trainer.build(train_module, data_loader)

            # Record the config to W&B/Comet and each checkpoint dir.
            config_dict = config.as_config_dict()
            typing.cast(
                callbacks.ConfigSaverCallback, trainer.callbacks["config_saver"]
            ).config = config_dict

            # Train (also handles checkpoint loading).
            trainer.fit()

            # MetricSaverCallback will write final metrics to 'metrics.json'
            if dist_utils.get_rank() == 0:
                metrics_path = io.join_path(trainer.save_folder, "metrics.json")
                assert io.file_exists(metrics_path)
        finally:
            teardown_training_environment()

    def get_metrics_for_run(
        self, run: TransformerLadderRun, overrides: list[str] | None = None
    ) -> dict[str, float]:
        config = self.configure_run_components(run, overrides)
        path = io.resource_path(config.trainer.save_folder, "metrics.json")
        with path.open("r") as f:
            return json.load(f)

    def configure_run_components(
        self, run: TransformerLadderRun, overrides: list[str] | None = None
    ) -> TransformerLadderRunComponents:
        if overrides:
            self = self.merge(overrides, strict=False)

        save_folder = io.join_path(
            self.root_dir, io.make_url_safe(self.name), run.id, f"seed-{self.seed}"
        )
        work_dir = (
            "./cache" if io.is_url(self.root_dir) else str(io.join_path(self.root_dir, "cache"))
        )
        run_name = f"{self.name}-{run.id}-seed-{self.seed}"

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
        components = TransformerLadderRunComponents(
            model=run.configure_model(self.tokenizer.padded_vocab_size()),
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
