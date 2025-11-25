from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.config import Config
from olmo_core.data import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)

M = TypeVar("M", bound=Config)


@dataclass(kw_only=True)
class ModelLadder(Config, Generic[M], metaclass=ABCMeta):
    """
    Represents a complete model ladder of runs.
    """

    name: str
    """A name to assign to the ladder."""
    sizes: list[str]
    """A list of model size specs to run as part of the ladder."""
    models: list[M]
    """A list of model specs to run as part of the ladder."""
    chinchilla_multiple: float
    """How long to train each run for, expressed as a multiple of the Chinchilla-optimal duration."""
    sequence_length: int = 4096
    """The sequence length to train each run on."""
    tokenizer: TokenizerConfig
    root_dir: str
    """The root directory where ladder run results should be saved."""
    seed: int = 42
    """The initial random seed to use for all runs in the ladder."""

    def __post_init__(self):
        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")

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

    def configure_trainer(self, size: str) -> TrainerConfig:
        run_id = f"{size}-{self.chinchilla_multiple:.2f}xC-{self.sequence_length}-{self.seed}"
        run_name = f"{self.name}-{run_id}"
        save_folder = io.join_path(self.root_dir, io.make_url_safe(self.name), run_id)
        work_dir = (
            "./cache" if io.is_url(self.root_dir) else str(io.join_path(self.root_dir, "cache"))
        )

        # Calculate training duration based on Chinchilla scaling laws.
        duration = Duration.chinchilla_tokens(
            self.chinchilla_multiple,
            model_params=model.num_non_embedding_params,
            #  model_params=run.target_num_parameters,
        )

        return TrainerConfig(
            save_folder=str(save_folder),
            work_dir=str(work_dir),
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
