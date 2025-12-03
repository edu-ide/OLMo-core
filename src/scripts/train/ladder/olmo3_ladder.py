import argparse
import logging
import sys
import textwrap

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import build_launch_config, get_gpu_type, get_root_dir
from olmo_core.launch.beaker import (
    BeakerLaunchConfig,
    BeakerPriority,
    OLMoCoreBeakerImage,
)
from olmo_core.model_ladder2 import *
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    commands = ["dry_run", "benchmark", "launch_benchmark", "run", "launch_run"]
    parser = argparse.ArgumentParser(
        sys.argv[0],
        usage=f"python {sys.argv[0]} [CMD] [OPTIONS...]",
        description=textwrap.dedent(
            """
            Launch and manage a ladder experiment on Beaker.
            """
        ),
        epilog=textwrap.dedent(
            f"""
            examples:
              ❯ python {sys.argv[0]} dry_run --size=190M
            """
        ),
        formatter_class=type(  # type: ignore[arg-type]
            "CustomFormatter",
            (
                argparse.ArgumentDefaultsHelpFormatter,
                argparse.RawDescriptionHelpFormatter,
            ),
            {},
        ),
    )
    parser.add_argument(
        "cmd",
        type=str,
        choices=commands,
        help="The command to execute.",
    )
    parser.add_argument(
        "--size", choices=list(TransformerSize), required=True, help="The model size."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="olmo3-ladder",
        help="A name to assign to the ladder experiment.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        choices=["ai2/augusta", "ai2/jupiter", "ai2/titan"],
        default="ai2/augusta",
        help="The Beaker cluster to launch each run on.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=64,
        help="The maximum number of GPUs to use for the ladder.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="ai2/oe-t-ladder",
        help="The Beaker workspace to use.",
    )
    parser.add_argument(
        "--budget",
        type=str,
        default="ai2/oe-base",
        help="The Beaker budget to use.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4 * 1024,
        help="The sequence length to configure the ladder with.",
    )
    parser.add_argument(
        "--chinchilla-multiple",
        type=float,
        default=4.0,
        help="The Chinchilla multiple to use for the ladder.",
    )
    parser.add_argument(
        "--beaker-image",
        choices=list(OLMoCoreBeakerImage),
        default=OLMoCoreBeakerImage.stable,
        help="The Beaker image to use.",
    )
    parser.add_argument(
        "--priority",
        choices=[p.value for p in BeakerPriority],
        default=BeakerPriority.normal,
        help="The priority level.",
    )
    parser.add_argument(
        "--preemptible",
        action=argparse.BooleanOptionalAction,
        help="""If the job should be preemptible.""",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="""Allow launching with uncommitted changes.""",
        default=False,
    )

    # Make sure the command is in the right position, otherwise the way we build the launch
    # config would fail.
    if sys.argv[1] not in commands:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=DataMix.OLMo_mix_0925,
                    mix_base_dir=get_root_dir(args.cluster),
                )
            ],
            sequence_length=args.sequence_length,
        ),
    ]
    return ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model_ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=TransformerModelConfigurator(),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(num_workers=8),
    )


def configure_launcher(
    args: argparse.Namespace, ladder: ModelLadder, cmd: str
) -> BeakerLaunchConfig:
    ladder = configure_ladder(args)
    num_gpus = ladder.get_num_devices_for_run(args.size)
    assert (num_gpus % 8 == 0) or num_gpus < 8
    launch_config = build_launch_config(
        cmd=[sys.argv[0], cmd] + sys.argv[2:],
        name=f"{args.name}-{args.size}",
        num_nodes=max(num_gpus // 8, 1),
        cluster=args.cluster,
        workspace=args.workspace,
        beaker_image=args.beaker_image,
        budget=args.budget,
    )
    if num_gpus < 8:
        launch_config.num_gpus = num_gpus
    launch_config.priority = BeakerPriority(args.priority)
    if args.preemptible is not None:
        launch_config.preemptible = args.preemptible
    launch_config.allow_dirty = args.allow_dirty
    return launch_config


def main():
    args = parse_args()
    if args.cmd == "dry_run":
        dry_run(args)
    elif args.cmd == "benchmark":
        benchmark(args)
    elif args.cmd == "launch_benchmark":
        launch_benchmark(args)
    else:
        raise NotImplementedError(f"Command '{args.cmd}' is not implemented.")


def dry_run(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)
    ladder.dry_run(args.size)


def benchmark(args: argparse.Namespace):
    ladder = configure_ladder(args)
    ladder.run_benchmark(args.size)


def launch_benchmark(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)
    launcher = configure_launcher(args, ladder, "benchmark")
    launcher.launch(follow=True, slack_notifications=False)


if __name__ == "__main__":
    main()
