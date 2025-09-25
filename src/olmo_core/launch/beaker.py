"""
Launch experiments on `Beaker <https://beaker.org>`_.
"""

import argparse
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import gantry
import rich
from beaker import Beaker, BeakerWorkload
from beaker.exceptions import BeakerImageNotFound

from ..config import Config, StrEnum
from ..distributed.utils import OLMO_SHARED_FS_ENV_VAR
from ..exceptions import OLMoConfigurationError, OLMoEnvironmentError
from ..utils import (
    LOG_FILTER_TYPE_ENV_VAR,
    LogFilterType,
    generate_uuid,
    prepare_cli_environment,
)
from ..version import VERSION
from .select_beaker_hosts import get_beaker_hostname_constraints

log = logging.getLogger(__name__)


__all__ = [
    "OLMoCoreBeakerImage",
    "BeakerLaunchConfig",
]


_DEFAULT_TORCH = "2.7.1".replace(".", "")
_DEFAULT_CUDA = "12.8".replace(".", "")


class OLMoCoreBeakerImage(StrEnum):
    """
    Official Beaker images that work well for OLMo-core.

    You can find the full list at
    `beaker.org/ws/ai2/OLMo-core/images <https://beaker.org/ws/ai2/OLMo-core/images>`_, which
    includes *versioned* images that are published with each release of the OLMo-core package.
    """

    # NOTE: when updating default images here, should also update images used in tests at .github/workflows/main.yml

    stable = f"olmo-core-tch{_DEFAULT_TORCH}cu{_DEFAULT_CUDA}-2025-09-15"
    """
    Built with the latest compatible stable version of PyTorch.
    """

    stable_cu126 = f"olmo-core-tch{_DEFAULT_TORCH}cu126-2025-09-15"
    """
    The stable image with CUDA pinned to 12.6.
    """

    stable_cu128 = f"olmo-core-tch{_DEFAULT_TORCH}cu128-2025-09-15"
    """
    The stable image with CUDA pinned to 12.8.
    """

    tch280_cu128 = "olmo-core-tch280cu128-2025-09-19"
    """
    Built with torch 2.8.0 and CUDA 12.8.
    """

    tch280_cu129 = "olmo-core-tch280cu129-2025-09-23"
    """
    Built with torch 2.8.0 and CUDA 12.9.
    """

    flash_attn_3 = "tylerr/olmo-core-tch270cu128-2025-09-24"
    """
    Built flash-attn 3 (beta release) with torch 2.7.0 and CUDA 12.8.
    """


def is_running_in_beaker() -> bool:
    """
    Check if the current process is running inside of a Beaker job (batch or session).
    """
    # There's a number of different environment variables set by the Beaker executor.
    # Checking any one of these would suffice, but we check a couple to reduce the
    # risk of false positives.
    return "BEAKER_JOB_ID" in os.environ and "BEAKER_NODE_ID" in os.environ


def is_running_in_beaker_batch_job() -> bool:
    """
    Check if the current process is running inside a Beaker batch job (as opposed to a session).
    """
    return is_running_in_beaker() and os.environ.get("BEAKER_JOB_KIND") == "batch"


@dataclass
class BeakerLaunchConfig(Config):
    """
    Config for launching experiments on Beaker.
    """

    name: str
    """
    A name to assign the Beaker experiment.
    """

    cmd: List[str]
    """
    The command to run in the container.
    """

    torchrun: bool = True
    """
    Run the command with ``torchrun``.
    """

    budget: Optional[str] = None
    """
    The budget group to assign.
    """

    task_name: str = "train"
    """
    A name to assign the Beaker tasks created.
    """

    workspace: Optional[str] = None
    """
    The Beaker workspace to use.
    """

    description: Optional[str] = None
    """
    A description for the experiment.
    """

    beaker_image: str = OLMoCoreBeakerImage.stable
    """
    The Beaker image to use.

    Suitable images can be found at `beaker.org/ws/ai2/OLMo-core/images <https://beaker.org/ws/ai2/OLMo-core/images>`_.
    """

    num_nodes: int = 1
    """
    The number of nodes to use.
    """

    num_gpus: int = 8
    """
    The number of GPUs to use per node.
    """

    shared_memory: str = "10GiB"
    """
    The amount of shared memory to use.
    """

    cluster: str = "ai2/jupiter"
    """
    The cluster to run on.
    """

    shared_filesystem: bool = False
    """
    Set this to true if the save folder and working directory for each node is part of a global
    shared filesystem (like weka or NFS).
    """

    priority: str = "normal"
    """
    The job priority.
    """

    preemptible: bool = True
    """
    If the job should be preemptible.
    """

    retries: Optional[int] = None
    """
    The number of times to retry the experiment if it fails.
    """

    env_vars: List[Tuple[str, str]] = field(default_factory=list)
    """
    Additional env vars to include.
    """

    env_secrets: List[Tuple[str, str]] = field(default_factory=list)
    """
    Environment variables to add from secrets.
    """

    nfs: bool = False
    """
    Attach the NFS drive.
    """

    weka_buckets: List[Tuple[str, str]] = field(default_factory=list)
    """
    Weka buckets to attach and where to attach them.
    """

    allow_dirty: bool = False
    """
    Allow running with uncommitted changed.
    """

    git: Optional[gantry.api.GitRepoState] = field(default_factory=gantry.api.GitRepoState.from_env)
    """
    Git configuration, specifies where to clone your source code from and which commit to check out.
    If not set, this will be initialized automatically from your working directory.
    """

    use_hostname_constraints: bool = False
    """
    Uses hostname constraints to restrict the hostnames on which the experiment runs. This is currently
    only supported for Augusta clusters, and can benefit performance by forcing the use of colocated nodes.

    This is NOT recommended to be used with lower priority preemptible jobs, since hostname constraints are not
    updated on preemption.
    """

    num_execution_units: Optional[int] = None
    """
    Number of "execution units", defaults to ``max(1, num_nodes // 32)``. An "execution unit" is abstraction
    for any node-using entity of which 1 or more copies are run, where each unit wants its nodes to be
    from colocated hardware (e.g., a model replica for large jobs, or a full distributed model for small jobs).

    For internal experiments, this defaults to the number of data-parallel model replicas instead.
    """

    # NOTE: don't assign a type here because omegaconf can't validate arbitrary classes
    #  _beaker: Optional[Beaker] = None
    _beaker = None

    @property
    def default_env_vars(self) -> List[Tuple[str, str]]:
        """
        Default env vars to add to the experiment.
        """
        env_vars: List[Tuple[str, str]] = [
            ("NCCL_DEBUG", "INFO"),
            (LOG_FILTER_TYPE_ENV_VAR, LogFilterType.local_rank0_only),
            ("OMP_NUM_THREADS", "8"),
            ("R2_PROFILE", "R2"),
            ("S3_PROFILE", "S3"),
            ("WEKA_PROFILE", "WEKA"),
            ("NUM_NODES", str(self.num_nodes)),
            ("OLMO_CORE_VERSION", VERSION),
            ("FORCE_COLOR", "1"),  # for 'rich' because Beaker supports ANSI colors in logs
        ]
        if self.shared_filesystem:
            env_vars.append((OLMO_SHARED_FS_ENV_VAR, "1"))
        return env_vars

    def _get_env_vars(self) -> List[Tuple[str, str]]:
        env_vars: List[Tuple[str, str]] = []
        env_var_names: Set[str] = set()
        for name, value in self.env_vars:
            env_vars.append((name, value))
            env_var_names.add(name)
        for name, val in self.default_env_vars:
            if name not in env_var_names:
                env_vars.append((name, val))
        return env_vars

    def build_recipe(self) -> gantry.api.Recipe:
        with Beaker.from_env() as beaker:
            cluster = beaker.cluster.get(self.cluster)

            if self.torchrun:
                recipe = gantry.api.Recipe.multi_node_torchrun(
                    self.cmd, gpus_per_node=self.num_gpus, num_nodes=self.num_nodes
                )
            else:
                recipe = gantry.api.Recipe(self.cmd)
                recipe.gpus = self.num_gpus
                if self.num_nodes > 1:
                    recipe = recipe.with_replicas(self.num_nodes)

            recipe.name = self.name
            recipe.task_name = self.task_name
            recipe.description = self.description
            recipe.budget = self.budget
            recipe.workspace = self.workspace
            recipe.priority = self.priority
            recipe.preemptible = self.preemptible
            recipe.shared_memory = self.shared_memory
            recipe.allow_dirty = self.allow_dirty
            recipe.clusters = [self.cluster]
            recipe.retries = self.retries
            recipe.env_vars = self._get_env_vars()
            recipe.env_secrets = [(n, s) for n, s in self.env_secrets]
            if self.git is not None:
                recipe.ref = self.git.ref
                recipe.branch = self.git.branch

            # Resolve beaker image.
            image = self.beaker_image
            try:
                image = beaker.image.get(image).id
            except BeakerImageNotFound as exc:
                # Image name was already a full name, so it probably doesn't exist.
                if "/" in image:
                    raise

                # Try pre-pending 'petew', since that's the account that we usually build the images from.
                try:
                    image = beaker.image.get(f"petew/{image}").id
                except BeakerImageNotFound:
                    raise exc
            recipe.beaker_image = image

            if self.nfs:
                if "storage:nfs" not in cluster.tags:
                    raise OLMoConfigurationError(
                        "NFS was requested but the cluster doesn't have access to NFS"
                    )
                recipe.mounts = ["/net/nfs.cirrascale"]

            if self.weka_buckets:
                if "storage:weka" not in cluster.tags:
                    raise OLMoConfigurationError(
                        "Weka buckets were requested but the cluster doesn't have access to Weka"
                    )
                recipe.weka = [(bucket, mount) for bucket, mount in self.weka_buckets]

            if "provider:gcp" in cluster.tags:
                if self.num_nodes > 0:
                    recipe.post_setup = (
                        "BEAKER_REPLICA_RANK=$("
                        "python -m olmo_core.launch.reorder_ranks_in_gcp "
                        "--verbose "
                        "${BEAKER_REPLICA_RANK} "
                        "${BEAKER_REPLICA_COUNT} "
                        "${BEAKER_LEADER_REPLICA_HOSTNAME}"
                        ")"
                    )

                if self.use_hostname_constraints:
                    if self.retries is not None and self.retries > 0:
                        raise OLMoConfigurationError(
                            "Hostname constraints cannot be used for beaker jobs with retries since constraints do not update on retry."
                        )
                    host_name_constraints = get_beaker_hostname_constraints(
                        self.num_nodes,
                        self.num_execution_units or max(1, self.num_nodes // 32),
                        1,
                        "us-central1-b",
                        beaker_cluster=self.cluster,
                        beaker_priority=self.priority,
                    )
                    assert (
                        len(host_name_constraints) == 1
                        and len(host_name_constraints[0]) >= self.num_nodes
                    )
                    recipe.hostnames = host_name_constraints[0]
                    recipe.clusters = []

        return recipe

    def launch(
        self,
        follow: bool = False,
        slack_notifications: Optional[bool] = None,
    ) -> BeakerWorkload:
        """
        Launch a Beaker experiment using this config.

        .. tip::
            You can preview what the Beaker experiment spec would like using
            :meth:`build_experiment_spec()`.

        :param follow: Stream the logs and follow the experiment until completion.
        :param torchrun: Launch the target command with ``torchrun``. This will default to ``True``
            if ``num_gpus > 1`` and ``False`` otherwise.
        :param entrypoint: Provide an optional entrypoint program if ``torchrun`` is ``False``.
            Defaults to 'python'.
        :param slack_notifications: If ``follow=True``, send Slack notifications when the run launches,
            fails, or succeeds. This requires the env var ``SLACK_WEBHOOK_URL``.

        :returns: The Beaker experiment.
        """
        # Check for webhook URL env var if needed.
        slack_webhook_url: Optional[str] = None
        if follow and slack_notifications is not False:
            from olmo_core.train.callbacks.slack_notifier import (
                SLACK_WEBHOOK_URL_ENV_VAR,
            )

            if slack_notifications is None:
                slack_notifications = SLACK_WEBHOOK_URL_ENV_VAR in os.environ
            elif slack_notifications and SLACK_WEBHOOK_URL_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(
                    f"Missing env var '{SLACK_WEBHOOK_URL_ENV_VAR}' for Slack notifications"
                )

            slack_webhook_url = os.environ.get(SLACK_WEBHOOK_URL_ENV_VAR)

        recipe = self.build_recipe()
        recipe.slack_webhook_url = slack_webhook_url

        return recipe.launch(show_logs=follow)


def _parse_args():
    parser = argparse.ArgumentParser(
        "olmo_core.launch.beaker",
        usage="python -m olmo_core.launch.beaker [OPTIONS...] -- [CMD...]",
        description=textwrap.dedent(
            """
            Launch a command on Beaker.
            """
        ),
        epilog=textwrap.dedent(
            """
            examples:
              ❯ python -m olmo_core.launch.beaker -- echo "Hello, World!"
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
        "--name", type=str, default="olmo-core-test", help="A name to assign to the run."
    )
    parser.add_argument(
        "--task-name", type=str, default="main", help="A name to assign to the task."
    )
    parser.add_argument("--gpus", type=int, default=0, help="The number of GPUs per node/replica.")
    parser.add_argument("--nodes", type=int, default=1, help="The number of nodes/replicas.")
    parser.add_argument("--budget", type=str, help="The Beaker budget account to use.")
    parser.add_argument("--workspace", type=str, help="The Beaker workspace to use.")
    parser.add_argument(
        "--description", type=str, help="A description to assign to the Beaker experiment."
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="ai2/jupiter",
        help="""Cluster to launch on.""",
    )
    parser.add_argument(
        "--priority",
        choices=["low", "normal", "high", "urgent"],
        default="normal",
        help="The priority level.",
    )
    parser.add_argument(
        "--preemptible",
        action="store_true",
        help="""If the job should be preemptible.""",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="""Allow launching with uncommitted changes.""",
        default=False,
    )
    parser.add_argument(
        "--beaker-image",
        type=str,
        default=OLMoCoreBeakerImage.stable,
        help="""The Beaker image to use.""",
    )
    parser.add_argument(
        "--shared-filesystem",
        action="store_true",
        help="""Use this flag if the save folder and working directory for each node is part of a global
        shared filesystem (like weka or NFS).""",
    )
    parser.add_argument("--weka", type=str, nargs="*", help="Weka buckets to mount at '/weka/'.")
    parser.add_argument(
        "--torchrun",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""If the command should be run via torchrun. This will default to true when '--gpus' is greater than 1.""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="""Set debugging env vars, like 'CUDA_LAUNCH_BLOCKING=1'.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Do a dry run where the launch config is printed.""",
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="*",
        help="""Environment variables to add to the Beaker experiment.
        Should be in the form '{NAME}={VALUE}'. Multiple allowed, space separated.""",
    )
    parser.add_argument(
        "--env-secret",
        type=str,
        nargs="*",
        help="""Environment variables to add to the Beaker experiment from Beaker secrets.
        Should be in the form '{NAME}={SECRET_NAME}'. Multiple allowed, space separated.""",
    )

    if len(sys.argv) < 3 or "--" not in sys.argv:
        parser.print_help()
        sys.exit(1)

    sep_index = sys.argv.index("--")
    args = sys.argv[1:sep_index]
    command = sys.argv[sep_index + 1 :]
    opts = parser.parse_args(args)
    return opts, command


def _build_config(opts: argparse.Namespace, command: List[str]) -> BeakerLaunchConfig:
    env_vars: List[Tuple[str, str]] = []
    if opts.debug:
        env_vars.append(("CUDA_LAUNCH_BLOCKING", "1"))
        env_vars.append(("NCCL_DEBUG", "INFO"))
    for e in opts.env or []:
        if "=" not in e:
            raise ValueError(f"Invalid env var '{e}', must be in the form NAME=VALUE")
        name, value = e.split("=", 1)
        env_vars.append((name, value))
    env_secrets: List[Tuple[str, str]] = []
    for e in opts.env_secret or []:
        if "=" not in e:
            raise ValueError(f"Invalid env secret '{e}', must be in the form NAME=SECRET_NAME")
        name, secret = e.split("=", 1)
        env_secrets.append((name, secret))
    return BeakerLaunchConfig(
        name=f"{opts.name}-{generate_uuid()[:8]}",
        budget=opts.budget,
        cmd=command,
        env_vars=env_vars,
        env_secrets=env_secrets,
        task_name=opts.task_name,
        description=opts.description,
        cluster=opts.cluster,
        num_nodes=opts.nodes,
        num_gpus=opts.gpus,
        preemptible=opts.preemptible,
        priority=opts.priority,
        beaker_image=opts.beaker_image,
        workspace=opts.workspace,
        allow_dirty=opts.allow_dirty,
        shared_filesystem=opts.shared_filesystem,
        weka_buckets=[(bucket, f"/weka/{bucket}") for bucket in (opts.weka or [])],
    )


def main():
    opts, command = _parse_args()
    prepare_cli_environment()
    config = _build_config(opts, command)
    if opts.dry_run:
        rich.print(config)
    else:
        config.launch(follow=True)


if __name__ == "__main__":
    main()
