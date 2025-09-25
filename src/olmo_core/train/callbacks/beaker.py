import dataclasses
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from ..common import TrainingProgress
from .callback import Callback
from .comet import CometCallback
from .wandb import WandBCallback

log = logging.getLogger(__name__)


BEAKER_EXPERIMENT_ID_ENV_VAR = "BEAKER_EXPERIMENT_ID"


@dataclass
class BeakerCallback(Callback):
    """
    Adds metadata to the Beaker experiment description when running as a Beaker batch job.
    """

    priority: ClassVar[int] = min(CometCallback.priority - 1, WandBCallback.priority - 1)
    update_interval: Optional[int] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    """
    A JSON-serializable config to save to the results dataset as ``config.json``.
    """

    _client = dataclasses.field(default=None, repr=False)
    _url = dataclasses.field(default=None, repr=False)
    _last_update: Optional[float] = dataclasses.field(default=None, repr=False)

    def post_attach(self):
        if self.enabled is None and BEAKER_EXPERIMENT_ID_ENV_VAR in os.environ:
            self.enabled = True

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            import gantry
            from beaker import Beaker

            if BEAKER_EXPERIMENT_ID_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{BEAKER_EXPERIMENT_ID_ENV_VAR}'")

            self._client = Beaker.from_env(check_for_upgrades=False)

            experiment_id = os.environ[BEAKER_EXPERIMENT_ID_ENV_VAR]
            workload = self._client.workload.get(experiment_id)

            log.info(f"Running in Beaker experiment {self._client.workload.url(workload)}")

            # Ensure result dataset directory exists.
            result_dir = Path(gantry.RESULTS_DIR) / "olmo-core"
            result_dir.mkdir(parents=True, exist_ok=True)

            # Save config to result dir.
            if self.config is not None:
                config_path = result_dir / "config.json"
                with config_path.open("w") as config_file:
                    log.info(f"Saving config to '{config_path}'")
                    json.dump(self.config, config_file)

            # Try saving Python requirements.
            requirements_path = result_dir / "requirements.txt"
            try:
                with requirements_path.open("w") as requirements_file:
                    requirements_file.write(f"# python={platform.python_version()}\n")
                with requirements_path.open("a") as requirements_file:
                    subprocess.call(
                        ["pip", "freeze"],
                        stdout=requirements_file,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
            except Exception as e:
                log.exception(f"Error saving Python packages: {e}")

            # Try to get W&B/Comet URL of experiment.
            for callback in self.trainer.callbacks.values():
                if isinstance(callback, WandBCallback) and callback.enabled:
                    if (url := callback.run.get_url()) is not None:
                        self._url = url
                    break
                elif isinstance(callback, CometCallback) and callback.enabled:
                    if (url := callback.exp.url) is not None:
                        self._url = url
                    break

            self._update()

    def post_step(self):
        update_interval = self.update_interval or self.trainer.metrics_collect_interval
        if self.enabled and get_rank() == 0 and self.step % update_interval == 0:
            # Make sure we don't update too frequently.
            if self._last_update is None or (time.monotonic() - self._last_update) > 10:
                self._update()

    def post_train(self):
        if self.enabled and get_rank() == 0:
            self._update()

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def _update(self):
        self.trainer.run_bookkeeping_op(
            self._set_description,
            self.trainer.training_progress,
            op_name="beaker_set_description",
            allow_multiple=False,
            distributed=False,
        )
        self._last_update = time.monotonic()

    def _set_description(self, progress: TrainingProgress):
        import gantry

        description = f"[{progress}] "
        if self._url is not None:
            description = f"{description}{self._url} "

        try:
            gantry.api.update_workload_description(
                description, strategy="prepend", client=self._client
            )
        except Exception as e:
            log.error(f"Failed to update Beaker experiment description: {e}")
