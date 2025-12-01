import dataclasses
import json
import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MetricSaverCallback(Callback):
    """
    A callback that captures the latest metrics on rank 0.
    """

    path_stem: str = "metrics"
    metrics_to_capture: Optional[List[str]] = None
    enabled: bool = True
    fixed_steps: Optional[List[int]] = None

    _metrics: Optional[Dict[str, Any]] = dataclasses.field(default=None, repr=False)
    _metrics_step: int = dataclasses.field(default=0, repr=False)

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """
        The latest metrics recorded.
        """
        return self._metrics

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self.enabled or get_rank() != 0:
            return

        if self._metrics is None:
            self._metrics = {}

        if step >= self._metrics_step:
            if self.metrics_to_capture is not None:
                metrics = {
                    k: v
                    for k, v in metrics.items()
                    if any(fnmatch(k, pattern) for pattern in self.metrics_to_capture)
                }
            self._metrics.update(metrics)
            self._metrics_step = step

        if self.fixed_steps is not None and step in self.fixed_steps:
            dest_path = self._write_metrics(f"{self.path_stem}_step{step}.json", self._metrics)
            log.info(f"Metrics for step {step} saved to '{dest_path}'")

    def post_train(self):
        if not self.enabled or get_rank() != 0:
            return

        if self.metrics is not None:
            dest_path = self._write_metrics(f"{self.path_stem}.json", self.metrics)
            log.info(f"Final metrics from step {self._metrics_step} saved to '{dest_path}'")

    def close(self):
        if not self.enabled or get_rank() != 0:
            return

        self._metrics = None
        self._metrics_step = 0

    def _write_metrics(self, fname: str, metrics: Dict[str, float]) -> PathOrStr:
        return self.trainer.write_file(fname, json.dumps(metrics))
