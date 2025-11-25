import dataclasses
import json
import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MetricSaverCallback(Callback):
    """
    A callback that captures the latest metrics on rank 0.
    """

    fname: str = "metrics.json"
    metrics_to_capture: Optional[List[str]] = None
    enabled: bool = True
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

    def post_train(self):
        if not self.enabled or get_rank() != 0:
            return

        if self.metrics is not None:
            dest_path = self.trainer.write_file(self.fname, json.dumps(self.metrics))
            log.info(f"Final metrics from step {self._metrics_step} saved to '{dest_path}'")

    def close(self):
        if not self.enabled or get_rank() != 0:
            return

        self._metrics = None
        self._metrics_step = 0
