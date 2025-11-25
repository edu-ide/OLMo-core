from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from olmo_core.config import Config


@dataclass(kw_only=True)
class ModelLadderRun(Config, metaclass=ABCMeta):
    """
    Defines a particular run within a :class:`ModelLadderExperiment`.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """A path/URL safe identifier for this run."""
        raise NotImplementedError


R = TypeVar("R", bound=ModelLadderRun)


@dataclass(kw_only=True)
class ModelLadderExperiment(Config, Generic[R], metaclass=ABCMeta):
    """
    Represents a complete model ladder experiment defined by a series concrete :class:`ModelLadderRun`s.
    """

    name: str
    """A name to assign to the experiment."""
    runs: list[R]
    """A list of runs to execute as part of this experiment."""
    root_dir: str
    """The root directory where experiment results should be saved."""
    seed: int = 42
    """The initial random seed to use for all runs in the experiment."""

    @abstractmethod
    def run(self, run: R, overrides: list[str] | None = None):
        """
        Execute a particular run of the experiment locally and store the results.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics_for_run(self, run: R, overrides: list[str] | None = None) -> dict[str, float]:
        """
        Retrieve the final metrics for a particular run of the experiment.
        """
        raise NotImplementedError
