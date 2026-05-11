from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple


@dataclass
class EWCState:
    old_params: Any
    cumulative_fisher: Any


@dataclass
class EWCVanillaState:
    anchors: list


@dataclass
class SIState:
    old_params: Any
    cumulative_omega: Any


class ContinualLearningMethod(Protocol):
    def train_task(
        self, model: Any, params: Any, state: Any, task: Any, task_idx: int
    ) -> Tuple[Any, Any, float]: ...

    def evaluate(
        self, model: Any, params: Any, task: Any, allowed_classes: Optional[list] = None
    ) -> float: ...
