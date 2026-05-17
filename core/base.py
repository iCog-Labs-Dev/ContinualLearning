from dataclasses import dataclass
from typing import Any


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

