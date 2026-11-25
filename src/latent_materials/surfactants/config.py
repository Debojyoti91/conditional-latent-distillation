"""Explicit controls for conditioning-ablation experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    """Reported defaults; reduced epochs/seeds are useful only for smoke checks."""

    global_seed: int = 42
    optimization_seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    property_shuffles: int = 20
    bootstrap_replicates: int = 10000
    max_epochs: int = 1000
    patience: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self):
        if not self.optimization_seeds or len(set(self.optimization_seeds)) != len(
            self.optimization_seeds
        ):
            raise ValueError("Use a nonempty tuple of distinct optimization seeds")
        for name in [
            "property_shuffles",
            "bootstrap_replicates",
            "max_epochs",
            "patience",
            "batch_size",
        ]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
