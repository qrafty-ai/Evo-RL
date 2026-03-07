#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("swing")
@dataclass
class SwingConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    swing_joint_index: int = 0
    swing_amplitude: float = 0.08
    swing_period_steps: int = 60
    center_from_observation: bool = True
    optimizer_lr: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.swing_period_steps <= 0:
            raise ValueError("`swing_period_steps` must be > 0.")
        if self.swing_amplitude < 0:
            raise ValueError("`swing_amplitude` must be >= 0.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("Swing policy requires an action feature.")
        if not self.action_feature.shape:
            raise ValueError("Swing policy action feature shape must be non-empty.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
