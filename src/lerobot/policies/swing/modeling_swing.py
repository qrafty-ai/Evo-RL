#!/usr/bin/env python

import math
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from .configuration_swing import SwingConfig


class SwingPolicy(PreTrainedPolicy):
    config_class = SwingConfig
    name = "swing"

    def __init__(self, config: SwingConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self._step = 0
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_optim_params(self) -> dict:
        return {"params": [self._anchor]}

    def reset(self):
        self._step = 0

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        loss = self._anchor.sum() * 0.0
        return loss, None

    def _action_dim(self) -> int:
        action_feature = self.config.action_feature
        if action_feature is None or not action_feature.shape:
            raise ValueError("Swing policy requires a valid action feature shape.")
        return int(action_feature.shape[0])

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        return self.select_action(batch, **kwargs)

    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        action_dim = self._action_dim()
        action = torch.zeros((1, action_dim), device=self._anchor.device, dtype=torch.float32)

        if self.config.center_from_observation and OBS_STATE in batch:
            obs_state = batch[OBS_STATE]
            if obs_state.dim() == 1:
                obs_state = obs_state.unsqueeze(0)
            if obs_state.shape[-1] >= action_dim:
                action = obs_state[:, :action_dim].to(dtype=torch.float32).clone()

        period = max(self.config.swing_period_steps, 1)
        phase = 2.0 * math.pi * ((self._step % period) / period)
        offset = self.config.swing_amplitude if math.sin(phase) >= 0 else -self.config.swing_amplitude
        joint_idx = max(0, min(self.config.swing_joint_index, action_dim - 1))
        action[:, joint_idx] = action[:, joint_idx] + offset
        self._step += 1
        return action
