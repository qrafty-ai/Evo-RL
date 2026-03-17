#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline policy evaluation on a LeRobot dataset (no real robot or sim).

Computes:
- Action error: MSE and L1 between policy-predicted actions and expert actions
  (behavior cloning / policy consistency metrics).
- Optional: episode-level success statistics from the dataset (what happened in
  the data; not policy-in-environment success).

Usage:

  # Basic (evaluate on full dataset)
  python examples/offline_eval/eval_policy_on_dataset.py \
    --policy.path=outputs/train/my_policy/checkpoints/005000/pretrained_model \
    --dataset.repo_id=username/my_dataset

  # Limit episodes and batch size
  python examples/offline_eval/eval_policy_on_dataset.py \
    --policy.path=outputs/train/my_policy/checkpoints/005000/pretrained_model \
    --dataset.repo_id=username/my_dataset \
    --dataset.episodes=0,1,2,3,4 \
    --batch_size=8

  # Save report to JSON
  python examples/offline_eval/eval_policy_on_dataset.py \
    --policy.path=... \
    --dataset.repo_id=... \
    --output_dir=outputs/offline_eval/run1
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import load_episodes
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class OfflineEvalConfig:
    """Configuration for offline policy evaluation on a dataset."""

    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: int = 16
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Max number of samples to evaluate (default: all)"},
    )
    device: str | None = field(
        default=None,
        metadata={"help": "Device (cuda, cpu, mps, auto)"},
    )
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Directory to save report JSON"},
    )
    seed: int = 1000
    num_workers: int = 0

    def __post_init__(self):
        register_third_party_plugins()
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("--policy.path is required")
        # Resolve to absolute so local checkpoint dirs are found when cwd is repo root
        policy_dir = Path(policy_path).resolve()
        if not policy_dir.is_dir():
            # Likely a local path that doesn't exist (e.g. wrong run name or step)
            if "/" in policy_path or "\\" in policy_path:
                raise FileNotFoundError(
                    f"Policy path is not an existing directory: {policy_dir}\n"
                    "Use an existing checkpoint path, e.g. "
                    "outputs/train/YOUR_RUN/checkpoints/030000/pretrained_model\n"
                    "List available runs: ls outputs/train/"
                )
        else:
            policy_path = str(policy_dir)
        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path

        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _expert_action_from_batch(batch: dict, action_key: str = ACTION) -> torch.Tensor:
    """Extract expert action tensor for comparison. Handles (B, D) or (B, T, D)."""
    a = batch[action_key]
    if not torch.is_tensor(a):
        a = torch.as_tensor(a, device=a.device if hasattr(a, "device") else None)
    if a.ndim == 3:
        # Chunk: use first step as the "current" action target
        a = a[:, 0, :]
    return a.float()


def run_offline_eval(cfg: OfflineEvalConfig) -> dict:
    set_seed(cfg.seed)
    init_logging()

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
    )

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.to(cfg.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        dataset_stats=dataset.meta.stats,
    )

    num_samples = len(dataset)
    if cfg.max_samples is not None:
        num_samples = min(num_samples, cfg.max_samples)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    mse_sum = 0.0
    l1_sum = 0.0
    n = 0
    action_dim = None

    with torch.no_grad():
        for batch in loader:
            if n >= num_samples:
                break
            # Move batch to device (preprocessor may expect it)
            batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            preprocessed = preprocessor(batch)
            pred_action = policy.select_action(preprocessed)
            pred_action = postprocessor(pred_action)
            expert = _expert_action_from_batch(batch).to(cfg.device)

            # Align shapes: policy may return (B, T, D), we compare to (B, D) by taking first step
            if pred_action.ndim == 3:
                pred_action = pred_action[:, 0, :]
            if expert.ndim == 1:
                expert = expert.unsqueeze(0)
            if pred_action.shape != expert.shape:
                expert = expert[:, : pred_action.shape[-1]]
            if action_dim is None:
                action_dim = pred_action.shape[-1]

            pred_action = pred_action.float().to(cfg.device)
            expert = expert.float().to(cfg.device)
            diff = pred_action - expert
            mse_sum += (diff * diff).sum().item()
            l1_sum += diff.abs().sum().item()
            n += pred_action.shape[0]

    if n == 0:
        report = {"num_samples": 0, "action_mse_mean": 0.0, "action_l1_mean": 0.0}
        logging.warning("No samples evaluated (empty dataset or max_samples=0)")
        return report

    total_elements = n * (action_dim or 1)
    mse_mean = mse_sum / total_elements
    l1_mean = l1_sum / total_elements

    report = {
        "num_samples": n,
        "action_mse_mean": mse_mean,
        "action_mse_per_sample": mse_sum / n,
        "action_l1_mean": l1_mean,
        "action_l1_per_sample": l1_sum / n,
    }

    # Optional: episode-level success from dataset (metadata only)
    try:
        episodes = load_episodes(dataset.meta.root)
        if episodes is not None and "episode_success" in episodes.column_names:
            success_col = episodes["episode_success"]
            success_counts = {}
            for v in success_col:
                s = str(v).strip().lower() if v is not None else "unknown"
                success_counts[s] = success_counts.get(s, 0) + 1
            total_ep = len(success_col)
            report["dataset_episode_success_counts"] = success_counts
            report["dataset_num_episodes"] = total_ep
            if "success" in success_counts:
                report["dataset_success_rate"] = success_counts["success"] / total_ep
    except Exception as e:
        logging.warning("Could not load episode_success from dataset: %s", e)

    logging.info("Offline eval report: %s", json.dumps(report, indent=2))
    return report


@parser.wrap()
def main(cfg: OfflineEvalConfig):
    report = run_offline_eval(cfg)
    if cfg.output_dir:
        out_path = Path(cfg.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "offline_eval_report.json", "w") as f:
            json.dump(report, f, indent=2)
        logging.info("Wrote %s", out_path / "offline_eval_report.json")


if __name__ == "__main__":
    main()
