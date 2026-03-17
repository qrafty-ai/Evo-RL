#!/usr/bin/env python

"""
Export overlay videos using Qwen-computed task completion percentages.

This script:
- Reads a JSONL predictions file produced by Qwen (per-episode, 10 frames each)
  with `predicted_percentages` for given `original_frames_indices`.
- Loads the corresponding LeRobot dataset.
- Builds a per-frame value signal and injects it as a new column into the
  in-memory dataset (no on-disk modification).
- Uses `lerobot.scripts.value_infer_viz._export_overlay_videos` to render
  videos with the Qwen completion curve overlaid on the robot camera stream.

Example:

python examples/value_eval/export_qwen_completion_overlay.py \\
  --dataset.repo_id=qrafty-ai/use_spoon_joint \\
  --predictions_path=outputs/value_infer/openarm_spoon_toprewad7b/Qwen_Qwen2.5-VL-7B-Instruct_2026-03-14T01-18-49.627556_predictions.jsonl \\
  --output_dir=outputs/value_infer/openarm_spoon_toprewad7b/viz_qwen \\
  --value_field=observation.qwen_completion
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.value_infer_viz import _export_overlay_videos


def load_predictions(path: Path) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def build_qwen_value_column(
    dataset: LeRobotDataset,
    predictions: list[dict[str, Any]],
    scale_to_unit: bool = True,
) -> np.ndarray:
    """Build a per-frame value array aligned with dataset.hf_dataset.

    For each prediction entry:
    - Use `eval_episode.episode_index` and `eval_episode.original_frames_indices`
      to locate frames in the dataset.
    - Assign `predicted_percentages` to those frames.
      Other frames remain 0 by default.
    """
    raw = dataset.hf_dataset.with_format(None)
    n = len(raw)
    episode_indices = np.asarray(raw["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(raw["frame_index"], dtype=np.int64).reshape(-1)

    values = np.zeros(n, dtype=np.float32)

    for entry in predictions:
        ep = int(entry["eval_episode"]["episode_index"])
        frame_ids = entry["eval_episode"]["original_frames_indices"]
        preds = entry["predicted_percentages"]
        if len(frame_ids) != len(preds):
            continue

        for frame_idx, pct in zip(frame_ids, preds):
            mask = (episode_indices == ep) & (frame_indices == int(frame_idx))
            if not np.any(mask):
                continue
            val = float(pct)
            if scale_to_unit:
                val = val / 100.0
            values[mask] = val

    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export overlay videos with Qwen completion predictions.",
    )
    parser.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        type=str,
        required=True,
        help="LeRobot dataset repo id, e.g. qrafty-ai/use_spoon_joint",
    )
    parser.add_argument(
        "--dataset.root",
        dest="dataset_root",
        type=str,
        default=None,
        help="Optional local root for dataset (defaults to HF_LEROBOT_HOME layout).",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to Qwen predictions JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write overlay videos.",
    )
    parser.add_argument(
        "--value_field",
        type=str,
        default="observation.qwen_completion",
        help="Name of the value field column to inject into the dataset.",
    )
    parser.add_argument(
        "--viz.episodes",
        dest="viz_episodes",
        type=str,
        default="all",
        help="Episodes to visualize (e.g. 'all', '0-4', '0,2,5').",
    )
    parser.add_argument(
        "--viz.video_key",
        dest="video_key",
        type=str,
        default=None,
        help="Single camera key to use (overrides video_keys).",
    )
    parser.add_argument(
        "--viz.video_keys",
        dest="video_keys",
        type=str,
        default=None,
        help="Comma-separated camera keys for multiview overlay.",
    )
    parser.add_argument(
        "--viz.overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing videos in output_dir.",
    )
    parser.add_argument(
        "--viz.vcodec",
        dest="vcodec",
        type=str,
        default="libsvtav1",
        help="Video codec (h264, hevc, libsvtav1).",
    )
    parser.add_argument(
        "--viz.frame_storage_mode",
        dest="frame_storage_mode",
        type=str,
        default="memory",
        help="Frame storage mode: 'memory' or 'disk'.",
    )
    parser.add_argument(
        "--viz.smooth_window",
        dest="smooth_window",
        type=int,
        default=1,
        help="Savitzky-Golay smoothing window (>=1, 1 disables smoothing).",
    )

    args = parser.parse_args()

    ds_root = args.dataset_root
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=ds_root,
        episodes=None,
        delta_timestamps=None,
        image_transforms=None,
        revision=None,
    )

    preds_path = Path(args.predictions_path)
    predictions = load_predictions(preds_path)

    values_all = build_qwen_value_column(dataset, predictions, scale_to_unit=True)

    raw = dataset.hf_dataset.with_format(None)
    if args.value_field in raw.column_names:
        dataset.hf_dataset = dataset.hf_dataset.remove_columns([args.value_field])
    dataset.hf_dataset = dataset.hf_dataset.add_column(args.value_field, values_all.tolist())

    out_dir = Path(args.output_dir)
    written = _export_overlay_videos(
        dataset=dataset,
        value_field=args.value_field,
        advantage_field="observation.qwen_advantage",
        indicator_field="observation.qwen_indicator",
        viz_episodes=args.viz_episodes,
        video_key=args.video_key,
        video_keys=args.video_keys,
        output_dir=out_dir,
        overwrite=args.overwrite,
        vcodec=args.vcodec,
        frame_storage_mode=args.frame_storage_mode,
        smooth_window=args.smooth_window,
    )

    print(f"Exported {len(written)} videos to {out_dir}")
    for p in written:
        print(p)


if __name__ == "__main__":
    main()

