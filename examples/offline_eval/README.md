# Offline evaluation (no real robot / no sim)

Two ways to evaluate **without** running the policy in simulation or on hardware:

---

## 1. Policy offline metrics: action error (MSE / L1)

**Script:** `eval_policy_on_dataset.py`

- Loads a **policy** and a **dataset** (same format as training).
- For each batch: feeds observations (images + state) to the policy, gets predicted actions, and compares them to the **expert actions** in the dataset.
- Reports **MSE** and **L1** over action dimensions (behavior cloning / consistency metrics).
- Optionally reports **episode-level success counts** from the dataset metadata (`episode_success`): this is “what happened in the data” (e.g. how many episodes were labeled success/failure during recording), **not** success rate of the policy when run in an environment.

**When to use:** To check how close the policy’s actions are to the expert on a fixed dataset (e.g. validation set, or a test repo). No robot and no sim required.

**Example:**

```bash
python examples/offline_eval/eval_policy_on_dataset.py \
  --policy.path=outputs/train/my_policy/checkpoints/005000/pretrained_model \
  --dataset.repo_id=username/my_dataset \
  --output_dir=outputs/offline_eval/run1
```

---

## 2. RTC strategy comparison on the dataset

**Script:** `examples/rtc/eval_dataset.py`

- **RTC (Real-Time Chunking)** is a way to run action-chunking policies in real time by reusing the tail of the previous chunk and guiding the next chunk.
- This script does **not** run the policy on a robot or in sim. It runs **on dataset samples only**:
  - Takes two random samples from the dataset.
  - Uses the first to generate a “previous chunk” of actions.
  - On the second sample, it runs the policy **twice**: once **without** RTC and once **with** RTC (same noise for a fair comparison).
- It then compares the **predicted action chunks** (with vs without RTC) and can plot denoising steps, corrections, and final actions.

**When to use:** To debug or compare RTC vs non-RTC **on fixed data** (e.g. check that RTC guidance changes the chunk in a sensible way, or compare consistency with ground truth). Still no robot and no sim.

**Example:**

```bash
uv run python examples/rtc/eval_dataset.py \
  --policy.path=lerobot/pi05_libero_finetuned \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --rtc.execution_horizon=8 \
  --device=cuda
```

---

## Summary

| Goal | Script | What it does |
|------|--------|--------------|
| How well do policy actions match expert actions on a dataset? | `offline_eval/eval_policy_on_dataset.py` | MSE / L1 vs expert; optional dataset success stats |
| How does RTC change predictions on fixed samples? | `rtc/eval_dataset.py` | Compare RTC vs no-RTC on two dataset samples, optional plots |

Both are **offline**: they only use recorded data and do not need a real robot or simulation environment.
