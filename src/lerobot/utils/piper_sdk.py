#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES) + ("gripper.pos",)


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


def _iter_candidate_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    env_path = os.environ.get("PIPER_SDK_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            Path.cwd(),
            repo_root,
            repo_root.parent,
            repo_root / "piper_sdk",
            repo_root.parent / "piper_sdk",
        ]
    )
    return candidates


def _add_path_if_matches(candidate: Path) -> bool:
    candidate = candidate.resolve()

    package_dir = candidate / "piper_sdk"
    if package_dir.is_dir() and (package_dir / "__init__.py").is_file():
        parent = str(candidate)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return True

    if candidate.name == "piper_sdk" and (candidate / "__init__.py").is_file():
        parent = str(candidate.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return True

    return False


@lru_cache(maxsize=1)
def get_piper_sdk() -> tuple[type[Any], Any]:
    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        return C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError:
        pass

    searched_paths: list[str] = []
    for candidate in _iter_candidate_paths():
        searched_paths.append(str(candidate))
        if not candidate.exists():
            continue
        _add_path_if_matches(candidate)
        try:
            from piper_sdk import C_PiperInterface_V2, LogLevel

            return C_PiperInterface_V2, LogLevel
        except ModuleNotFoundError:
            continue

    searched = "\n- ".join(searched_paths)
    raise ModuleNotFoundError(
        "Could not import `piper_sdk`. Install it (for example: `pip install -e /path/to/piper_sdk`) "
        "or set PIPER_SDK_PATH. Searched paths:\n- "
        f"{searched}"
    )


def parse_piper_log_level(level_name: str) -> Any:
    _, log_level_enum = get_piper_sdk()
    normalized = level_name.upper()
    try:
        return getattr(log_level_enum, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Invalid Piper log level '{level_name}'. "
            "Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT."
        ) from exc
