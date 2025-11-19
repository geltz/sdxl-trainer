#!/usr/bin/env python3
"""
Thin CLI wrapper for the trainer.

Usage examples
--------------
python cli.py rftest.json
python cli.py rftest
python cli.py configs/rftest.json

This simply resolves the JSON config and then invokes:

    <python_exe> -u train.py --config <resolved_json>
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List


def _resolve_train_script(user_path: Optional[str]) -> Path:
    """
    Return the path to train.py.

    If user_path is None, train.py next to cli.py is used.
    If user_path is a directory, '<user_path>/train.py' is used.
    """
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_dir():
            p = p / "train.py"
    else:
        p = Path(__file__).resolve().parent / "train.py"

    if not p.is_file():
        raise SystemExit(f"ERROR: train.py not found at: {p}")

    return p


def _resolve_config(config_arg: str) -> Path:
    """
    Resolve the JSON config path with a few convenient fallbacks.

    Accepted forms:
      * Absolute or relative path to a JSON file
      * Bare name (e.g. 'rftest' or 'rftest.json'), in which case the
        following are tried:
          - ./<name>.json
          - ./configs/<name>.json
    """
    base = Path(config_arg).expanduser()
    candidates: List[Path] = []

    # If user included .json, keep it; otherwise add the suffix.
    if base.suffix.lower() == ".json":
        candidates.append(base)
    else:
        candidates.append(base.with_suffix(".json"))

    here = Path(__file__).resolve().parent

    # For bare names or relative paths, also try next to cli.py and in ./configs
    more_candidates: List[Path] = []
    for cand in candidates:
        if not cand.is_absolute():
            more_candidates.append(here / cand)
            more_candidates.append(here / "configs" / cand.name)

    candidates.extend(more_candidates)

    # Deduplicate while preserving order
    seen = set()
    unique_candidates: List[Path] = []
    for cand in candidates:
        try:
            key = cand.resolve()
        except OSError:
            key = cand
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        if cand.is_file():
            return cand

    msg_lines = [
        f"ERROR: Could not find config JSON for {config_arg!r}.",
        "Tried:",
    ]
    msg_lines.extend(f"  - {c}" for c in unique_candidates)
    raise SystemExit("\n".join(msg_lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI wrapper that runs train.py using a JSON configuration file.",
    )
    parser.add_argument(
        "config",
        help=(
            "Path or name of the JSON config. "
            "Examples: './rftest.json', 'configs/rftest.json', or just 'rftest'."
        ),
    )
    parser.add_argument(
        "--train-script",
        dest="train_script",
        default=None,
        help="Optional path to train.py (defaults to train.py next to cli.py).",
    )
    parser.add_argument(
        "--python",
        dest="python_exe",
        default=sys.executable,
        help="Python executable to use (defaults to the current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit without starting training.",
    )

    args = parser.parse_args()

    train_py = _resolve_train_script(args.train_script)
    config_path = _resolve_config(args.config)

    python_exe = args.python_exe
    if python_exe != sys.executable and shutil.which(python_exe) is None:
        parser.error(f"Specified Python executable not found: {python_exe}")

    cmd = [python_exe, "-u", str(train_py), "--config", str(config_path)]

    print(f"[cli] train.py : {train_py}")
    print(f"[cli] config   : {config_path}")
    print(f"[cli] python   : {python_exe}")
    print(f"[cli] working  : {train_py.parent}")
    print(f"[cli] command  : {' '.join(cmd)}")

    if args.dry_run:
        return

    env = os.environ.copy()
    proc = subprocess.Popen(cmd, cwd=str(train_py.parent), env=env)

    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        print("\n[cli] Interrupt received, terminating training process...")
        proc.terminate()
        proc.wait()
        exit_code = proc.returncode

    if exit_code is None:
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
