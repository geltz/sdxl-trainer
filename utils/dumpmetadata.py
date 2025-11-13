#!/usr/bin/env python3
"""
dump_metadata.py

Dump metadata from a .safetensors file into a text file.

Usage:
    python dump_metadata.py model.safetensors
    python dump_metadata.py model.safetensors --out metadata.txt
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import safe_open


def maybe_pretty_json(value: str):
    """
    Some safetensors files store JSON *inside* a string.
    Try to parse it; if it works, return pretty JSON, else return original.
    """
    if not isinstance(value, str):
        return value
    value_str = value.strip()
    if not value_str:
        return value
    try:
        parsed = json.loads(value_str)
    except Exception:
        return value  # not JSON
    return parsed


def dump_metadata(in_path: Path, out_path: Path):
    with safe_open(str(in_path), framework="pt", device="cpu") as f:
        meta = f.metadata()

    if not meta:
        raise RuntimeError("This .safetensors file has no metadata at all.")

    # Try to expand any JSON-looking values
    expanded = {}
    for k, v in meta.items():
        expanded[k] = maybe_pretty_json(v)

    # Write as pretty JSON
    out_path.write_text(json.dumps(expanded, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadata written to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump ALL metadata from a .safetensors file to a text file."
    )
    parser.add_argument("model", help="Input .safetensors file")
    parser.add_argument(
        "--out",
        help="Output text file (default: <input>.metadata.json)",
    )
    args = parser.parse_args()

    in_path = Path(args.model)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_suffix(".metadata.json")

    dump_metadata(in_path, out_path)


if __name__ == "__main__":
    main()
