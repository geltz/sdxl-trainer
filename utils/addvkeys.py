#!/usr/bin/env python3
"""
add_vpred_ztsnr.py

Add missing `v_pred` and `ztsnr` keys to an SDXL .safetensors model.

Usage:
    python addvkeys.py input.safetensors
    python addvkeys.py input.safetensors --out output.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


def add_vpred_ztsnr(in_path: Path, out_path: Path, vpred_val: float = 0.0, ztsnr_val: float = 0.0):
    tensors = {}

    # copy all existing tensors
    with safe_open(str(in_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # add v_pred if missing
    if "v_pred" not in tensors:
        tensors["v_pred"] = torch.tensor([float(vpred_val)])
        print("Added key: v_pred")
    else:
        print("Key already present: v_pred")

    # add ztsnr if missing
    if "ztsnr" not in tensors:
        tensors["ztsnr"] = torch.tensor([float(ztsnr_val)])
        print("Added key: ztsnr")
    else:
        print("Key already present: ztsnr")

    # save new file
    save_file(tensors, str(out_path))
    print(f"Saved patched model to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add v_pred/ztsnr keys to an SDXL .safetensors model."
    )
    parser.add_argument("model", help="Path to the input .safetensors file")
    parser.add_argument(
        "--out",
        help="Path to write the patched file (default: input name + .patched.safetensors)",
    )
    parser.add_argument(
        "--vpred", type=float, default=0.0, help="Value to store in v_pred (default: 0.0)"
    )
    parser.add_argument(
        "--ztsnr", type=float, default=0.0, help="Value to store in ztsnr (default: 0.0)"
    )

    args = parser.parse_args()

    in_path = Path(args.model)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_suffix(".patched.safetensors")

    add_vpred_ztsnr(in_path, out_path, args.vpred, args.ztsnr)


if __name__ == "__main__":
    main()

