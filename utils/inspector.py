#!/usr/bin/env python3
"""
SDXL Model Inspector

Extracts training scheduler information and architecture hints from
Stable Diffusion .safetensors checkpoints.
"""

import torch
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from safetensors import safe_open

def load_metadata(model_path: Path) -> Dict[str, str]:
    """
    Safely load metadata from a .safetensors file.

    Returns an empty dict if no metadata is present or an error occurs.
    """
    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
    except Exception as exc:
        print(f"ERROR: Failed to read metadata: {exc}")
        return {}

    return metadata


def load_state_dict(model_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    """
    Safely load the full state dict from a .safetensors file.

    Returns None if loading fails.
    """
    try:
        state_dict: Dict[str, torch.Tensor] = {}
        # use safetensors.safe_open instead of safetensors.torch.load_file
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    except Exception as exc:
        print(f"ERROR: Failed to load weights from '{model_path}': {exc}")
        return None

    if not isinstance(state_dict, dict):
        print("ERROR: Unexpected state_dict type returned from safe_open.")
        return None

    return state_dict



def extract_scheduler_config(metadata: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Try to parse the scheduler configuration from common metadata keys.
    """
    candidate_keys = ["scheduler_config", "noise_scheduler_config", "scheduler"]
    scheduler_config: Optional[Dict[str, Any]] = None

    for key in candidate_keys:
        if key in metadata:
            raw_value = metadata[key]
            if isinstance(raw_value, str):
                try:
                    scheduler_config = json.loads(raw_value)
                    print(f"Found scheduler config in metadata key '{key}'.")
                    return scheduler_config
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from metadata key '{key}'.")

    # Fallback: search any string value with "scheduler" in the key name
    for key, value in metadata.items():
        if "scheduler" in key.lower() and isinstance(value, str):
            try:
                scheduler_config = json.loads(value)
                print(f"Found scheduler config in metadata key '{key}'.")
                return scheduler_config
            except json.JSONDecodeError:
                continue

    return None


def print_raw_metadata(metadata: Dict[str, str]) -> None:
    """
    Print the raw metadata with simple truncation for long values.
    """
    if not metadata:
        print("No metadata found in this checkpoint.")
        return

    print("=" * 70)
    print("raw metadata")
    print("=" * 70)

    for key, value in metadata.items():
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"{key}: {value_str}")


def print_scheduler_analysis(scheduler_config: Dict[str, Any]) -> None:
    """
    Print scheduler configuration and derived training recommendations.
    """
    print("\n" + "=" * 70)
    print("training scheduler settings")
    print("=" * 70)

    if not scheduler_config:
        print("\nNo scheduler configuration found in metadata.")
        print("This checkpoint may be using default SDXL scheduler settings.")
        print_default_sdxl_scheduler()
        return

    print("\nScheduler configuration:")
    print(json.dumps(scheduler_config, indent=2))

    print("\n" + "-" * 70)
    print("key training parameters")
    print("-" * 70)

    prediction_type = scheduler_config.get("prediction_type", "UNKNOWN")
    class_name = scheduler_config.get("_class_name", "UNKNOWN")
    num_train_timesteps = scheduler_config.get("num_train_timesteps", "UNKNOWN")
    beta_schedule = scheduler_config.get("beta_schedule", "UNKNOWN")
    beta_start = scheduler_config.get("beta_start", "UNKNOWN")
    beta_end = scheduler_config.get("beta_end", "UNKNOWN")
    zero_snr = scheduler_config.get("rescale_betas_zero_snr", False)

    print(f"  Prediction Type:     {prediction_type}")
    print(f"  Recommended Sampler: {class_name}")
    print(f"  Training Timesteps:  {num_train_timesteps}")
    print(f"  Beta Schedule:       {beta_schedule}")
    print(f"  Beta Range:          {beta_start} to {beta_end}")
    print(f"  Zero Terminal SNR:   {zero_snr}")

    training_scheduler = "DDIMScheduler"
    if isinstance(class_name, str) and "ddpm" in class_name.lower():
        training_scheduler = "DDPMScheduler"

    print("\n" + "=" * 70)
    print("training recommendations")
    print("=" * 70)
    print("\nSuggested configuration snippet for config.json:")
    print("-" * 70)
    print(
        "{\n"
        f'    "PREDICTION_TYPE": "{prediction_type}",\n'
        f'    "USE_ZERO_TERMINAL_SNR": {str(bool(zero_snr)).lower()},\n'
        "\n"
        f"    // Training Scheduler Settings (use {training_scheduler})\n"
        f"    \"NUM_TRAIN_TIMESTEPS\": {num_train_timesteps},\n"
        f'    "BETA_SCHEDULE": "{beta_schedule}",\n'
        f"    \"BETA_START\": {beta_start},\n"
        f"    \"BETA_END\": {beta_end}\n"
        "}\n"
    )

    if prediction_type == "v_prediction":
        print("\nAdditional recommendations for v-prediction models:")
        print("-" * 70)
        print(
            "{\n"
            '    "USE_MIN_SNR_GAMMA": true,\n'
            '    "MIN_SNR_GAMMA": 5.0,\n'
            '    "LEARNING_RATE": 1e-06,\n'
            '    "CLIP_GRAD_NORM": 0.5\n'
            "}\n"
        )
    elif prediction_type == "epsilon":
        print("\nAdditional recommendations for epsilon-prediction models:")
        print("-" * 70)
        print(
            "{\n"
            '    "LEARNING_RATE": 2e-06,\n'
            '    "CLIP_GRAD_NORM": 1.0\n'
            "}\n"
        )

def analyze_weights_only(
    model_path: Path, state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> None:
    """
    Analyze the model weights when no metadata is available.
    """
    print("\n" + "=" * 70)
    print("weight-only analysis")
    print("=" * 70)

    if state_dict is None:
        state_dict = load_state_dict(model_path)
        if state_dict is None:
            return

    vpred_indicators = ["v_pred", "velocity", "v_prediction"]
    has_vpred = any(
        any(indicator in key.lower() for indicator in vpred_indicators)
        for key in state_dict.keys()
    )

    if has_vpred:
        print("Heuristic: model appears to use v-prediction (based on parameter names).")
    else:
        print(
            "Heuristic: model likely uses epsilon prediction "
            "(standard stable diffusion) based on parameter names."
        )

    try:
        total_params = sum(
            tensor.numel() for tensor in state_dict.values() if tensor is not None
        )
        print(f"\nTotal parameters (all tensors): {total_params / 1e9:.2f}B")
        if 2.0e9 < total_params < 3.5e9:
            print("Approximate model type: SDXL (around 2.5B parameters).")
            print_default_sdxl_scheduler()
    except Exception as exc:
        print(f"Warning: could not compute parameter count: {exc}")


def analyze_architecture(
    model_path: Path, state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> None:
    """
    Analyze model architecture for additional insights.
    """
    print("\n" + "=" * 70)
    print("model architecture analysis")
    print("=" * 70)

    if state_dict is None:
        state_dict = load_state_dict(model_path)
        if state_dict is None:
            return

    unet_keys = [
        key for key in state_dict.keys() if "diffusion_model" in key or "unet" in key
    ]
    if not unet_keys:
        print(
            "Warning: no UNet-like keys were found. "
            "This may not be a standard UNet checkpoint."
        )
        return

    has_add_embedding = any("add_embedding" in key for key in unet_keys)
    has_label_emb = any("label_emb" in key for key in unet_keys)

    if has_add_embedding or has_label_emb:
        print("Heuristic: SDXL-style UNet detected (additional embeddings present).")
    else:
        print(
            "Heuristic: model may be SD 1.5/2.1 style "
            "(no obvious SDXL-specific embedding layers found)."
        )

    attention_keys = [key for key in unet_keys if "attn" in key.lower()]
    resnet_keys = [
        key
        for key in unet_keys
        if "resnet" in key.lower() or "conv" in key.lower()
    ]

    print("\nApproximate architecture components:")
    print(f"  Attention-related parameter groups: {len(attention_keys)}")
    print(f"  ResNet/Conv-related parameter groups: {len(resnet_keys)}")

    lora_keys = [key for key in state_dict.keys() if "lora" in key.lower()]
    if lora_keys:
        print(
            f"\nNotice: {len(lora_keys)} LoRA-related parameter groups detected. "
            "This checkpoint may contain LoRA weights rather than a full model."
        )


def inspect_model(model_path: Path) -> None:
    """
    Full inspection entrypoint for a .safetensors checkpoint.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        return

    if not model_path.is_file():
        print(f"ERROR: path is not a file: {model_path}")
        return

    if model_path.suffix.lower() != ".safetensors":
        print(
            f"Warning: expected a .safetensors file, got '{model_path.suffix}'. "
            "Continuing anyway."
        )

    print("=" * 70)
    print(f"inspecting: {model_path}")
    print("=" * 70)

    metadata = load_metadata(model_path)

    if not metadata:
        print(
            "\nNo metadata available. Falling back to weight-only analysis and "
            "architecture heuristics.\n"
        )
        state_dict = load_state_dict(model_path)
        if state_dict is None:
            return
        analyze_weights_only(model_path, state_dict=state_dict)
        analyze_architecture(model_path, state_dict=state_dict)
        print("\n" + "=" * 70)
        print("inspection complete")
        print("=" * 70)
        return

    print_raw_metadata(metadata)

    scheduler_config = extract_scheduler_config(metadata)
    if scheduler_config is not None:
        print_scheduler_analysis(scheduler_config)
    else:
        print_default_sdxl_scheduler()

    # Architecture analysis from weights
    analyze_architecture(model_path)

    print("\n" + "=" * 70)
    print("inspection complete")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Stable Diffusion / SDXL .safetensors checkpoint and "
            "print training scheduler and architecture information."
        )
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the .safetensors model file to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_model(args.model_path)


if __name__ == "__main__":
    main()
