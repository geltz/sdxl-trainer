from __future__ import annotations

from typing import List, Optional, Dict
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

from optimizer.lora import extract_lora_state_dict_comfy_peft


_LOCON_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
    "conv",
    "conv_in",
    "conv_out",
    "conv1",
    "conv2",
    "conv_shortcut",
]


def _ensure_peft_available() -> None:
    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed or failed to import. "
            "Install it with `pip install peft` before using inject_locon_peft."
        )


def inject_locon_peft(
    unet: nn.Module,
    config,
    target_modules: Optional[List[str]] = None,
    init_lora_weights: str | bool = "gaussian",
) -> nn.Module:
    _ensure_peft_available()

    if target_modules is None:
        target_modules = list(_LOCON_TARGET_MODULES)

    locon_config = LoraConfig(
        r=int(config.LOCON_RANK),
        lora_alpha=int(config.LOCON_ALPHA),
        lora_dropout=float(getattr(config, "LOCON_DROPOUT", 0.0)),
        target_modules=target_modules,
        bias="none",
        init_lora_weights=init_lora_weights,
    )

    peft_unet = get_peft_model(unet, locon_config)

    if not hasattr(peft_unet, "lora_config_for_export"):
        peft_unet.lora_config_for_export = locon_config

    trainable = [n for n, p in peft_unet.named_parameters() if p.requires_grad]
    print(f"[LoCon-PEFT] Number of trainable parameters: {len(trainable)}")
    if trainable:
        print("[LoCon-PEFT] First few trainable parameter keys:")
        for n in trainable[:8]:
            print(f"  - {n}")

    return peft_unet


def extract_locon_state_dict_comfy_peft(
    model: nn.Module,
    to_cpu: bool = True,
) -> Dict[str, torch.Tensor]:
    return extract_lora_state_dict_comfy_peft(
        model,
        key_prefix="locon_unet",
        to_cpu=to_cpu,
    )